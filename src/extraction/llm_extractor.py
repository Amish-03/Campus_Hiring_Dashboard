"""
LLM-based structured information extractor for campus hiring emails.

Uses Mistral-7B-Instruct-v0.3 (4-bit quantized) to parse emails into
structured JSON with company, role, CTC, CGPA, dates, and more.
"""

import json
import re
import datetime
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..models import StructuredHiringData

# ── Model Configuration ──────────────────────────────────────

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

# ── Prompt Template ──────────────────────────────────────────

EXTRACTION_PROMPT = """<s>[INST] You are a data extraction assistant for a university placement office. Given a campus placement email, extract structured information as JSON.

STRICT RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no extra text.
2. Use null for any field that is missing or cannot be determined.
3. CTC must be in LPA (lakhs per annum as a float). If over 100, set to null.
4. CGPA cutoff must be a float between 0 and 10. If not mentioned, set to null.
5. All dates must be ISO format (YYYY-MM-DD). If the year is ambiguous, assume 2025/2026 based on context.
6. company_name: Look for company names in quotes in the subject line, or extract from body. Never guess.
7. eligibility_branches: comma-separated list (e.g., "CSE, ECE, EEE"). null if not mentioned.
8. selection_count: number of students selected (from "X students selected"). null if not mentioned.
9. role: the specific job title offered. null if not clear.
10. Do NOT hallucinate or invent any information.

EMAIL SUBJECT: {subject}

EMAIL BODY (first 2000 chars):
{body}

Output this exact JSON schema:
{{
  "company_name": "string",
  "role": "string or null",
  "ctc_lpa": "float or null",
  "cgpa_cutoff": "float or null",
  "eligibility_branches": "string or null",
  "registration_deadline": "YYYY-MM-DD or null",
  "test_date": "YYYY-MM-DD or null",
  "interview_date": "YYYY-MM-DD or null",
  "selection_count": "int or null",
  "total_openings": "int or null"
}}
[/INST]"""


class LLMExtractor:
    """Extracts structured hiring data from emails using a local LLM."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the quantized model into GPU memory."""
        print(f"Loading model: {self.model_id}")
        print("Configuring 4-bit quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {self.model.device}")

    def _build_prompt(self, subject: str, body: str) -> str:
        """Build the extraction prompt from email subject and body."""
        # Truncate body to 2000 chars to fit context window
        truncated_body = body[:2000] if body else "(empty)"
        return EXTRACTION_PROMPT.format(subject=subject, body=truncated_body)

    def _parse_json_response(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from the model's raw output."""
        # Try to find JSON in the output
        # Strategy 1: find JSON block between { and }
        json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: try the entire output
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 3: find JSON between ```json and ```
        json_block = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def extract(self, email_id: str, subject: str, body: str) -> Optional[StructuredHiringData]:
        """Extract structured data from a single email."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self._build_prompt(subject, body)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse the JSON response
        parsed = self._parse_json_response(raw_output)
        if parsed is None:
            print(f"  [WARN] Failed to parse JSON for email: {subject[:50]}")
            return None

        # Build StructuredHiringData from parsed JSON
        try:
            return StructuredHiringData(
                email_id=email_id,
                company_name=parsed.get("company_name", "Unknown"),
                role=parsed.get("role"),
                ctc_lpa=self._safe_float(parsed.get("ctc_lpa")),
                cgpa_cutoff=self._safe_float(parsed.get("cgpa_cutoff")),
                eligibility_branches=parsed.get("eligibility_branches"),
                registration_deadline=parsed.get("registration_deadline"),
                test_date=parsed.get("test_date"),
                interview_date=parsed.get("interview_date"),
                selection_count=self._safe_int(parsed.get("selection_count")),
                total_openings=self._safe_int(parsed.get("total_openings")),
                model_used=self.model_id,
            )
        except Exception as e:
            print(f"  [WARN] Failed to build StructuredHiringData: {e}")
            return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded, GPU memory freed.")
