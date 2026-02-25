"""
LLM-based structured information extractor for campus hiring emails.

Uses Mistral-7B-Instruct-v0.3 (4-bit quantized via bitsandbytes) with a
placement-officer-aware prompt template and JSON retry logic.
"""

import json
import re
import datetime
import logging
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..models import StructuredHiringData

logger = logging.getLogger(__name__)

# ── Model Configuration ──────────────────────────────────────

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 256
MAX_RETRIES = 3

# ── JSON Schema for validation ───────────────────────────────

EXPECTED_KEYS = {
    "is_hiring_email", "company_name", "role", "ctc_lpa", "cgpa_cutoff",
    "eligibility_branches", "registration_deadline",
    "test_date", "interview_date", "selection_count", "total_openings"
}

# ── Prompt Template ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise classification and data extraction assistant for a university placement office.
You MUST output ONLY valid JSON. No explanations. No markdown. No extra text.
You MUST NOT guess, infer, or hallucinate any information.
If a field is not explicitly mentioned in the email, you MUST set it to null."""

EXTRACTION_PROMPT = """<s>[INST] {system}

CONTEXT:
- All emails are sent by a university placement officer (not by companies directly).
- The officer forwards company announcements, reminders, shortlists, and selection results.
- Multiple emails may refer to the same company drive.
- Company names are often in the subject line inside quotes (e.g., "NVIDIA" or "Rossell Techsys").

CLASSIFICATION + EXTRACTION RULES:

0. is_hiring_email (REQUIRED — decide FIRST):
   - Set to true if the email is about a campus hiring drive, placement opportunity,
     job opening, internship opportunity, recruitment, or selection results.
   - Set to false if the email is about general notices, circulars, academic matters,
     event invitations, club activities, holidays, or anything unrelated to hiring.
   - If is_hiring_email is false, set ALL other fields to null.

1. company_name:
   - FIRST: Check the subject line for a name in quotes — prioritize this.
   - SECOND: Look for company name early in the email body.
   - If unclear or ambiguous, return null. NEVER guess.

2. role:
   - Extract the specific job title if mentioned (e.g., "Software Engineer", "ASIC Design Engineer").
   - If only a generic term like "hiring" is used, return null.

3. ctc_lpa:
   - Extract ONLY if a numeric CTC/salary is explicitly stated.
   - Convert to LPA (lakhs per annum) as a float.
   - If value > 100, return null (unrealistic for campus hiring).
   - Do NOT infer CTC from context.

4. cgpa_cutoff:
   - Extract ONLY if a numeric CGPA (0–10 scale) is explicitly mentioned.
   - If a percentage (e.g., "60%") is mentioned instead, return null.
   - Do NOT convert percentage to CGPA.

5. Dates (registration_deadline, test_date, interview_date):
   - Assign registration_deadline ONLY if near words: "register", "apply", "deadline", "last date".
   - Assign test_date ONLY if near words: "test", "assessment", "online test", "aptitude".
   - Assign interview_date ONLY if near words: "interview", "HR round", "technical round".
   - Convert all dates to ISO format: YYYY-MM-DD.
   - If year is ambiguous, assume 2025. If date is ambiguous, return null.

6. eligibility_branches:
   - Comma-separated branch list (e.g., "CSE, ECE, EEE").
   - null if not mentioned.

7. selection_count:
   - Extract ONLY if an explicit number of selected students is stated.
   - Do NOT count names or infer from lists.

8. total_openings:
   - Extract only if explicitly mentioned. null otherwise.

EMAIL SUBJECT: {subject}

EMAIL BODY:
{body}

Output ONLY this JSON (no other text):
{{"is_hiring_email": true_or_false, "company_name": str_or_null, "role": str_or_null, "ctc_lpa": float_or_null, "cgpa_cutoff": float_or_null, "eligibility_branches": str_or_null, "registration_deadline": "YYYY-MM-DD_or_null", "test_date": "YYYY-MM-DD_or_null", "interview_date": "YYYY-MM-DD_or_null", "selection_count": int_or_null, "total_openings": int_or_null}}
[/INST]"""

RETRY_HINT = "\n\n[INST] Your previous response was not valid JSON. Output ONLY a single valid JSON object with the schema shown above. No text before or after the JSON. [/INST]"


class LLMExtractor:
    """Extracts structured hiring data from emails using a local LLM."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        self._stats = {"success": 0, "failed": 0, "retries": 0}

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    def load_model(self):
        """Load the quantized model into GPU memory."""
        print(f"  Loading model: {self.model_id}")
        print("  Configuring 4-bit NF4 quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  Model loaded on {self.model.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    def _build_prompt(self, subject: str, body: str) -> str:
        """Build the extraction prompt."""
        truncated_body = body[:1500] if body else "(empty email body)"
        return EXTRACTION_PROMPT.format(
            system=SYSTEM_PROMPT,
            subject=subject,
            body=truncated_body
        )

    def _generate(self, prompt: str) -> str:
        """Run inference and return raw text output."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _parse_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from model output with multiple strategies."""

        # Strategy 1: Find the outermost { ... } block
        brace_depth = 0
        start_idx = None
        for i, ch in enumerate(raw_output):
            if ch == '{':
                if start_idx is None:
                    start_idx = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    try:
                        return json.loads(raw_output[start_idx:i+1])
                    except json.JSONDecodeError:
                        break

        # Strategy 2: Regex for simple JSON block
        json_match = re.search(r'\{[^{}]*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try entire output
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 4: ```json blocks
        json_block = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_output, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_schema(self, parsed: Dict[str, Any]) -> bool:
        """Check that parsed JSON has the expected keys."""
        return "is_hiring_email" in parsed

    def extract(self, email_id: str, subject: str, body: str) -> Optional[StructuredHiringData]:
        """
        Classify and extract structured data from a single email.
        Returns None if LLM classifies it as non-hiring OR if JSON parse fails.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = self._build_prompt(subject, body)

        for attempt in range(MAX_RETRIES):
            raw_output = self._generate(prompt)
            parsed = self._parse_json(raw_output)

            if parsed and self._validate_schema(parsed):
                # Check LLM classification
                is_hiring = parsed.get("is_hiring_email")
                if is_hiring is False or str(is_hiring).lower() == "false":
                    self._stats["skipped_non_hiring"] = self._stats.get("skipped_non_hiring", 0) + 1
                    return None  # LLM says not a hiring email

                self._stats["success"] += 1
                if attempt > 0:
                    self._stats["retries"] += attempt
                return self._build_result(email_id, parsed)

            # Add retry hint to prompt
            if attempt < MAX_RETRIES - 1:
                prompt = prompt + RETRY_HINT
                logger.debug(f"Retry {attempt+1} for: {subject[:50]}")

        # All retries failed
        self._stats["failed"] += 1
        safe_subj = subject.encode('ascii', errors='replace').decode('ascii')
        logger.warning(f"JSON parse failed after {MAX_RETRIES} attempts: {safe_subj[:60]}")
        return None

    def _build_result(self, email_id: str, parsed: Dict[str, Any]) -> StructuredHiringData:
        """Convert parsed JSON dict into StructuredHiringData."""
        return StructuredHiringData(
            email_id=email_id,
            company_name=parsed.get("company_name") or "Unknown",
            role=parsed.get("role"),
            ctc_lpa=self._safe_float(parsed.get("ctc_lpa")),
            cgpa_cutoff=self._safe_float(parsed.get("cgpa_cutoff")),
            eligibility_branches=parsed.get("eligibility_branches"),
            registration_deadline=self._safe_str(parsed.get("registration_deadline")),
            test_date=self._safe_str(parsed.get("test_date")),
            interview_date=self._safe_str(parsed.get("interview_date")),
            selection_count=self._safe_int(parsed.get("selection_count")),
            total_openings=self._safe_int(parsed.get("total_openings")),
            model_used=self.model_id,
        )

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        if value is None or value == "null":
            return None
        try:
            v = float(value)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        if value is None or value == "null":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_str(value) -> Optional[str]:
        if value is None or str(value).lower() in ("null", "none", ""):
            return None
        return str(value).strip()

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  Model unloaded, GPU memory freed.")
