"""
Evaluation framework for LLM extraction accuracy.

Compares LLM output against manually annotated ground truth to compute
field-level precision, recall, JSON validity rate, and hallucination rate.
"""

import json
import os
import random
import logging
from typing import List, Dict, Any, Optional

from ..storage.db import DatabaseManager

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

EVAL_FIELDS = [
    "company_name", "role", "ctc_lpa", "cgpa_cutoff",
    "eligibility_branches", "registration_deadline",
    "test_date", "interview_date", "selection_count", "total_openings"
]


class Evaluator:
    """Evaluates LLM extraction quality against ground truth."""

    def __init__(self, db: DatabaseManager, ground_truth_path: str = GROUND_TRUTH_PATH):
        self.db = db
        self.gt_path = ground_truth_path

    def sample_emails(self, n: int = 20, seed: int = 42) -> List[dict]:
        """Sample n hiring emails for manual annotation."""
        all_data = self.db.get_structured_data()
        if not all_data:
            print("No structured data available for sampling.")
            return []

        random.seed(seed)
        sample = random.sample(all_data, min(n, len(all_data)))

        # Create ground truth template
        template = []
        for record in sample:
            entry = {
                "email_id": record["email_id"],
                "subject": record.get("subject", ""),
                "llm_output": {f: record.get(f) for f in EVAL_FIELDS},
                "ground_truth": {f: None for f in EVAL_FIELDS},
                "notes": ""
            }
            template.append(entry)

        # Save template
        with open(self.gt_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"\n  Ground truth template saved to: {self.gt_path}")
        print(f"  Sampled {len(template)} emails for annotation.")
        print(f"  → Fill in the 'ground_truth' fields manually, then run evaluation.")
        return template

    def load_ground_truth(self) -> List[dict]:
        """Load annotated ground truth."""
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(
                f"Ground truth file not found: {self.gt_path}\n"
                f"Run 'python -m src.main --sample-eval' first to generate the template."
            )
        with open(self.gt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self) -> Dict[str, Any]:
        """
        Compare LLM output vs ground truth.
        Returns evaluation metrics.
        """
        entries = self.load_ground_truth()

        # Check if ground truth has been filled in
        annotated = [e for e in entries if any(v is not None for v in e["ground_truth"].values())]
        if not annotated:
            print("  Ground truth not annotated yet. Please fill in the 'ground_truth' fields.")
            return {}

        total_fields = 0
        correct = 0
        hallucinated = 0
        missed = 0
        json_valid = 0

        field_stats: Dict[str, Dict[str, int]] = {f: {"tp": 0, "fp": 0, "fn": 0} for f in EVAL_FIELDS}

        for entry in annotated:
            llm = entry["llm_output"]
            gt = entry["ground_truth"]

            # Check if LLM output is valid JSON (it was parsed, so yes)
            json_valid += 1

            for field in EVAL_FIELDS:
                llm_val = self._normalize(llm.get(field))
                gt_val = self._normalize(gt.get(field))

                total_fields += 1

                if gt_val is None and llm_val is None:
                    # Both null — true negative (correct)
                    correct += 1
                elif gt_val is not None and llm_val is not None:
                    if self._match(field, llm_val, gt_val):
                        # Correct extraction
                        correct += 1
                        field_stats[field]["tp"] += 1
                    else:
                        # Wrong value (hallucination)
                        hallucinated += 1
                        field_stats[field]["fp"] += 1
                elif gt_val is not None and llm_val is None:
                    # Missed extraction
                    missed += 1
                    field_stats[field]["fn"] += 1
                elif gt_val is None and llm_val is not None:
                    # Hallucinated field
                    hallucinated += 1
                    field_stats[field]["fp"] += 1

        # Compute aggregate metrics
        precision_denom = correct + hallucinated
        recall_denom = correct + missed

        results = {
            "total_entries": len(annotated),
            "total_fields_evaluated": total_fields,
            "correct": correct,
            "hallucinated": hallucinated,
            "missed": missed,
            "accuracy": round(correct / total_fields, 4) if total_fields > 0 else 0,
            "precision": round(correct / precision_denom, 4) if precision_denom > 0 else 0,
            "recall": round(correct / recall_denom, 4) if recall_denom > 0 else 0,
            "json_validity_rate": round(json_valid / len(annotated), 4) if annotated else 0,
            "hallucination_rate": round(hallucinated / total_fields, 4) if total_fields > 0 else 0,
            "field_stats": {},
        }

        # Per-field precision/recall
        for field, stats in field_stats.items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            results["field_stats"][field] = {
                "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else None,
                "tp": tp, "fp": fp, "fn": fn,
            }

        return results

    def print_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        if not results:
            return

        print("\n" + "=" * 60)
        print("  EVALUATION REPORT")
        print("=" * 60)
        print(f"  Entries evaluated:    {results['total_entries']}")
        print(f"  Fields evaluated:     {results['total_fields_evaluated']}")
        print(f"  Correct:              {results['correct']}")
        print(f"  Hallucinated:         {results['hallucinated']}")
        print(f"  Missed:               {results['missed']}")
        print(f"  Overall Accuracy:     {results['accuracy']:.1%}")
        print(f"  Overall Precision:    {results['precision']:.1%}")
        print(f"  Overall Recall:       {results['recall']:.1%}")
        print(f"  JSON Validity Rate:   {results['json_validity_rate']:.1%}")
        print(f"  Hallucination Rate:   {results['hallucination_rate']:.1%}")

        print(f"\n  {'Field':<25} {'Prec':<8} {'Recall':<8} {'TP':<5} {'FP':<5} {'FN':<5}")
        print(f"  {'-'*56}")
        for field, stats in results["field_stats"].items():
            p = f"{stats['precision']:.1%}" if stats['precision'] is not None else "—"
            r = f"{stats['recall']:.1%}" if stats['recall'] is not None else "—"
            print(f"  {field:<25} {p:<8} {r:<8} {stats['tp']:<5} {stats['fp']:<5} {stats['fn']:<5}")

        print("=" * 60)

    @staticmethod
    def _normalize(value) -> Optional[str]:
        """Normalize a value for comparison."""
        if value is None:
            return None
        s = str(value).strip().lower()
        if s in ("null", "none", "n/a", "unknown", ""):
            return None
        return s

    @staticmethod
    def _match(field: str, llm_val: str, gt_val: str) -> bool:
        """Field-aware matching."""
        if field in ("ctc_lpa", "cgpa_cutoff"):
            try:
                return abs(float(llm_val) - float(gt_val)) < 0.5
            except ValueError:
                return False
        if field in ("selection_count", "total_openings"):
            try:
                return int(float(llm_val)) == int(float(gt_val))
            except ValueError:
                return False
        # String fields: fuzzy containment match
        return llm_val in gt_val or gt_val in llm_val
