from ..storage.db import DatabaseManager
from typing import Dict, Any, List

class AnalyticsEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def compute_metrics(self) -> Dict[str, Any]:
        details = self.db.get_hiring_details()
        if not details:
            return {"error": "No data available"}

        # CTC metrics
        ctcs = [d['ctc'] for d in details if d['ctc'] is not None]
        avg_ctc = sum(ctcs) / len(ctcs) if ctcs else 0
        max_ctc = max(ctcs) if ctcs else 0

        # CGPA metrics
        cgpas = [d['cgpa_cutoff'] for d in details if d['cgpa_cutoff'] is not None]
        avg_cgpa = sum(cgpas) / len(cgpas) if cgpas else 0

        # Hiring frequency by month
        months = {}
        for d in details:
            if d['extracted_at']:
                month = d['extracted_at'][:7] # YYYY-MM
                months[month] = months.get(month, 0) + 1

        return {
            "total_hiring_emails": len(details),
            "average_ctc": round(avg_ctc, 2),
            "highest_ctc": max_ctc,
            "average_cgpa_cutoff": round(avg_cgpa, 2),
            "hiring_frequency": months
        }

    def print_report(self):
        metrics = self.compute_metrics()
        print("\n=== Campus Hiring Analytics Report ===")
        print(f"Total Companies Found: {metrics.get('total_hiring_emails')}")
        print(f"Average CTC: {metrics.get('average_ctc')} LPA")
        print(f"Highest CTC: {metrics.get('highest_ctc')} LPA")
        print(f"Average CGPA Cutoff: {metrics.get('average_cgpa_cutoff')}")
        print("\nHiring Frequency by Month:")
        for month, count in metrics.get('hiring_frequency', {}).items():
            print(f"  {month}: {count} emails")
        print("======================================")
