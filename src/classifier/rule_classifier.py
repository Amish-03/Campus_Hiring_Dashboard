from typing import Dict

class HiringClassifier:
    def __init__(self, threshold: int = 15):
        # Weighted keywords for classification
        self.keywords: Dict[str, int] = {
            "hiring": 10,
            "recruitment": 10,
            "job opportunity": 15,
            "campus placement": 20,
            "internship": 5,
            "selection process": 10,
            "registration link": 10,
            "eligibility criteria": 10,
            "ctc": 10,
            "cgpa": 10,
            "placement drive": 20,
            "on-campus": 15,
            "off-campus": 10,
            "pool campus": 15
        }
        self.threshold = threshold

    def calculate_score(self, text: str) -> int:
        score = 0
        text_lower = text.lower()
        for word, weight in self.keywords.items():
            if word in text_lower:
                score += weight
        return score

    def is_hiring_email(self, sender: str, subject: str, body: str) -> bool:
        # Strict sender check
        if "placement_officer@kletech.ac.in" not in sender.lower():
            return False
            
        # Give more weight to subject matches
        subject_score = self.calculate_score(subject) * 2
        body_score = self.calculate_score(body)
        
        total_score = subject_score + body_score
        return total_score >= self.threshold
