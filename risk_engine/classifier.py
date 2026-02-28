class RiskClassifier:

    def __init__(self):
        self.risk_categories = [
            "Bias",
            "Privacy",
            "Security",
            "Compliance",
            "Operational"
        ]

    def classify(self, description: str):
        """
        Very naive placeholder classification logic.
        """
        description = description.lower()

        if "personal data" in description:
            return "Privacy"

        if "credit" in description or "loan" in description:
            return "Compliance"

        return "Operational"
