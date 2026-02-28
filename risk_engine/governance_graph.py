from __future__ import annotations
import networkx as nx
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans


class GovernanceGraph:

    def __init__(self):
        self.G = nx.DiGraph()
        self.risk_embeddings = {}
        self.risk_weights = {
            "privacy_risk": 3,
            "security_risk": 2,
            "hallucination_risk": 2
        }

    def add_risk(self, risk_id: str, description: str, embedding: np.ndarray):
        self.G.add_node(risk_id, type="risk", description=description)
        self.risk_embeddings[risk_id] = embedding

    def add_clause(self, clause_id: str, title: str):
        self.G.add_node(clause_id, type="clause", title=title)

    def add_control(self, control_id: str, description: str):
        self.G.add_node(control_id, type="control", description=description)

    def link_risk_to_control(self, risk_id: str, control_id: str):
        self.G.add_edge(risk_id, control_id, relation="requires")

    # -----------------------------
    # Similarity matrix
    # -----------------------------

    def compute_similarity_matrix(self, clause_embeddings):

        risk_ids = list(self.risk_embeddings.keys())
        risk_matrix = np.array([self.risk_embeddings[r] for r in risk_ids])

        sim_matrix = clause_embeddings @ risk_matrix.T

        return risk_ids, sim_matrix

    # -----------------------------
    # Dynamic threshold
    # -----------------------------

    def dynamic_threshold(self, sim_matrix):

        mean = np.mean(sim_matrix)
        std = np.std(sim_matrix)

        return float(mean + 0.5 * std)

    # -----------------------------
    # Risk activation
    # -----------------------------

    def activate_risks(
        self,
        clause_ids,
        clause_embeddings,
        threshold
    ):

        risk_ids, sim_matrix = self.compute_similarity_matrix(clause_embeddings)

        activated = {}
        explanations = []

        for i, cid in enumerate(clause_ids):
            for j, rid in enumerate(risk_ids):

                sim = float(sim_matrix[i, j])

                if sim > threshold:

                    self.G.add_edge(cid, rid, relation="addresses")

                    activated.setdefault(rid, []).append((cid, sim))

                    explanations.append(
                        f"{cid} triggered {rid} (similarity={sim:.3f})"
                    )

        return activated, explanations

    # -----------------------------
    # Propagation
    # -----------------------------

    def propagate(self):

        activated_risks = set()
        activated_controls = set()

        for node in self.G.nodes:
            if self.G.nodes[node]["type"] == "clause":
                for neighbor in self.G.successors(node):
                    if self.G.nodes[neighbor]["type"] == "risk":
                        activated_risks.add(neighbor)

                        for control in self.G.successors(neighbor):
                            if self.G.nodes[control]["type"] == "control":
                                activated_controls.add(control)

        return activated_risks, activated_controls

    # -----------------------------
    # Weighted risk score
    # -----------------------------

    def weighted_risk_score(self, activated):

        score = 0

        for rid, entries in activated.items():
            weight = self.risk_weights.get(rid, 1)
            for _, sim in entries:
                score += sim * weight

        level = "LOW"
        if score > 6:
            level = "MEDIUM"
        if score > 12:
            level = "HIGH"

        return float(score), level

    # -----------------------------
    # Executive summary generator
    # -----------------------------

    def generate_executive_summary(
        self,
        activated: Dict,
        activated_controls: set,
        missing_controls: set,
        score: float,
        level: str,
        threshold: float,
    ) -> str:

        lines = []

        lines.append("Executive Risk Assessment Summary\n")

        lines.append(
            f"The dynamic similarity threshold applied was {threshold:.3f}, "
            "which filtered clause-risk relationships based on relative semantic strength."
        )

        if activated:
            lines.append(
                f"\nA total of {len(activated)} primary risk domains were activated."
            )

            for rid, entries in activated.items():
                weight = self.risk_weights.get(rid, 1)

                avg_sim = sum(sim for _, sim in entries) / len(entries)

                lines.append(
                    f"- {rid.replace('_', ' ').title()} "
                    f"(avg similarity {avg_sim:.3f}, severity weight {weight})"
                )

        else:
            lines.append(
                "\nNo material risk domains exceeded the semantic activation threshold."
            )

        lines.append(
            f"\nThe weighted composite risk score is {score:.2f}, "
            f"resulting in an overall risk level classified as {level}."
        )

        coverage_ratio = (
            len(activated_controls) /
            (len(activated_controls) + len(missing_controls))
            if (len(activated_controls) + len(missing_controls)) > 0 else 0
        )

        lines.append(
            f"\nControl coverage ratio is {coverage_ratio:.2f}. "
        )

        if missing_controls:
            lines.append(
                "The following governance controls are currently not triggered "
                f"and may represent residual exposure: {', '.join(missing_controls)}."
            )
        else:
            lines.append(
                "All defined governance controls are activated, "
                "indicating comprehensive mitigation coverage."
            )

        if level == "LOW":
            lines.append(
                "\nInterpretation: The structural risk profile is currently limited "
                "and largely mitigated by existing controls."
            )
        elif level == "MEDIUM":
            lines.append(
                "\nInterpretation: The risk profile warrants enhanced monitoring "
                "and potentially additional governance safeguards."
            )
        else:
            lines.append(
                "\nInterpretation: The system exhibits elevated structural risk. "
                "Immediate governance reinforcement and oversight escalation are recommended."
            )

        return "\n".join(lines)
