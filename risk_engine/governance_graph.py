# risk_engine/governance_graph.py

from __future__ import annotations
import networkx as nx
from typing import List, Dict, Tuple
import numpy as np


class GovernanceGraph:

    def __init__(self):
        self.G = nx.DiGraph()
        self.risk_embeddings = {}
        self.similarity_threshold = 0.55

    def add_risk(self, risk_id: str, description: str, embedding: np.ndarray):
        self.G.add_node(risk_id, type="risk", description=description)
        self.risk_embeddings[risk_id] = embedding

    def add_clause(self, clause_id: str, title: str):
        self.G.add_node(clause_id, type="clause", title=title)

    def add_control(self, control_id: str, description: str):
        self.G.add_node(control_id, type="control", description=description)

    def link_risk_to_control(self, risk_id: str, control_id: str):
        self.G.add_edge(risk_id, control_id, relation="requires")

    def map_clause_to_risks(
        self,
        clause_id: str,
        clause_embedding: np.ndarray,
    ) -> List[Tuple[str, float]]:

        activated = []

        for risk_id, risk_emb in self.risk_embeddings.items():
            sim = float(np.dot(clause_embedding, risk_emb))

            if sim > self.similarity_threshold:
                self.G.add_edge(clause_id, risk_id, relation="addresses")
                activated.append((risk_id, sim))

        return activated

    def propagate(self) -> Dict:

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

        return {
            "risks": list(activated_risks),
            "controls": list(activated_controls),
        }
