# risk_engine/governance_graph.py

from __future__ import annotations
import networkx as nx
from typing import List, Dict


class GovernanceGraph:

    def __init__(self):
        self.G = nx.DiGraph()

    def add_risk(self, risk_id: str, description: str):
        self.G.add_node(risk_id, type="risk", description=description)

    def add_clause(self, clause_id: str, title: str):
        self.G.add_node(clause_id, type="clause", title=title)

    def add_control(self, control_id: str, description: str):
        self.G.add_node(control_id, type="control", description=description)

    def link_clause_to_risk(self, clause_id: str, risk_id: str):
        self.G.add_edge(clause_id, risk_id, relation="addresses")

    def link_risk_to_control(self, risk_id: str, control_id: str):
        self.G.add_edge(risk_id, control_id, relation="requires")

    def propagate_from_clauses(self, clause_ids: List[str]) -> Dict:

        activated_risks = set()
        activated_controls = set()

        for cid in clause_ids:
            if cid not in self.G:
                continue

            for neighbor in self.G.successors(cid):
                if self.G.nodes[neighbor]["type"] == "risk":
                    activated_risks.add(neighbor)

                    for control in self.G.successors(neighbor):
                        if self.G.nodes[control]["type"] == "control":
                            activated_controls.add(control)

        return {
            "risks": list(activated_risks),
            "controls": list(activated_controls),
        }
