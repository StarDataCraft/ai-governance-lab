from __future__ import annotations
import networkx as nx
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans


class GovernanceGraph:

    def __init__(self):
        self.G = nx.DiGraph()
        self.risk_embeddings = {}
        self.similarity_threshold = 0.75  # 提高阈值

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

    # -----------------------------
    # Centrality
    # -----------------------------

    def compute_centrality(self):

        degree = nx.degree_centrality(self.G)
        between = nx.betweenness_centrality(self.G)

        return {
            "degree": degree,
            "betweenness": between
        }

    # -----------------------------
    # Risk clustering
    # -----------------------------

    def cluster_risks(self, n_clusters=2):

        if len(self.risk_embeddings) < n_clusters:
            return {}

        risk_ids = list(self.risk_embeddings.keys())
        emb_matrix = np.array(
            [self.risk_embeddings[r] for r in risk_ids]
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(emb_matrix)

        clusters = {}
        for rid, label in zip(risk_ids, labels):
            clusters.setdefault(int(label), []).append(rid)

        return clusters
