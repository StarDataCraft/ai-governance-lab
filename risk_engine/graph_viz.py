# risk_engine/graph_viz.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import math

import networkx as nx
import matplotlib.pyplot as plt

# Two simple palettes (color-blind friendlier than random)
PALETTES = {
    "soft": {
        "clause": "#4C78A8",
        "risk": "#F58518",
        "control": "#54A24B",
        "edge": "#666666",
        "label": "#222222",
        "bg": "#FFFFFF",
    },
    "contrast": {
        "clause": "#1F77B4",
        "risk": "#D62728",
        "control": "#2CA02C",
        "edge": "#000000",
        "label": "#111111",
        "bg": "#FFFFFF",
    },
}

def _node_type(n: str, node_data: Dict[str, Any]) -> str:
    # You can adapt this if your graph stores type explicitly.
    # Priority: explicit node_data["type"] else heuristics.
    if isinstance(node_data, dict) and node_data.get("type"):
        return str(node_data["type"])
    s = str(n)
    if s.startswith(("EU-", "JP-", "AU-", "UK-", "US-", "CA-", "GDPR-", "OECD-")):
        return "clause"
    if s.endswith("_risk") or "risk" in s:
        return "risk"
    return "control"

def draw_governance_graph(
    G: nx.Graph,
    palette_name: str = "soft",
    node_alpha: float = 0.92,
    edge_alpha: float = 0.55,
    edge_width_scale: float = 3.0,
    seed: int = 7,
) -> plt.Figure:
    pal = PALETTES.get(palette_name, PALETTES["soft"])

    fig = plt.figure(figsize=(9.2, 5.4))
    ax = plt.gca()
    ax.set_facecolor(pal["bg"])
    ax.axis("off")

    # layout
    pos = nx.spring_layout(G, seed=seed, k=None)

    # group nodes by type
    clause_nodes, risk_nodes, control_nodes = [], [], []
    for n, d in G.nodes(data=True):
        nt = _node_type(n, d)
        if nt == "clause":
            clause_nodes.append(n)
        elif nt == "risk":
            risk_nodes.append(n)
        else:
            control_nodes.append(n)

    # edge widths by weight if present
    widths = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        widths.append(max(0.6, edge_width_scale * (0.6 + math.sqrt(max(0.0, w)))))

    nx.draw_networkx_edges(
        G, pos,
        alpha=edge_alpha,
        width=widths,
        edge_color=pal["edge"],
        ax=ax,
    )

    # different shapes per type
    nx.draw_networkx_nodes(
        G, pos, nodelist=clause_nodes,
        node_color=pal["clause"],
        node_shape="s",
        node_size=900,
        alpha=node_alpha,
        linewidths=0.8,
        edgecolors="#FFFFFF",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=risk_nodes,
        node_color=pal["risk"],
        node_shape="o",
        node_size=820,
        alpha=node_alpha,
        linewidths=0.8,
        edgecolors="#FFFFFF",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=control_nodes,
        node_color=pal["control"],
        node_shape="^",
        node_size=820,
        alpha=node_alpha,
        linewidths=0.8,
        edgecolors="#FFFFFF",
        ax=ax,
    )

    # labels (small)
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color=pal["label"],
        ax=ax,
    )

    return fig
