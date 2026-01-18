import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def build_neighbors(edges: List[Dict]) -> Dict[str, List[str]]:
    """
    Build an undirected neighbors list from directed edges.
    For SME risk, we care that components are connected, not direction.
    """
    neighbor_sets: Dict[str, Set[str]] = {}

    for e in edges:
        s = e["source"]
        t = e["target"]

        if s not in neighbor_sets:
            neighbor_sets[s] = set()
        if t not in neighbor_sets:
            neighbor_sets[t] = set()

        # undirected neighborhood: s↔t
        neighbor_sets[s].add(t)
        neighbor_sets[t].add(s)

    # convert sets to sorted lists for JSON
    return {nid: sorted(list(neis)) for nid, neis in neighbor_sets.items()}


def make_payload(
    graph_path: Path,
    out_path: Path,
    top_n: int = 20,
    metric: str = "betweenness",
) -> None:
    """
    Load the full graph JSON, select top_n nodes by metric, and write a
    compact JSON payload suitable for SME risk rating in ChatGPT.

    Output schema:

    {
      "nodes": [
        {
          "id": "C36",
          "name": "...",
          "type": "software",
          "description": "...",
          "degree": 15,
          "betweenness": 0.081,
          "closeness": 0.198,
          "neighbors": ["C21", "C51", "C39", ...]
        },
        ...
      ]
    }
    """
    data = json.loads(graph_path.read_text(encoding="utf-8"))

    nodes = data["nodes"]
    edges = data["edges"]

    neighbors = build_neighbors(edges)

    # Attach metric value and sort
    if metric not in ("degree", "betweenness", "closeness"):
        raise ValueError(f"metric must be degree|betweenness|closeness, got {metric}")

    for n in nodes:
        feats = n.get("features", {})
        n["_metric_value"] = float(feats.get(metric, 0.0))

    # Sort descending by chosen metric
    nodes_sorted = sorted(nodes, key=lambda n: n["_metric_value"], reverse=True)

    # Take top N
    selected = nodes_sorted[:top_n]

    compact_nodes = []
    for n in selected:
        nid = n["id"]
        feats = n.get("features", {})
        compact_nodes.append(
            {
                "id": nid,
                "name": n.get("name", ""),
                "type": n.get("type", ""),
                "description": n.get("description", ""),
                "degree": feats.get("degree"),
                "betweenness": feats.get("betweenness"),
                "closeness": feats.get("closeness"),
                # neighbor IDs only (no risk labels, to avoid biasing SME)
                "neighbors": neighbors.get(nid, []),
            }
        )

    payload = {"nodes": compact_nodes}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SME payload with {len(compact_nodes)} nodes → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create compact SME payload from system graph JSON"
    )
    parser.add_argument(
        "--graph-json",
        required=True,
        help="Path to warehouse_graph_with_gnn.json (or similar)",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output JSON payload file (e.g., sme_payload_top20.json)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many highest-ranked nodes to include (default: 20)",
    )
    parser.add_argument(
        "--metric",
        choices=["degree", "betweenness", "closeness"],
        default="betweenness",
        help="Centrality metric used to rank nodes (default: betweenness)",
    )

    args = parser.parse_args()

    make_payload(
        graph_path=Path(args.graph_json),
        out_path=Path(args.out_json),
        top_n=args.top_n,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
