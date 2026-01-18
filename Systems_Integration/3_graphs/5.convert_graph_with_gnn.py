import argparse
import json
import csv
from pathlib import Path


def export_gephi_from_json(graph_path: Path, out_prefix: Path) -> None:
    """
    Read a system_graph_with_gnn.json-style file and export
    Gephi-friendly nodes/edges CSVs that include risk_gnn.

    Nodes CSV columns:
      Id,Label,risk_heuristic,risk_gnn,degree,betweenness,closeness,
      score_low,score_medium,score_high

    Edges CSV columns:
      Source,Target,Type,Label
    """
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    edges = data["edges"]

    nodes_path = out_prefix.with_name(out_prefix.stem + "_nodes.csv")
    edges_path = out_prefix.with_name(out_prefix.stem + "_edges.csv")

    # ----- Nodes -----
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Id",
            "Label",
            "risk_heuristic",
            "risk_gnn",
            "degree",
            "betweenness",
            "closeness",
            "score_low",
            "score_medium",
            "score_high",
        ])
        for n in nodes:
            feats = n.get("features", {})
            scores = n.get("risk_scores") or {}
            w.writerow([
                n["id"],
                n.get("name", n["id"]),
                n.get("risk_heuristic", ""),
                n.get("risk_gnn", ""),
                feats.get("degree", ""),
                feats.get("betweenness", ""),
                feats.get("closeness", ""),
                scores.get("low", ""),
                scores.get("medium", ""),
                scores.get("high", ""),
            ])

    # ----- Edges -----
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target", "Type", "Label"])
        for e in edges:
            w.writerow([
                e["source"],
                e["target"],
                "Directed",
                e.get("relation", ""),
            ])

    print(f"Wrote Gephi nodes CSV: {nodes_path}")
    print(f"Wrote Gephi edges CSV: {edges_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Gephi CSVs from a *_graph_with_gnn.json file."
    )
    parser.add_argument(
        "--graph-json",
        required=True,
        help="Path to warehouse_graph_with_gnn.json",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Base path for CSVs (e.g. warehouse_gnn_gephi)",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_json)
    out_prefix = Path(args.out_prefix)
    export_gephi_from_json(graph_path, out_prefix)


if __name__ == "__main__":
    main()
