import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import csv

def load_chunk_graph(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def merge_chunks(chunk_json_files: List[Path]) -> Dict[str, Any]:
    """
    Merge multiple chunk-level graphs.

    Strategy:
    - Ignore chunk-local IDs, use component 'name' (case-insensitive) as key.
    - Create new global IDs C1, C2, ... based on unique names.
    - Map edges' source/target via local id -> name -> global id.
    """
    global_nodes: Dict[str, Dict[str, Any]] = {}  # name_lower -> node dict (with global_id)
    name_to_global_id: Dict[str, str] = {}
    global_edges: List[Dict[str, Any]] = []

    next_id = 1

    for f in chunk_json_files:
        data = load_chunk_graph(f)
        # map local id -> name for this chunk
        local_id_to_name: Dict[str, str] = {}

        for n in data.get("nodes", []):
            local_id = n.get("id")
            name = n.get("name", "").strip()
            if not name:
                continue
            name_lower = name.lower()

            if name_lower not in global_nodes:
                global_id = f"C{next_id}"
                next_id += 1
                node_copy = dict(n)
                node_copy["id"] = global_id
                global_nodes[name_lower] = node_copy
                name_to_global_id[name_lower] = global_id
            else:
                pass

            if local_id:
                local_id_to_name[local_id] = name

        for e in data.get("edges", []):
            src_local = e.get("source")
            tgt_local = e.get("target")
            if not src_local or not tgt_local:
                continue
            src_name = local_id_to_name.get(src_local)
            tgt_name = local_id_to_name.get(tgt_local)
            if not src_name or not tgt_name:
                continue

            src_global = name_to_global_id[src_name.lower()]
            tgt_global = name_to_global_id[tgt_name.lower()]

            if src_global == tgt_global:
                continue

            global_edges.append({
                "source": src_global,
                "target": tgt_global,
                "relation": e.get("relation", ""),
                "rationale": e.get("rationale", "")
            })

    nodes_list = list(global_nodes.values())
    return {"nodes": nodes_list, "edges": global_edges}

def build_graph(data: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in data["nodes"]:
        nid = n["id"]
        G.add_node(nid, **n)
    for e in data["edges"]:
        G.add_edge(e["source"], e["target"], **e)
    return G

def compute_features(G: nx.DiGraph) -> None:
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)

    for n in G.nodes():
        G.nodes[n]["degree"] = float(deg[n])
        G.nodes[n]["betweenness"] = float(bet[n])
        G.nodes[n]["closeness"] = float(clo[n])

def assign_heuristic_risk(G: nx.DiGraph) -> None:
    bet_values = [G.nodes[n]["betweenness"] for n in G.nodes()]
    deg_values = [G.nodes[n]["degree"] for n in G.nodes()]
    if not bet_values:
        return

    bet_sorted = sorted(bet_values)
    deg_sorted = sorted(deg_values)
    bet_p75 = bet_sorted[int(0.75 * (len(bet_sorted) - 1))]
    bet_p50 = bet_sorted[int(0.50 * (len(bet_sorted) - 1))]
    deg_p75 = deg_sorted[int(0.75 * (len(deg_sorted) - 1))]
    deg_p25 = deg_sorted[int(0.25 * (len(deg_sorted) - 1))]

    for n in G.nodes():
        b = G.nodes[n]["betweenness"]
        d = G.nodes[n]["degree"]

        if (b >= bet_p75) or (d >= deg_p75):
            risk = "high"
        elif (b <= bet_p50 and d <= deg_p25):
            risk = "low"
        else:
            risk = "medium"

        G.nodes[n]["risk_heuristic"] = risk


def export_json_for_node(G: nx.DiGraph, out_json: Path) -> None:
    nodes = []
    edges = []
    for nid, attrs in G.nodes(data=True):
        nodes.append({
            "id": nid,
            "name": attrs.get("name", nid),
            "type": attrs.get("type", "other"),
            "description": attrs.get("description", ""),
            "features": {
                "degree": attrs.get("degree"),
                "betweenness": attrs.get("betweenness"),
                "closeness": attrs.get("closeness"),
            },
            "risk_heuristic": attrs.get("risk_heuristic"),
            "risk_gnn": attrs.get("risk_gnn"), 
            "risk_scores": attrs.get("risk_scores"),  
        })

    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relation": attrs.get("relation", ""),
            "rationale": attrs.get("rationale", "")
        })

    data = {"nodes": nodes, "edges": edges}
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote Node.js JSON graph to {out_json}")

def export_gephi_csv(G: nx.DiGraph, basepath: Path) -> None:
    nodes_path = basepath.with_name(basepath.stem + "_nodes.csv")
    edges_path = basepath.with_name(basepath.stem + "_edges.csv")

    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Label"])
        for nid, attrs in G.nodes(data=True):
            label = attrs.get("name", nid)
            w.writerow([nid, label])

    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target", "Type", "Label"])
        for u, v, attrs in G.edges(data=True):
            w.writerow([u, v, "Directed", attrs.get("relation", "")])

    print(f"Wrote Gephi CSVs: {nodes_path.name}, {edges_path.name}")

def export_neo4j_csv(G: nx.DiGraph, basepath: Path) -> None:
    """
    Export nodes and edges in a Neo4j-friendly CSV format.

    Nodes CSV columns:
      id,name,type,description,degree,betweenness,closeness,risk_heuristic,risk_gnn

    Edges CSV columns:
      source,target,relation,rationale
    """
    nodes_path = basepath.with_name(basepath.stem + "_neo4j_nodes.csv")
    edges_path = basepath.with_name(basepath.stem + "_neo4j_edges.csv")

    import csv

    # Nodes
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "name", "type", "description",
            "degree", "betweenness", "closeness",
            "risk_heuristic", "risk_gnn"
        ])
        for nid, attrs in G.nodes(data=True):
            feats = attrs.get("features", {})
            w.writerow([
                nid,
                attrs.get("name", nid),
                attrs.get("type", "other"),
                attrs.get("description", ""),
                attrs.get("degree", ""),
                attrs.get("betweenness", ""),
                attrs.get("closeness", ""),
                attrs.get("risk_heuristic", ""),
                attrs.get("risk_gnn", ""),
            ])

    # Edges
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "relation", "rationale"])
        for u, v, attrs in G.edges(data=True):
            w.writerow([
                u,
                v,
                attrs.get("relation", ""),
                attrs.get("rationale", ""),
            ])

    print(f"Wrote Neo4j CSVs: {nodes_path.name}, {edges_path.name}")

def visualize(G: nx.DiGraph, out_png: Path) -> None:
    risk_to_val = {"low": 0.2, "medium": 0.5, "high": 0.9}
    color_vals = [
        risk_to_val.get(G.nodes[n].get("risk_heuristic", "low"), 0.2)
        for n in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color_vals,
        cmap=plt.cm.coolwarm,
        edge_color="gray",
        node_size=1500,
        font_size=7,
        ax=ax,
    )
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    plt.title("System Graph Colored by Heuristic Risk")
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"Saved visualization to {out_png}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge LLM chunk JSON outputs into a risk graph and export for Node.js & Gephi."
    )
    parser.add_argument("--chunks-dir", required=True, help="Directory with chunk_XX_graph.json files")
    parser.add_argument("--out-json", required=True, help="Output Node.js JSON graph path")
    parser.add_argument("--export-gephi", action="store_true", help="Export *_nodes.csv and *_edges.csv")
    parser.add_argument("--visualize", action="store_true", help="Create PNG visualization")
    args = parser.parse_args()

    cdir = Path(args.chunks_dir)
    chunk_files = sorted(cdir.glob("chunk_*_graph.json"))
    if not chunk_files:
        raise SystemExit(f"No chunk_*_graph.json files found in {cdir}")

    merged = merge_chunks(chunk_files)
    G = build_graph(merged)
    compute_features(G)
    assign_heuristic_risk(G)

    out_json = Path(args.out_json)
    export_json_for_node(G, out_json)
    export_neo4j_csv(G, out_json)

    if args.export_gephi:
        export_gephi_csv(G, out_json)

    if args.visualize:
        visualize(G, out_json.with_suffix(".png"))

if __name__ == "__main__":
    main()
