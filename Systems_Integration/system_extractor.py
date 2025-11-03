# system_extractor.py
# Updated to filter pronouns like 'they' and avoid tight_layout warning

import argparse
import json
import re
import csv
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

import spacy
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# Stopword loader

def load_stopwords(fp: Path) -> Set[str]:
    return set(
        line.strip().lower() for line in fp.read_text("utf-8").splitlines() if line.strip()
    )

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_components_and_edges(text: str, stopwords: Set[str]) -> Dict:
    doc = nlp(text)
    components = set()
    edges = set()

    for chunk in doc.noun_chunks:
        chunk_words = [w.lemma_.lower() for w in chunk if not w.is_punct]
        # Exclude noun chunks that are just pronouns or contain any stopwords
        if len(chunk_words) > 1 and all(w not in stopwords and not nlp(w)[0].pos_ == "PRON" for w in chunk_words):
            components.add(chunk.text.strip())

    for sent in doc.sents:
        subj = obj = None
        verb = None

        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.text
            if token.dep_ == "dobj":
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_

        if subj and obj:
            edges.add((subj.strip(), obj.strip(), verb))

    return {
        "nodes": sorted(components),
        "edges": [
            {"source": s, "target": t, "relation": v} for (s, t, v) in edges if s != t
        ],
    }

def write_csv_for_gephi(data: Dict, basepath: Path):
    nodes_path = basepath.with_name(basepath.stem + "_nodes.csv")
    edges_path = basepath.with_name(basepath.stem + "_edges.csv")

    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        for node in data["nodes"]:
            writer.writerow([node, node])

    with edges_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target", "Type", "Label"])
        for edge in data["edges"]:
            writer.writerow([edge["source"], edge["target"], "Directed", edge["relation"]])

    print(f"CSV exported for Gephi: {nodes_path.name}, {edges_path.name}")

def visualize_graph(json_path: Path, out_img: Path):
    data = json.loads(json_path.read_text("utf-8"))
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node)
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], relation=edge["relation"])

    centrality = nx.degree_centrality(G)
    color_vals = [centrality[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42, k=0.6/len(G.nodes()))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color_vals,
        cmap=plt.cm.coolwarm,
        edge_color="gray",
        node_size=1800,
        font_size=7,
        ax=ax
    )
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    plt.title("System Graph Colored by Degree Centrality")
    fig.tight_layout()
    fig.savefig(out_img)
    print(f"Graph visualization saved to {out_img}")

def main():
    parser = argparse.ArgumentParser(description="System Description Extractor")
    parser.add_argument("--input", required=True, help="Input .txt system description")
    parser.add_argument("--stopwords", required=True, help="Stopword file")
    parser.add_argument("--domain-stopwords", nargs="*", default=[], help="Domain-specific stopwords")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--visualize", action="store_true", help="Create a PNG of the graph")
    parser.add_argument("--export-csv", action="store_true", help="Export Gephi-compatible CSV files")
    args = parser.parse_args()

    raw = Path(args.input).read_text("utf-8")
    stopwords = load_stopwords(Path(args.stopwords))
    stopwords.update(w.lower() for w in args.domain_stopwords)
    data = extract_components_and_edges(normalize(raw), stopwords)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Extracted {len(data['nodes'])} components and {len(data['edges'])} edges → {args.out}")

    if args.visualize:
        img_path = out_path.with_suffix(".png")
        visualize_graph(out_path, img_path)

    if args.export_csv:
        write_csv_for_gephi(data, out_path)

if __name__ == "__main__":
    main()