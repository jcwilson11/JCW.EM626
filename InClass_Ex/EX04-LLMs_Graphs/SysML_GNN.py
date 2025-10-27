# ADDED: reproducibility seeds & small runtime hygiene 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ADDED
import random
import numpy as np
import torch
import xml.etree.ElementTree as ET  

SEED = 42  # ADDED
random.seed(SEED)            
np.random.seed(SEED)         
torch.manual_seed(SEED)      

# ADDED: imports for CLI + YAML + lxml 
import argparse  
import yaml      
from lxml import etree  

import xml.etree.ElementTree as ET
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
# ADDED: metrics for proper evaluation
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # ADDED
import itertools
import copy

# ADDED: Lightweight SpecXMLAdapter implemented inline to avoid adding new files
class SpecXMLAdapter:  
    """
    Namespace-aware XPath adapter driven by YAML mapping.
    parse(xml_path) -> (nodes, edges)
      nodes: list of dicts with keys: id, name, type, documentation (others allowed)
      edges: list of tuples (src, dst, attrs_dict) where attrs_dict may include 'type'
    """
    def __init__(self, mapping: dict, nsmap: dict | None = None):  
        # ADDED: minimal mapping validation for helpful errors
        required = [
            ("nodes", "select"), ("nodes", "id"),
            ("edges", "select"), ("edges", "src"), ("edges", "dst")
        ]
        for sect, key in required:
            if sect not in mapping or key not in mapping[sect]:
                raise ValueError(f"SpecXMLAdapter mapping missing '{sect}.{key}'")
        self.m = mapping
        self.ns = nsmap or {}

    def _str_first(self, el, xp: str) -> str:  
        vals = el.xpath(xp, namespaces=self.ns)
        if not vals:
            return ""
        v = vals[0]
        return (v if isinstance(v, str) else getattr(v, "text", "") or "").strip()

    def _str_join(self, el, xp: str) -> str:  
        vals = el.xpath(xp, namespaces=self.ns)
        out = []
        for v in vals:
            if isinstance(v, str):
                out.append(v.strip())
            else:
                out.append((getattr(v, "text", "") or "").strip())
        return " ".join([s for s in out if s])

    def parse(self, xml_path: str):  
        tree = etree.parse(xml_path)
        root = tree.getroot()

        # Nodes
        ncfg = self.m["nodes"]
        node_elems = root.xpath(ncfg["select"], namespaces=self.ns)
        nodes = []
        for idx, el in enumerate(node_elems):  # ADDED: stable fallback id synth
            nid   = self._str_first(el, ncfg["id"]) or f"specnode_{idx}"  # ADDED (fallback id)
            ntype = self._str_first(el, ncfg.get("type", "string('Generic')")) or "Generic"
            attrs = {}
            for k, xp in (ncfg.get("attrs") or {}).items():
                attrs[k] = self._str_join(el, xp)
            # standardize for downstream usage
            name = attrs.get("name", "").strip() or nid or "Unnamed"
            documentation = attrs.get("documentation", "")
            node = {"id": nid, "name": name, "type": ntype, "documentation": documentation}
            # keep any extra attrs
            for k, v in attrs.items():
                if k not in node:
                    node[k] = v
            nodes.append(node)

        # Edges
        ecfg = self.m["edges"]
        edge_elems = root.xpath(ecfg["select"], namespaces=self.ns)
        edges = []
        for el in edge_elems:
            src  = self._str_first(el, ecfg["src"])
            dst  = self._str_first(el, ecfg["dst"])
            etyp = self._str_first(el, ecfg.get("type", "string('relatedTo')")) or "relatedTo"
            attrs = {"relation": etyp}
            for k, xp in (ecfg.get("attrs") or {}).items():
                attrs[k] = self._str_join(el, xp)
            edges.append((src, dst, attrs))

        # drop edges referencing missing nodes
        ids = {n["id"] for n in nodes}
        edges = [(s, d, a) for (s, d, a) in edges if s in ids and d in ids]
        return nodes, edges
# === END ADDED SpecXMLAdapter ===

#  ADDED: CLI to select between original SysML XML and new spec XML 
_cli = argparse.ArgumentParser(description="SysML / SpecXML GNN pipeline (backward compatible)")  
_cli.add_argument("--input", default="sysml_graph.xml", help="Path to XML input (default: sysml_graph.xml)")  
_cli.add_argument("--input-type", choices=["sysml", "specxml"], default="sysml", help="Input format")  
_cli.add_argument("--mapping", help="YAML mapping file (required for --input-type specxml)")  
_cli.add_argument("--namespaces", help="YAML prefix->URI map for XML namespaces (optional)")  
args, _ = _cli.parse_known_args()  
is_specxml = (args.input_type == "specxml")  
# === END ADDED CLI ===

# === Step 1: Parse XML ===
# REPLACED: hardcoded path with CLI arg; preserved original branch for sysml.
if not is_specxml:  
    tree = ET.parse(args.input)  # REPLACED: "sysml_graph.xml" -> args.input
    root = tree.getroot()
else:
    # ADDED: load mapping + namespaces and parse via SpecXMLAdapter
    if not args.mapping:
        raise SystemExit("--mapping is required when --input-type specxml")
    mapping = yaml.safe_load(open(args.mapping, "r"))
    nsmap = yaml.safe_load(open(args.namespaces, "r")) if args.namespaces else {}
    # ADDED: log effective namespaces
    if nsmap:
        print("Namespaces:", nsmap)  
    adapter = SpecXMLAdapter(mapping, nsmap)
    spec_nodes, spec_edges = adapter.parse(args.input)  

G = nx.DiGraph()

# === Step 2: Load MiniLM ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# FIX: mask-aware mean pooling to avoid padding bias
def get_embedding(text):  
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        token_emb = outputs.last_hidden_state            
        mask = inputs["attention_mask"].unsqueeze(-1)    
        summed = (token_emb * mask).sum(dim=1)           
        counts = mask.sum(dim=1).clamp(min=1)            
        return (summed / counts).squeeze(0).numpy()
    except Exception:
        return np.zeros(384)

# Step 3: Parse Nodes and Build type_map
# REPLACED: type_set computation supports both sysml (original) and specxml (new).
type_set = set()
if not is_specxml:  # original path (unchanged)
    for node in (root.findall(".//node") if not is_specxml else []):
        t = node.attrib.get("type")
        if t:
            type_set.add(t)
else:  # ADDED: collect types from spec adapter nodes
    for n in spec_nodes:
        t = n.get("type")
        if t:
            type_set.add(t)

# Ensure 'Generic' is always present
type_set.add('Generic')
type_map = {t: i for i, t in enumerate(sorted(type_set))}
rev_type_map = {v: k for k, v in type_map.items()}

# parse nodes for id, name, type, and documentation
if not is_specxml:
    # === ORIGINAL NODE LOOP (unchanged) ===
    for node in root.findall(".//node"):
        nid = node.attrib.get("id")
        name = node.attrib.get("name", "").strip()
        if not name:
            name = "Unnamed"

        t = node.attrib.get("type", "Generic")
        doc = node.attrib.get("documentation", "")
        text = f"{name} {doc}".strip()

        emb = get_embedding(text)
        label = type_map.get(t, type_map["Generic"])

        G.add_node(nid,
                   name=name,
                   type=t,
                   documentation=doc,
                   x=emb.tolist(),
                   y=label,
                   label=name)  # Set label for PyVis and other graph use
else:
    # ADDED: NODE LOOP for specxml path; mirrors original behavior
    for n in spec_nodes:
        nid = n.get("id")
        name = (n.get("name") or "").strip() or "Unnamed"
        t = n.get("type", "Generic")
        doc = n.get("documentation", "")
        text = f"{name} {doc}".strip()

        emb = get_embedding(text)
        label = type_map.get(t, type_map["Generic"])

        # Keep extra attributes if present
        attrs = {k: v for k, v in n.items() if k not in {"id", "name", "type", "documentation"}}
        G.add_node(nid,
                   name=name,
                   type=t,
                   documentation=doc,
                   x=emb.tolist(),
                   y=label,
                   label=name,
                   **attrs)

# === Step 4: Parse Edges ===
if not is_specxml:
    # FIX: prefer SysML 'label' (exported by many tools), fallback to 'type'
    for edge in root.findall(".//edge"):  
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        relation = edge.attrib.get("label") or edge.attrib.get("type") or "relatedTo"  
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, relation=relation)
else:
    # ADDED: EDGE LOOP for specxml path; mirrors original behavior 
    for (source, target, edata) in spec_edges:
        if source in G.nodes and target in G.nodes:
            # normalize attribute key to 'relation' for consistency with original downstream code
            relation = edata.get("relation") or edata.get("type") or "relatedTo"
            edata = {**edata, "relation": relation}
            G.add_edge(source, target, **edata)

print("✅ Parsed Nodes:", len(G.nodes))
print("✅ Parsed Edges:", len(G.edges))

#  ADDED: Basic Graph Checks
stats = {
    "num_nodes": G.number_of_nodes(),
    "num_edges": G.number_of_edges(),
    "num_types": len({d.get("type", "Generic") for _, d in G.nodes(data=True)}),
    "weak_components": nx.number_weakly_connected_components(G),
    "self_loops": nx.number_of_selfloops(G),
}
orphans = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0]
try:
    cycles = list(itertools.islice(nx.simple_cycles(G), 50))  # cap for speed
except nx.NetworkXNoCycle:
    cycles = []
pd.DataFrame([stats]).to_csv("graph_stats.csv", index=False)
pd.Series(orphans, name="orphan_node_id").to_csv("orphans.csv", index=False)
pd.Series([len(c) for c in cycles], name="cycle_lengths").to_csv("cycles.csv", index=False)

# Copy 'name' into 'label' for each node
for node, data in G.nodes(data=True):
    name_value = data.get("name", "")
    data["label"] = str(name_value) if name_value is not None else ""

# === Step 5: Save the Graph ===
G_export = copy.deepcopy(G)

# STEP 5.1: Forcefully replace 'label' with 'name'
for node_id, data in G_export.nodes(data=True):
    name_val = data.get("name", "").strip()
    data["label"] = name_val if name_val else "Unnamed"

# STEP 5.2: Clean up types in node attributes
for _, data in G_export.nodes(data=True):
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, list):
            data[k] = ", ".join(map(str, v))
        elif isinstance(v, (dict, tuple, set)):
            data[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            del data[k]

# STEP 5.3: Clean up types in edge attributes
for _, _, data in G_export.edges(data=True):
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, list):
            data[k] = ", ".join(map(str, v))
        elif isinstance(v, (dict, tuple, set)):
            data[k] = str(v)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            del data[k]

for node, data in G_export.nodes(data=True):
    if "x" in data:
        del data["x"]
for u, v, data in G_export.edges(data=True):
    if "x" in data:
        del data["x"]

print("Sample node labels:")
for node_id, data in list(G_export.nodes(data=True))[:5]:
    print(f"{node_id}: label = '{data.get('label')}' | name = '{data.get('name')}'")

# STEP 5.4: Save as GraphML
nx.write_graphml(G_export, "sysml_model_graph.graphml")

# STEP 5.5 — PyVis Interactive HTML
net = Network(height="750px", width="100%", directed=True, notebook=False)
net.from_nx(G_export)

for node in net.nodes:
    meta = G_export.nodes[node["id"]]
    tooltip = f"Name: {meta.get('name', '')}<br>Type: {meta.get('type', '')}"
    if 'documentation' in meta and meta['documentation']:
        tooltip += f"<br><br>{meta['documentation']}"
    node["title"] = tooltip
    node["label"] = meta.get("name", "Unnamed")
    node["shape"] = "dot"
    node["size"] = 15

for edge in net.edges:
    edge["arrows"] = "to"
    if "relation" in edge:
        edge["title"] = edge["relation"]

net.write_html("sysml_model_graph_interactive.html")

# === Step 6: GNN with PyTorch Geometric ===
data = from_networkx(G)
data.x = torch.tensor([G.nodes[n]['x'] for n in G.nodes], dtype=torch.float)
data.y = torch.tensor([G.nodes[n]['y'] for n in G.nodes], dtype=torch.long)

print("Classes:", len(type_map))
print("Feature shape:", data.x.shape)

# ADDED: train/val/test split without changing overall training structure 
num_nodes = data.x.size(0)  # ADDED
perm = torch.randperm(num_nodes)  
train_end = int(0.6 * num_nodes)  
val_end = int(0.8 * num_nodes)    
train_mask = torch.zeros(num_nodes, dtype=torch.bool)  
val_mask   = torch.zeros(num_nodes, dtype=torch.bool)  
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)  
train_mask[perm[:train_end]] = True                    
val_mask[perm[train_end:val_end]] = True              
test_mask[perm[val_end:]] = True                     

# === Step 7: Simple GCN ===
W1 = GCNConv(data.x.shape[1], 64)
W2 = GCNConv(64, len(type_map))
params = list(W1.parameters()) + list(W2.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

best_val_acc = 0.0   
best_state = None    
for epoch in range(50):
    optimizer.zero_grad()
    h = W1(data.x, data.edge_index)
    h = F.relu(h)
    out = W2(h, data.edge_index)
    # FIX: compute loss only on train split
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])  # FIX
    loss.backward()
    optimizer.step()

    # ADDED: track validation accuracy
    with torch.no_grad():
        val_pred = out[val_mask].argmax(dim=1)
        val_acc = (val_pred == data.y[val_mask]).float().mean().item() if val_mask.any() else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "W1": copy.deepcopy(W1.state_dict()),
                "W2": copy.deepcopy(W2.state_dict())
            }

# ADDED: restore best validation state before final eval
if best_state is not None:  
    W1.load_state_dict(best_state["W1"])
    W2.load_state_dict(best_state["W2"])

W1.eval()
W2.eval()
with torch.no_grad():
    h = F.relu(W1(data.x, data.edge_index))
    logits = W2(h, data.edge_index)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    embeddings = logits.cpu().numpy()

#  ADDED: Test metrics & save reports 
true_test = data.y[test_mask].cpu().numpy() if test_mask.any() else np.array([])
pred_test = preds[test_mask.cpu().numpy()] if test_mask.any() else np.array([])

if test_mask.any():  # ADDED
    # ADDED: derive class subset that appears in test labels
    unique_labels = np.unique(true_test)
    label_names = [rev_type_map[int(i)] for i in unique_labels]

    acc = accuracy_score(true_test, pred_test)
    f1m = f1_score(true_test, pred_test, average="macro")

    rep = classification_report(
        true_test,
        pred_test,
        labels=unique_labels,           # FIX: restrict to present classes
        target_names=label_names,       # FIX: correct label names only
        zero_division=0
    )

    cm = confusion_matrix(true_test, pred_test, labels=unique_labels)

    with open("gnn_test_report.txt", "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {f1m:.4f}\n\n")
        f.write(rep + "\n")
        f.write("Confusion Matrix:\n")
        f.write(pd.DataFrame(cm,
                             index=label_names,
                             columns=label_names).to_string())

    print(f"Test Acc={acc:.4f}, MacroF1={f1m:.4f}")
else:
    print("Not enough nodes for a test split; metrics skipped.")


# === Step 8: Save Embeddings & Predictions ===================================
labels = [G.nodes[n]['name'] for n in G.nodes]
types = [rev_type_map[y] for y in data.y.numpy()]
df = pd.DataFrame(embeddings, index=labels)
df['type'] = types
df.to_csv("sysml_node_embeddings.csv")

# ADDED: save per-node predictions with confidence (softmax max)
with torch.no_grad():  
    probs = F.softmax(torch.tensor(embeddings), dim=1).numpy()
conf = probs.max(axis=1) 
node_ids = list(G.nodes)  
pred_types = [rev_type_map[int(i)] for i in preds]  
pred_df = pd.DataFrame({
    "node_id": node_ids,
    "name": labels,
    "true_type": types,
    "pred_type": pred_types,
    "confidence": conf
})
pred_df.to_csv("node_predictions.csv", index=False)  

# === Step 9: Visualize with t-SNE and PCA ===
if embeddings.shape[0] > 2 and embeddings.shape[1] > 2:
    # FIX: robust perplexity for small-N
    N = embeddings.shape[0]                                   
    perplexity = max(5, min(30, (N - 1) // 3))               
    tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=SEED).fit_transform(embeddings)  
    pca_result = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)  # ADDED seed

    tsne_df = pd.DataFrame({'x': tsne_result[:,0], 'y': tsne_result[:,1], 'label': types, 'node': labels})
    pca_df = pd.DataFrame({'x': pca_result[:,0], 'y': pca_result[:,1], 'label': types, 'node': labels})

    px.scatter(tsne_df, x="x", y="y", color="label", hover_name="node", title="t-SNE").write_html("tsne_visualization.html")
    px.scatter(pca_df, x="x", y="y", color="label", hover_name="node", title="PCA").write_html("pca_visualization.html")
else:
    print("Not enough samples/features for t-SNE or PCA")

# === Step 10: Link Prediction via Cosine Similarity ===
similarity_matrix = cosine_similarity(embeddings)
threshold = 0.95  
existing_edges = set(G.edges())
existing_edges_rev = existing_edges | {(v, u) for (u, v) in existing_edges}  # ADDED

predicted_edges = []
node_list = list(G.nodes)
for i, j in itertools.combinations(range(len(node_list)), 2):
    # FIX: skip if either direction already exists
    if (node_list[i], node_list[j]) in existing_edges_rev or (node_list[j], node_list[i]) in existing_edges_rev:  # FIX
        continue
    sim = similarity_matrix[i, j]
    if sim > threshold:
        # ADDED: emit both directions explicitly so downstream can decide directionality
        predicted_edges.append((node_list[i], node_list[j], sim, "i->j"))  # ADDED
        predicted_edges.append((node_list[j], node_list[i], sim, "j->i"))  # ADDED

link_df = pd.DataFrame(predicted_edges, columns=["source", "target", "similarity", "direction"])  # ADDED column
link_df.to_csv("predicted_links.csv", index=False)

print("Script completed.")

def parse_sysml_graph(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    nodes = []
    edges = []

    for node in root.findall('node'):
        nodes.append({
            'id': node.get('id'),
            'name': node.get('name'),
            'type': node.get('type'),
            'documentation': node.get('documentation', ''),
            'stereotypes': node.get('stereotypes', ''),
            'tags': node.get('tags', ''),
            'package': node.get('package', '')
        })

    for edge in root.findall('edge'):
        edges.append({
            'source': edge.get('source'),
            'target': edge.get('target'),
            'label': edge.get('label'),
            'roleSource': edge.get('roleSource', ''),
            'roleTarget': edge.get('roleTarget', ''),
            'multiplicitySource': edge.get('multiplicitySource', ''),
            'multiplicityTarget': edge.get('multiplicityTarget', ''),
            'aggregation': edge.get('aggregation', '')
        })

    return pd.DataFrame(nodes), pd.DataFrame(edges)
