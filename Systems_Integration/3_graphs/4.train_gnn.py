"""
Train a simple GNN on system_graph.json, using heuristic risk labels as targets.
Writes an updated JSON file with risk_gnn and risk_scores fields filled in.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


RISK_TO_INT = {"low": 0, "medium": 1, "high": 2}
INT_TO_RISK = {v: k for k, v in RISK_TO_INT.items()}


def load_graph(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_pyg_graph(graph: Dict[str, Any]) -> Tuple[Data, List[str]]:
    """
    Build a PyG Data object from system_graph.json.

    Returns:
      - Data object with x, edge_index, y, train/val/test masks.
      - node_order: list of node ids in the order of x/y rows.
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    node_ids = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Features: [degree, betweenness, closeness]
    feats = []
    labels = []
    for n in nodes:
        f = n.get("features", {})
        degree = float(f.get("degree", 0.0))
        bet = float(f.get("betweenness", 0.0))
        clo = float(f.get("closeness", 0.0))
        feats.append([degree, bet, clo])

        rh = n.get("risk_heuristic")
        if rh in RISK_TO_INT:
            labels.append(RISK_TO_INT[rh])
        else:
            # If no heuristic label, treat as unlabeled (-1)
            labels.append(-1)

    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Edge index (treat as undirected: add both directions)
    edge_index_list = []
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        if src not in id_to_idx or tgt not in id_to_idx:
            continue
        i = id_to_idx[src]
        j = id_to_idx[tgt]
        edge_index_list.append((i, j))
        edge_index_list.append((j, i))

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        # graph with no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Normalize basic features 
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

    # Build train/val/test masks
    num_nodes = x.size(0)
    labeled_mask = y >= 0
    labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).view(-1)
    num_labeled = labeled_indices.numel()

    if num_labeled == 0:
        raise ValueError("No labeled nodes (risk_heuristic) found in graph.")

    perm = torch.randperm(num_labeled)
    train_end = int(0.6 * num_labeled)
    val_end = int(0.8 * num_labeled)

    train_idx = labeled_indices[perm[:train_end]]
    val_idx = labeled_indices[perm[train_end:val_end]]
    test_idx = labeled_indices[perm[val_end:]]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data, node_ids

'''
# 2 layer
class RiskGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
'''

# 3 layer
class RiskGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


def train_model(
    data: Data,
    num_epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    hidden_dim: int = 64,
    dropout: float = 0.5,
) -> RiskGCN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskGCN(
        in_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        train_mask = data.train_mask & (data.y >= 0)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=-1)

            def acc(mask):
                mask = mask & (data.y >= 0)
                if mask.sum() == 0:
                    return float("nan")
                correct = (pred[mask] == data.y[mask]).sum().item()
                return correct / mask.sum().item()

            train_acc = acc(data.train_mask)
            val_acc = acc(data.val_mask)
            test_acc = acc(data.test_mask)

        # track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 20 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.3f} | "
                f"Val Acc {val_acc:.3f} | "
                f"Test Acc {test_acc:.3f}"
            )

        # early stopping check
        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    # load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()



def apply_model_and_update_json(
    model: RiskGCN,
    data: Data,
    node_ids: List[str],
    graph: Dict[str, Any],
    out_path: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=-1)

    preds = probs.argmax(dim=-1).tolist()
    probs_np = probs.tolist()

    nid_to_index = {nid: i for i, nid in enumerate(node_ids)}
    nodes = graph["nodes"]

    for n in nodes:
        nid = n["id"]
        idx = nid_to_index[nid]
        pred_class = preds[idx]
        n["risk_gnn"] = INT_TO_RISK[pred_class]
        p = probs_np[idx]
        n["risk_scores"] = {
            "low": float(p[0]),
            "medium": float(p[1]),
            "high": float(p[2]),
        }

    out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"Wrote updated graph with GNN predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN on system_graph.json and write back risk_gnn + risk_scores."
    )
    parser.add_argument("--graph-json", required=True, help="Input system_graph.json")
    parser.add_argument(
        "--out-json",
        required=False,
        help="Output JSON path (default: *_with_gnn.json)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    in_path = Path(args.graph_json)
    out_path = Path(args.out_json) if args.out_json else in_path.with_name(
        in_path.stem + "_with_gnn.json"
    )

    graph = load_graph(in_path)
    data, node_ids = build_pyg_graph(graph)

    model = train_model(
    data,
    num_epochs=args.epochs,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
    hidden_dim=args.hidden_dim,
    dropout=args.dropout,
)


    apply_model_and_update_json(model, data, node_ids, graph, out_path)


if __name__ == "__main__":
    main()
