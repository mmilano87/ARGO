#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARGO: Ricci-Curvature-Guided Graph Neural Network Framework
for Community Detection in Biological Networks
------------------------------------------------------------
Final version (no subfolders, all results in /Users/mariannamilano/Desktop/results/)
Generates:
 - Communities
 - Metrics (AUC, AUPR, Silhouette, K)
 - Training loss log + plot
 - ROC and PR curves
 - PCA plot colored by community
"""

import os, re, math
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA

# Use Agg backend for headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise RuntimeError(
        "PyTorch and PyTorch Geometric are required.\n"
        f"Error: {e}\n"
        "Install with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install torch_geometric"
    )

# ======================
# PATH CONFIGURATION
# ======================
BASE = "/Users/mariannamilano/Desktop/"
EDGE_FILE_CURV = os.path.join(BASE, "curvature_edges.txt")
NODE_FEAT_FILE = os.path.join(BASE, "node_labels_features.txt")
OUT_DIR = os.path.join(BASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_COMM       = os.path.join(OUT_DIR, "communities.txt")
OUT_EMB_NPY    = os.path.join(OUT_DIR, "gnn_embeddings.npy")
OUT_NODE_ORDER = os.path.join(OUT_DIR, "node_order.txt")
OUT_METRICS    = os.path.join(OUT_DIR, "metrics.txt")
OUT_LOSS_CSV   = os.path.join(OUT_DIR, "training_loss.csv")
OUT_LOSS_PNG   = os.path.join(OUT_DIR, "training_loss.png")
OUT_ROC_PNG    = os.path.join(OUT_DIR, "roc_curve.png")
OUT_PR_PNG     = os.path.join(OUT_DIR, "pr_curve.png")
OUT_PCA_PNG    = os.path.join(OUT_DIR, "embedding_pca.png")

# ======================
# UTILITY FUNCTIONS
# ======================
def load_curvature_edges(path):
    """Load curvature-weighted edge list."""
    rows = []
    pat = re.compile(r'^\s*([A-Za-z0-9._-]+)[,;\t]\s*([A-Za-z0-9._-]+)[,;\t]\s*([+-]?\d+(?:[.,]\d+)?)')
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            src, dst, w = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            w = float(w.replace(',', '.'))
            if src != dst:
                rows.append((src, dst, w))
    df = pd.DataFrame(rows, columns=['source', 'target', 'ricci_curvature']).drop_duplicates()
    if df.empty:
        raise ValueError("No valid edges found in curvature_edges.txt")
    return df

def load_user_node_features(path):
    """Load optional node features."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=['symbol'])
    df = pd.read_csv(path, header=None, engine='python')
    if df.empty:
        return pd.DataFrame(columns=['symbol'])
    df.columns = ['symbol'] + [f'f{i}' for i in range(1, df.shape[1])]
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def write_communities_txt(path, labels, nodes):
    """Save detected communities."""
    comm = {}
    for n, c in zip(nodes, labels):
        comm.setdefault(int(c), []).append(n)
    with open(path, 'w', encoding='utf-8') as f:
        for cid, members in sorted(comm.items(), key=lambda x: (-len(x[1]), x[0])):
            f.write(f"Community {cid} (size={len(members)}): {', '.join(sorted(members))}\n")

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def evaluate_link_reconstruction(Z, edge_index, num_neg=None, seed=42):
    """Compute AUC, AUPR, and ROC/PR curves."""
    rng = np.random.default_rng(seed)
    ei = edge_index.cpu().numpy()
    pos = np.unique(np.sort(ei.T, axis=1), axis=0)
    n_pos = pos.shape[0]
    existing = set((int(u), int(v)) for u, v in pos)
    N = Z.shape[0]
    if num_neg is None:
        num_neg = n_pos

    neg = []
    while len(neg) < num_neg:
        i, j = int(rng.integers(0, N)), int(rng.integers(0, N))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in existing:
            continue
        neg.append((a, b))
    neg = np.array(neg, dtype=int)

    pos_scores = np.sum(Z[pos[:, 0]] * Z[pos[:, 1]], axis=1)
    neg_scores = np.sum(Z[neg[:, 0]] * Z[neg[:, 1]], axis=1)
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = _sigmoid(np.concatenate([pos_scores, neg_scores]))
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc, aupr, fpr, tpr, prec, rec

# ======================
# MODEL
# ======================
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=16, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight=edge_weight)

def train_gcn_link_ae(encoder, x, edge_index, edge_weight, epochs=200, lr=1e-3):
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    N = x.shape[0]
    losses = []
    for ep in range(1, epochs + 1):
        encoder.train()
        opt.zero_grad()
        z = encoder(x, edge_index, edge_weight=edge_weight)
        pos = edge_index.t()
        pos_scores = (z[pos[:, 0]] * z[pos[:, 1]]).sum(dim=1)
        i_rand = torch.randint(0, N, (pos.shape[0],))
        j_rand = torch.randint(0, N, (pos.shape[0],))
        neg_scores = (z[i_rand] * z[j_rand]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if ep % 50 == 0:
            print(f"[ARGO] epoch {ep}/200 - loss {loss.item():.4f}")
    encoder.eval()
    with torch.no_grad():
        Z = encoder(x, edge_index, edge_weight=edge_weight).cpu().numpy()
    return Z, losses

# ======================
# MAIN
# ======================
print("🚀 Running ARGO...")

df = load_curvature_edges(EDGE_FILE_CURV)
min_c = df['ricci_curvature'].min()
df['weight'] = df['ricci_curvature'] - (min_c if not math.isnan(min_c) else 0.0) + 1e-9

# ✅ FIXED: use only required columns
G = nx.Graph()
for s, t, w in df[['source', 'target', 'weight']].itertuples(index=False, name=None):
    G.add_edge(s, t, weight=w)

deg = pd.Series(dict(G.degree()), name='degree')
strn = pd.Series({n: 0.0 for n in G.nodes()}, name='strength')
for u, v, d in G.edges(data=True):
    w = float(d.get('weight', 0.0))
    strn[u] += w
    strn[v] += w

node_df = pd.DataFrame({'symbol': list(G.nodes()), 'degree': deg.values, 'strength': strn.values})
feat = node_df.fillna(0.0)

nodes_sorted = sorted(G.nodes())
idx = {n: i for i, n in enumerate(nodes_sorted)}
X = feat.set_index('symbol').reindex(nodes_sorted).fillna(0.0).values.astype(np.float32)

edges = [(idx[u], idx[v], float(d['weight'])) for u, v, d in G.edges(data=True)]
edge_index = torch.tensor([[e[0] for e in edges] + [e[1] for e in edges],
                           [e[1] for e in edges] + [e[0] for e in edges]], dtype=torch.long)
edge_weight = torch.tensor([e[2] for e in edges] + [e[2] for e in edges], dtype=torch.float32)
x = torch.tensor(X, dtype=torch.float32)

encoder = GCNEncoder(in_dim=x.shape[1])
Z, losses = train_gcn_link_ae(encoder, x, edge_index, edge_weight)

np.save(OUT_EMB_NPY, Z)
open(OUT_NODE_ORDER, 'w').write('\n'.join(nodes_sorted))

# Training loss plot
pd.DataFrame({'epoch': range(1, len(losses) + 1), 'loss': losses}).to_csv(OUT_LOSS_CSV, index=False)
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ARGO Training Loss")
plt.tight_layout()
plt.savefig(OUT_LOSS_PNG, dpi=150)
plt.close()

# Link reconstruction
auc, aupr, fpr, tpr, prec, rec = evaluate_link_reconstruction(Z, edge_index)
print(f"[ARGO] AUC={auc:.4f} | AUPR={aupr:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (ARGO)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_ROC_PNG, dpi=150)
plt.close()

plt.figure()
plt.plot(rec, prec, label=f"AUPR={aupr:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve (ARGO)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PR_PNG, dpi=150)
plt.close()

# K-Means Clustering
best_k, best_s, best_labels = None, -1.0, None
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Z)
    try:
        s = silhouette_score(Z, labels)
    except:
        s = float('nan')
    if not np.isnan(s) and s > best_s:
        best_s, best_k, best_labels = s, k, labels

write_communities_txt(OUT_COMM, best_labels, nodes_sorted)
sizes = pd.Series(best_labels).value_counts().sort_index()
with open(OUT_METRICS, 'w') as f:
    f.write(f"AUC: {auc:.6f}\nAUPR: {aupr:.6f}\nK: {best_k}\nSilhouette: {best_s:.6f}\n")
    for cid, sz in sizes.items():
        f.write(f"  - Cluster {cid}: {sz} nodes\n")

# PCA visualization
pca = PCA(n_components=2)
Z2 = pca.fit_transform(Z)
plt.figure(figsize=(6, 5))
sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=best_labels, cmap="viridis", s=25)
plt.colorbar(sc, label="Community")
plt.title("ARGO Embeddings (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUT_PCA_PNG, dpi=150)
plt.close()

print("✅ ARGO completed successfully!")
print("Results saved in:", OUT_DIR)
