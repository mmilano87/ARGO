import os
import re
import math
import pandas as pd
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# ============ CONFIGURATION ============
EDGE_FILE = "/Users/mariannamilano/Desktop/Edge file.txt"      # Edge file: source target category correlation pvalue padj (TSV)
NODE_FILE = "/Users/mariannamilano/Desktop/Node file.txt"          # Node file: Name symbol info1 info2 info3 location category cluster (TSV)
OUT_DIR   = "/Users/mariannamilano/Desktop/results_sub42"         # Output folder
# Optional filters for correlation network (None to disable)
MIN_ABS_CORR = 0.0       # e.g. 0.1 or 0.3
MAX_PADJ     = 1.0       # e.g. 0.05
# ======================================

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) LOAD INPUT ----------
# Edge list (TSV with header): source target category correlation pvalue padj
edges = pd.read_csv(EDGE_FILE, sep="\t", dtype=str)
# Normalize column names
edges.columns = [c.strip().lower() for c in edges.columns]

# Cast numeric columns
for c in ["correlation", "pvalue", "padj"]:
    if c in edges.columns:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")

# Node list (TSV with header): Name symbol info1 info2 info3 location category cluster
nodes = pd.read_csv(NODE_FILE, sep="\t", dtype=str)
nodes.columns = [c.strip() for c in nodes.columns]

# Create mapping: Name (Ensembl ID with version) → Symbol
if "Name" in nodes.columns and "symbol" in nodes.columns:
    name_to_symbol = dict(zip(nodes["Name"], nodes["symbol"]))
else:
    raise ValueError("Node file must contain columns 'Name' and 'symbol'.")

# ---------- 2) MAP ENSEMBL → SYMBOL ON EDGES ----------
# If a symbol is missing, keep the original ID to avoid losing the edge
def map_id(x):
    return name_to_symbol.get(x, x)

edges["src_sym"] = edges["source"].map(map_id)
edges["tgt_sym"] = edges["target"].map(map_id)

# ---------- 3) OPTIONAL EDGE FILTERS ----------
mask = np.ones(len(edges), dtype=bool)
if MIN_ABS_CORR is not None:
    mask &= edges["correlation"].abs().fillna(0) >= float(MIN_ABS_CORR)
if MAX_PADJ is not None:
    mask &= edges["padj"].fillna(1.0) <= float(MAX_PADJ)

edges_f = edges[mask].copy().reset_index(drop=True)

# ---------- 4) BUILD WEIGHTED GRAPH ----------
# Base weight = |corr| (you can modify, e.g., |corr| * -log10(padj))
def safe_nlog10(x):
    if pd.isna(x):
        return 0.0
    x = min(max(x, 1e-300), 1.0)
    return -math.log10(x)

edges_f["w_corr"] = edges_f["correlation"].abs().fillna(0.0)
edges_f["w_sig"]  = edges_f["padj"].apply(safe_nlog10)
edges_f["weight"] = edges_f["w_corr"] * (1.0 + edges_f["w_sig"])  # robust and simple choice

G = nx.Graph()
for _, r in edges_f.iterrows():
    u, v = str(r["src_sym"]).strip(), str(r["tgt_sym"]).strip()
    if u == "" or v == "" or u is None or v is None:
        continue
    w = float(r["weight"]) if not pd.isna(r["weight"]) else 1.0
    G.add_edge(u, v, weight=w, corr=float(r["correlation"]) if not pd.isna(r["correlation"]) else 0.0)

# ---------- 5) COMPUTE CURVATURE (Ollivier–Ricci) ----------
# Uses edge weights as 'weight' attribute; alpha=0.5 is standard
if G.number_of_edges() == 0:
    raise RuntimeError("Filtered graph has no edges: relax MIN_ABS_CORR/MAX_PADJ thresholds.")

orc = OllivierRicci(G, alpha=0.5, verbose="INFO")  # to ignore weights, set method='OTD'
orc.compute_ricci_curvature()

# Extract curvature for each edge
curv_rows = []
for u, v, d in orc.G.edges(data=True):
    k = d.get("ricciCurvature", None)
    if k is None:
        continue
    curv_rows.append((u, v, float(k)))

df_curv = pd.DataFrame(curv_rows, columns=["gene1", "gene2", "ricci_curvature"])

# ---------- 6) SAVE RESULTS ----------
# 6a) Save curvature edges in format: "GENE1, GENE2, 0.xxxx"
curv_out_csv = os.path.join(OUT_DIR, "curvature_edges.txt")
df_curv_out = df_curv.copy()
df_curv_out["ricci_curvature"] = df_curv_out["ricci_curvature"].map(lambda x: f"{x:.4f}")
df_curv_out.to_csv(curv_out_csv, index=False, header=False)

# 6b) Save summary statistics
curv_vals = df_curv["ricci_curvature"].values
stats = {
    "num_nodes": [G.number_of_nodes()],
    "num_edges": [G.number_of_edges()],
    "curv_mean": [float(np.mean(curv_vals))],
    "curv_var":  [float(np.var(curv_vals))],
    "curv_min":  [float(np.min(curv_vals))],
    "curv_max":  [float(np.max(curv_vals))],
}
pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "curvature_stats.txt"), index=False)

# ---------- 7) BUILD NODE LABEL/FEATURE FILE ----------
# Extract node features from input file:
# - symbol
# - category, cluster
# - mean_expr, sd_expr (from info2: "X (+/- Y)")
# - min_expr, max_expr (from info3: "A - B")
def parse_info2(s):
    """Parse 'mean (+/- sd)' pattern."""
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.search(r"([-\d\.Ee]+)\s*\(\s*\+/-\s*([-\d\.Ee]+)\s*\)", str(s))
    if m:
        return (float(m.group(1)), float(m.group(2)))
    try:
        return (float(str(s).strip()), np.nan)
    except:
        return (np.nan, np.nan)

def parse_info3(s):
    """Parse 'min - max' pattern."""
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.search(r"([-\d\.Ee]+)\s*-\s*([-\d\.Ee]+)", str(s))
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (np.nan, np.nan)

nodes_feat = nodes.copy()
nodes_feat["symbol"] = nodes_feat["symbol"].astype(str)
nodes_feat[["mean_expr", "sd_expr"]] = nodes_feat["info2"].apply(lambda x: pd.Series(parse_info2(x)))
nodes_feat[["min_expr", "max_expr"]] = nodes_feat["info3"].apply(lambda x: pd.Series(parse_info3(x)))

# Keep only relevant columns
keep_cols = ["symbol", "category", "cluster", "mean_expr", "sd_expr", "min_expr", "max_expr"]
for c in keep_cols:
    if c not in nodes_feat.columns:
        nodes_feat[c] = np.nan
labels = nodes_feat[keep_cols].drop_duplicates()

# Add network-derived features
deg_dict      = dict(G.degree())
strength_dict = {n: 0.0 for n in G.nodes()}
abs_corr_sum  = {n: 0.0 for n in G.nodes()}

for u, v, d in G.edges(data=True):
    w = float(d.get("weight", 1.0))
    c = abs(float(d.get("corr", 0.0)))
    strength_dict[u] += w
    strength_dict[v] += w
    abs_corr_sum[u]  += c
    abs_corr_sum[v]  += c

labels["degree"]        = labels["symbol"].map(deg_dict).fillna(0).astype(int)
labels["strength"]      = labels["symbol"].map(strength_dict).fillna(0.0)
labels["mean_abs_corr"] = labels["symbol"].map(abs_corr_sum).fillna(0.0)
labels["mean_abs_corr"] = labels.apply(lambda r: (r["mean_abs_corr"]/r["degree"]) if r["degree"]>0 else 0.0, axis=1)

# Add curvature-based feature: mean curvature of incident edges
curv_incident = {n: [] for n in G.nodes()}
for _, row in df_curv.iterrows():
    u, v, k = row["gene1"], row["gene2"], float(row["ricci_curvature"])
    if u in curv_incident:
        curv_incident[u].append(k)
    if v in curv_incident:
        curv_incident[v].append(k)

mean_k = {n: (float(np.mean(v)) if len(v)>0 else 0.0) for n, v in curv_incident.items()}
labels["mean_edge_curvature"] = labels["symbol"].map(mean_k).fillna(0.0)

# Save node label/feature file
labels_out = os.path.join(OUT_DIR, "node_labels_features.txt")
labels.to_csv(labels_out, index=False)

print("== DONE ==")
print(f"Curvature edges  : {curv_out_csv}")
print(f"Curvature stats  : {os.path.join(OUT_DIR, 'curvature_stats.txt')}")
print(f"Node labels file : {labels_out}")
print(f"Number of nodes: {G.number_of_nodes()} — Number of edges: {G.number_of_edges()}")
