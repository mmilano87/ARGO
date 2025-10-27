# ARGO: Ricci-Curvature-Guided Graph Neural Network Framework

**ARGO** (Algorithm for Ricci-curvature-Guided Optimization) is a Python framework for **community detection and topological learning** on biological networks.  
It combines **network geometry (Ricci curvature)** with **Graph Neural Networks (GNNs)** to produce biologically meaningful embeddings and identify functional modules in gene co-expression or interaction networks.

Developed at the **University “Magna Graecia” of Catanzaro**, ARGO is part of a research pipeline integrating curvature, graph topology, and neural learning methods for biomedical data.

---

## 🧩 What the Script Does

### **Input**
Reads two files, previously generated from curvature analysis (e.g., using `curvaturanew.py`):

| File | Description | Required Columns |
|------|--------------|------------------|
| `curvature_edges.txt` | Edge list with Ricci curvature values | `source`, `target`, `ricci_curvature` |
| `node_labels_features.txt` | Node-level metadata and features | `symbol`, `degree`, `strength`, ... |

Default paths can be modified at the top of the script:
```python
BASE = "/Users/mariannamilano/Desktop/"
EDGE_FILE_CURV = os.path.join(BASE, "curvature_edges.txt")
NODE_FEAT_FILE = os.path.join(BASE, "node_labels_features.txt")
OUT_DIR = os.path.join(BASE, "results")


Processing Steps
1. Network Construction

Builds a weighted undirected graph from curvature-weighted edges.

Normalizes curvature values to positive weights.

Computes node-level statistics (degree and strength).

2. Graph Neural Network (GCN Autoencoder)

Initializes a 2-layer GCNEncoder (Graph Convolutional Network).

Learns low-dimensional embeddings by reconstructing network edges.

Uses a link prediction loss (binary cross-entropy with random negative sampling).

3. Training Phase

Optimizes the GCN with Adam optimizer (default 200 epochs, lr=1e-3).

Logs and plots the training loss (training_loss.csv and training_loss.png).

4. Evaluation

Performs link reconstruction to compute:

AUC (Area Under ROC Curve)

AUPR (Area Under Precision-Recall Curve)

Generates corresponding ROC and PR plots.

5. Community Detection

Applies K-Means clustering on the learned embeddings.

Automatically selects the best number of clusters (K) maximizing the Silhouette Score.

Outputs community composition and sizes.

6. Visualization

Performs PCA on the embedding space (2D visualization).

Colors nodes by community assignment (embedding_pca.png).

📂 Output Files

All results are saved in the specified output folder (default: /Users/mariannamilano/Desktop/results/).

File	Description
communities.txt	List of detected communities and their members
metrics.txt	AUC, AUPR, best K, Silhouette score, and cluster sizes
gnn_embeddings.npy	Learned node embeddings
training_loss.csv / training_loss.png	Training loss log and plot
roc_curve.png / pr_curve.png	ROC and Precision–Recall curves
embedding_pca.png	PCA visualization of GNN embeddings
node_order.txt	Node order corresponding to embeddings
⚙️ Requirements

Install all required dependencies with:

pip install torch torch_geometric pandas numpy networkx scikit-learn matplotlib

Minimum Python version

Python ≥ 3.8

Note:
If PyTorch Geometric installation fails on CPU, use:

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric

🚀 Usage

Run directly from terminal:

python argo.py


All results will be generated in the results/ directory.
The script will display AUC, AUPR, and cluster metrics in the terminal.


# Curvature Analysis on Gene Co-Expression Networks

This Python script performs a complete topological analysis pipeline on gene co-expression networks, integrating biological and geometric properties through **Ollivier–Ricci curvature**.  
It builds a weighted graph from correlation data (e.g., GTEx/TCGA), computes curvature values using the `GraphRicciCurvature` library, and generates node-level features for downstream machine learning or **Graph Neural Network (GNN)** models.

---

## 🧩 What the Script Does

### **Input**
Reads two tab-separated (`.txt` or `.tsv`) files:

- **Edge file:** contains pairwise gene correlations  
  Columns required:  
  `source`, `target`, `category`, `correlation`, `pvalue`, `padj`
  
- **Node file:** contains metadata for each gene  
  Columns required:  
  `Name`, `symbol`, `info1`, `info2`, `info3`, `location`, `category`, `cluster`

Example input paths (can be changed in the script header):
```python
EDGE_FILE = "/Users/mariannamilano/Desktop/.txt"
NODE_FILE = "/Users/mariannamilano/Desktop/.txt"
OUT_DIR   = "/Users/mariannamilano/Desktop/output"


Network Construction

Maps Ensembl gene identifiers (e.g., ENSG000001234.5) to gene symbols.

Filters edges according to:

Minimum absolute correlation (MIN_ABS_CORR)

Maximum adjusted p-value (MAX_PADJ)

Builds a weighted graph with edge weight:

weight
=
∣
correlation
∣
×
(
1
−
log
⁡
10
(
padj
)
)
weight=∣correlation∣×(1−log
10
	​

(padj))
Curvature Computation

Uses the Ollivier–Ricci curvature via the GraphRicciCurvature
 Python package.

Edge weights are considered during curvature computation (alpha = 0.5).

Outputs a file containing each edge with its corresponding curvature value.

Feature Extraction

Generates a node-level feature file including:

Metadata: symbol, category, cluster

Expression statistics: mean, standard deviation, min, max (parsed from node info columns)

Network metrics: degree, strength, mean absolute correlation

Curvature-based feature: mean curvature of incident edges per node

Statistical Summary

After curvature computation, the script reports:

Number of nodes and edges

Mean, variance, minimum, and maximum of curvature values

These statistics are saved in a summary file for further analysis.

📂 Output Files

All results are saved to the folder specified by OUT_DIR.
Example: /Users/mariannamilano/Desktop/results_sub42/

File	Description
curvature_edges.txt	Edge list with computed Ollivier–Ricci curvature values
curvature_stats.txt	Summary statistics of curvature distribution
node_labels_features.txt	Node-level features including biological and topological metrics

Example output line from curvature_edges.txt:

AP2S1, QSER1, 0.0732

⚙️ Requirements

You can install all dependencies using:

pip install pandas numpy networkx GraphRicciCurvature

Required Python version:

Python ≥ 3.8

Main dependencies:

pandas — for tabular data management

numpy — for numerical operations

networkx — for graph construction and analysis

GraphRicciCurvature — for curvature computation

🚀 Usage

To run the script:

python curvaturanew.py


You can modify file paths and thresholds directly in the configuration section:

MIN_ABS_CORR = 0.1
MAX_PADJ     = 0.05


The script will automatically:

Build the graph

Compute curvature

Save all outputs to the selected folder
