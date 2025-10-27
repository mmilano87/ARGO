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
