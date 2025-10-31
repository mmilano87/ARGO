import gseapy as gp
import pandas as pd
import os
import re

# === BASE PATH CONFIGURATION ===
base_path = "/Users/mariannamilano/Desktop/risultati/"
output_dir = os.path.join(base_path, "result_enrichment")
os.makedirs(output_dir, exist_ok=True)

# === LIST OF TISSUE DIRECTORIES ===
tissue_dirs = [
    # "Bladder",
    "Brain",
    # "Breast",
    "Colon",
    # "Liver",
    "Lung",
    # "Pancreas",
    "Prostate",
    "Skin",
    "Stomac",
    "Testis",
    "Tyroid"
]

# === FUNCTION: Parse community files ===
def parse_communities(file_path):
    """
    Reads a community file and returns a dictionary {Community: [genes]}.

    Only includes communities containing at least 10 genes.
    The file is expected to have lines formatted like:
        Community 1 (size=42): GENE1, GENE2, GENE3, ...
    """
    communities = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    for line in lines:
        match = re.match(r"Community\s+(\d+).*:\s*(.*)", line)
        if match:
            comm_id = f"C{match.group(1)}"
            genes = [g.strip() for g in match.group(2).split(",") if g.strip()]
            if len(genes) >= 10:
                communities[comm_id] = genes
    return communities


# === FUNCTION: Perform enrichment (GO or KEGG) with FDR < 0.05 ===
def run_enrichment(genes, community_id, method, gene_set):
    """
    Runs enrichment analysis using Enrichr via gseapy.

    Parameters
    ----------
    genes : list
        List of gene symbols.
    community_id : str
        Community identifier (e.g., "C1").
    method : str
        Detection method (e.g., ARGO, Louvain).
    gene_set : str
        Gene set name from Enrichr (e.g., "GO_Biological_Process_2023", "KEGG_2021_Human").

    Returns
    -------
    list of dict
        Enrichment results filtered for FDR < 0.05 (top 5 pathways).
    """
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=[gene_set],
            organism="Human",
            outdir=None,
            cutoff=0.05
        )
    except Exception as e:
        print(f"⚠️ Enrichment failed for {community_id} ({method}, {gene_set}): {e}")
        return []

    results = []
    if enr.results is not None and not enr.results.empty:
        # Keep only significant terms (FDR < 0.05)
        df = enr.results[enr.results["Adjusted P-value"] < 0.05]
        df = df.head(5)  # top 5 enriched pathways
        for _, row in df.iterrows():
            results.append({
                "Community": community_id,
                "Method": method,
                "Top Enriched Terms": row["Term"],
                "FDR": row["Adjusted P-value"]
            })
    return results


# === MAIN LOOP: iterate over all tissues ===
for tissue_dir in tissue_dirs:
    tissue_name = tissue_dir.split("-results")[0].capitalize()
    print(f"\n🧬 Processing {tissue_name}...")

    # Define the path for each method's community file
    results_path = os.path.join(base_path, tissue_dir, "results")
    input_files = {
        "ARGO": os.path.join(results_path, "ARGO", "communities.txt"),
        "Louvain_plain": os.path.join(results_path, "Louvain_plain", "communities.txt"),
        "Louvain_weighted": os.path.join(results_path, "Louvain_weighted", "communities.txt")
    }

    all_go, all_kegg = [], []

    # Loop through methods (ARGO, Louvain)
    for method, file_path in input_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        # Parse communities and run enrichment per community
        comm_dict = parse_communities(file_path)
        for comm_id, genes in comm_dict.items():
            go_res = run_enrichment(genes, comm_id, method, "GO_Biological_Process_2023")
            kegg_res = run_enrichment(genes, comm_id, method, "KEGG_2021_Human")
            all_go.extend(go_res)
            all_kegg.extend(kegg_res)

    # === SAVE RESULTS FOR EACH TISSUE ===
    if all_go:
        go_file = os.path.join(output_dir, f"GO_enrichment_{tissue_name}.txt")
        pd.DataFrame(all_go).to_csv(go_file, sep="\t", index=False)
        print(f"✅ GO enrichment saved to {go_file}")
    else:
        print(f"⚠️ No GO enrichment results for {tissue_name}")

    if all_kegg:
        kegg_file = os.path.join(output_dir, f"KEGG_enrichment_{tissue_name}.txt")
        pd.DataFrame(all_kegg).to_csv(kegg_file, sep="\t", index=False)
        print(f"✅ KEGG enrichment saved to {kegg_file}")
    else:
        print(f"⚠️ No KEGG enrichment results for {tissue_name}")
