"""
CPSC 573 Final Project — Gephi Network Export

Builds two files for Gephi:
  gephi_nodes.csv  — one row per model (id, label, org, domain, year, params, is_frontier, is_notable)
  gephi_edges.csv  — directed edges: Base model → derived model (source, target, weight)

The resulting network reveals how a handful of foundation models (GPT, BERT, Llama, etc.)
spawned entire families of derived and fine-tuned systems.
"""

import os
import pandas as pd
import numpy as np

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, "data", "cleaned", "all_ai_models_cleaned.csv")
EXPORT = os.path.join(ROOT, "data", "gephi")
os.makedirs(EXPORT, exist_ok=True)

print("Loading cleaned dataset …")
df = pd.read_csv(DATA, low_memory=False)

for col in ["Parameters", "Training compute (FLOP)", "year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["log_parameters"] = np.where(df["Parameters"] > 0, np.log10(df["Parameters"]), np.nan)
df["is_frontier"] = df["is_frontier"].fillna(False).astype(bool)
df["is_notable"]  = df["is_notable"].fillna(False).astype(bool)
df["is_open_weights"] = df["is_open_weights"].fillna(False).astype(bool)

# ---------------------------------------------------------------------------
# Nodes — one per unique model name
# ---------------------------------------------------------------------------
nodes = df[["Model", "Organization", "primary_domain", "year",
            "log_parameters", "org_type", "is_frontier", "is_notable",
            "is_open_weights", "primary_country"]].drop_duplicates(subset=["Model"]).copy()

nodes = nodes.rename(columns={
    "Model":           "Id",
    "Organization":    "Organization",
    "primary_domain":  "Domain",
    "year":            "Year",
    "log_parameters":  "LogParameters",
    "org_type":        "OrgType",
    "is_frontier":     "IsFrontier",
    "is_notable":      "IsNotable",
    "is_open_weights": "OpenWeights",
    "primary_country": "Country",
})
nodes["Label"] = nodes["Id"]

# Assign numeric size for Gephi: use log(params), fallback to 1
nodes["Size"] = nodes["LogParameters"].fillna(5).clip(lower=3, upper=14)

# Colour category for Gephi: encode as string
nodes["Category"] = "Standard"
nodes.loc[nodes["IsFrontier"], "Category"] = "Frontier"
nodes.loc[nodes["IsNotable"], "Category"] = "Notable"
nodes.loc[nodes["IsFrontier"] & nodes["IsNotable"], "Category"] = "Frontier+Notable"

print(f"  Nodes: {len(nodes):,}")
nodes.to_csv(os.path.join(EXPORT, "gephi_nodes.csv"), index=False)
print("  [saved] gephi_nodes.csv")

# ---------------------------------------------------------------------------
# Edges — Base model → derived model
# ---------------------------------------------------------------------------
# The "Base model" column contains the name of the model a given model was built on.
# We build edges: Base model → current model.
edge_df = df[["Model", "Base model"]].dropna(subset=["Base model"]).copy()

# A model can list multiple base models separated by commas
edge_df["Base model"] = edge_df["Base model"].str.strip()
edges_expanded = edge_df.assign(
    base_split=edge_df["Base model"].str.split(",")
).explode("base_split")
edges_expanded["base_split"] = edges_expanded["base_split"].str.strip()

# Only keep edges where both endpoints exist in our node list
valid_ids = set(nodes["Id"].dropna())
edges_expanded = edges_expanded[
    edges_expanded["Model"].isin(valid_ids) &
    edges_expanded["base_split"].isin(valid_ids)
].copy()

edges = edges_expanded.rename(columns={
    "base_split": "Source",
    "Model":      "Target",
})[["Source", "Target"]]
edges["Weight"] = 1
edges["Type"]   = "Directed"

print(f"  Edges: {len(edges):,}")
edges.to_csv(os.path.join(EXPORT, "gephi_edges.csv"), index=False)
print("  [saved] gephi_edges.csv")

# ---------------------------------------------------------------------------
# Summary stats for the network
# ---------------------------------------------------------------------------
in_degree = edges.groupby("Target").size().rename("in_degree")
top_parents = edges.groupby("Source").size().sort_values(ascending=False).head(20)

print("\n  Top 20 most-forked base models:")
for model, count in top_parents.items():
    print(f"    {count:>4}  ←  {model}")

print("\nGephi export complete.")
print(f"  Import gephi_nodes.csv and gephi_edges.csv into Gephi.")
print(f"  Use 'Id' as node ID. Enable 'Has header' for both files.")
print(f"  Recommended layout: ForceAtlas2  |  Node size: LogParameters  |  Color: Category")
