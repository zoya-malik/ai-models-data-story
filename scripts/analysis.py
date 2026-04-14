"""
CPSC 573 Final Project — Part 2 Analysis
Generates all analytical figures for the data story blog post.

Covers 7 research questions:
  Q1  Scaling law divergence: compute vs parameters pre/post-2020
  Q2  Geographic concentration (CSV exported for Tableau)
  Q3  Logistic regression: predictors of open model weights
  Q4  K-means clustering on resource profiles
  Q5  Cost regression: drivers of training cost
  Q6  Language vs Multimodal scaling trajectories
  Q7  Random forest: what predicts notability?
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, "data", "cleaned", "all_ai_models_cleaned.csv")
FIGS   = os.path.join(ROOT, "report", "figures", "analysis")
EXPORT = os.path.join(ROOT, "data", "tableau_exports")
os.makedirs(FIGS,   exist_ok=True)
os.makedirs(EXPORT, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    "Industry":           "#2563EB",
    "Academia":           "#16A34A",
    "Research collective":"#9333EA",
    "Government":         "#DC2626",
    "Language":           "#2563EB",
    "Biology":            "#16A34A",
    "Vision":             "#F59E0B",
    "Multimodal":         "#9333EA",
    "Image generation":   "#EC4899",
    "Speech":             "#14B8A6",
    "Video":              "#F97316",
    "Other":              "#94A3B8",
}

plt.rcParams.update({
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#FAFAFA",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E5E7EB",
    "grid.linewidth":     0.6,
    "font.family":        "sans-serif",
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
})

def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  [saved] {name}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading cleaned dataset …")
df = pd.read_csv(DATA, low_memory=False)
df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
for col in ["Parameters", "Training compute (FLOP)", "Training dataset size (total)",
            "Training compute cost (2023 USD)", "Training time (hours)", "Hardware quantity"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["log_parameters"] = np.where(df["Parameters"] > 0, np.log10(df["Parameters"]), np.nan)
df["log_compute"]    = np.where(df["Training compute (FLOP)"] > 0,
                                np.log10(df["Training compute (FLOP)"]), np.nan)
df["log_cost"]       = np.where(df["Training compute cost (2023 USD)"] > 0,
                                np.log10(df["Training compute cost (2023 USD)"]), np.nan)
df["log_data"]       = np.where(df["Training dataset size (total)"] > 0,
                                np.log10(df["Training dataset size (total)"]), np.nan)
df["log_hw"]         = np.where(df["Hardware quantity"] > 0,
                                np.log10(df["Hardware quantity"]), np.nan)

# Simplified domain groups
DOMAIN_MAP = {
    "Language": "Language", "Biology": "Biology", "Vision": "Vision",
    "Multimodal": "Multimodal", "Image generation": "Image generation",
    "Speech": "Speech", "Video": "Video",
}
df["domain_grp"] = df["primary_domain"].map(DOMAIN_MAP).fillna("Other")

print(f"  Loaded {len(df)} rows, {df.shape[1]} columns.\n")

# ===========================================================================
# Q1  SCALING LAW DIVERGENCE — compute vs parameters, pre/post 2020
# ===========================================================================
print("Q1 — Scaling law divergence …")

q1 = df.dropna(subset=["log_parameters", "log_compute", "year", "org_type"]).copy()
q1 = q1[(q1["log_parameters"] > 0) & (q1["log_compute"] > 0)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle("How the Scaling Law Has Shifted: Compute vs Parameters", fontsize=15, y=1.01)

periods = [("Pre-2020", q1[q1["year"] < 2020]), ("Post-2020", q1[q1["year"] >= 2020])]
colors   = {"Industry": PALETTE["Industry"], "Academia": PALETTE["Academia"],
            "Research collective": PALETTE["Research collective"], "Government": PALETTE["Government"]}

for ax, (label, sub) in zip(axes, periods):
    for org, grp in sub.groupby("org_type"):
        ax.scatter(grp["log_parameters"], grp["log_compute"],
                   color=colors.get(org, "#94A3B8"), alpha=0.45, s=22,
                   edgecolors="white", linewidth=0.3, label=org)

    # Fit overall trend line
    mask = np.isfinite(sub["log_parameters"]) & np.isfinite(sub["log_compute"])
    if mask.sum() > 10:
        slope, intercept, r, p, _ = stats.linregress(
            sub.loc[mask, "log_parameters"], sub.loc[mask, "log_compute"])
        x_fit = np.linspace(sub["log_parameters"].min(), sub["log_parameters"].max(), 200)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=2.2, alpha=0.8,
                label=f"Trend (slope={slope:.2f}, R²={r**2:.2f})")

    ax.set_title(label, fontsize=13)
    ax.set_xlabel("log₁₀(Parameters)")
    ax.set_ylabel("log₁₀(Training Compute, FLOP)" if ax is axes[0] else "")
    ax.legend(loc="upper left", fontsize=8)

    # Annotate count
    ax.text(0.97, 0.05, f"n = {len(sub):,}", transform=ax.transAxes,
            ha="right", fontsize=9, color="#6B7280")

# Annotation: efficiency story
axes[1].annotate(
    "Chinchilla (2022) showed\nyou can train smaller models\nbetter — flatter slope post-2020",
    xy=(10.5, 24.5), xytext=(7.5, 27.5), fontsize=8.5, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.9)
)

fig.tight_layout()
save(fig, "q1_scaling_law_divergence.png")


# ===========================================================================
# Q3  OPEN WEIGHTS PREDICTOR — logistic regression
# ===========================================================================
print("Q3 — Open weights predictor …")

q3 = df.dropna(subset=["is_open_weights"]).copy()
q3["is_open_weights"] = q3["is_open_weights"].astype(bool)

# Feature engineering
q3["is_industry"]  = (q3["org_type"] == "Industry").astype(int)
q3["is_academia"]  = (q3["org_type"] == "Academia").astype(int)
q3["is_language"]  = (q3["primary_domain"] == "Language").astype(int)
q3["is_frontier"]  = q3["is_frontier"].fillna(False).astype(int)
q3["has_params"]   = q3["log_parameters"].notna().astype(int)
q3["log_params_f"] = q3["log_parameters"].fillna(q3["log_parameters"].median())
q3["log_comp_f"]   = q3["log_compute"].fillna(q3["log_compute"].median())
q3["year_norm"]    = (q3["year"].fillna(q3["year"].median()) - 2010) / 10.0

FEATURES = ["is_industry", "is_academia", "is_language", "is_frontier",
            "log_params_f", "log_comp_f", "year_norm"]
LABELS   = ["Industry org", "Academia org", "Language model", "Frontier model",
            "log(Parameters)", "log(Compute)", "Publication year"]

X = q3[FEATURES].values
y = q3["is_open_weights"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_scaled, y)
cv_acc = cross_val_score(lr, X_scaled, y, cv=5, scoring="roc_auc").mean()

coefs = pd.Series(lr.coef_[0], index=LABELS).sort_values()

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = ["#DC2626" if c < 0 else "#16A34A" for c in coefs]
bars = ax.barh(coefs.index, coefs.values, color=colors_bar, edgecolor="white", height=0.6)
ax.axvline(0, color="#374151", lw=1.2)
ax.set_title("What Predicts Open Model Weights?\nLogistic Regression Coefficients (standardised)", fontsize=13)
ax.set_xlabel("Coefficient  →  more positive = stronger predictor of open weights")

# Value labels
for bar, val in zip(bars, coefs.values):
    ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right",
            fontsize=9, fontweight="bold")

ax.text(0.99, 0.03, f"CV AUC: {cv_acc:.3f}", transform=ax.transAxes,
        ha="right", fontsize=9, color="#6B7280")

green_patch = mpatches.Patch(color="#16A34A", label="Predicts open weights")
red_patch   = mpatches.Patch(color="#DC2626", label="Predicts closed weights")
ax.legend(handles=[green_patch, red_patch], loc="lower right")

fig.tight_layout()
save(fig, "q3_open_weights_predictor.png")

# Open weights trend over time (companion chart)
ow_trend = (
    df[df["year"].between(2012, 2025)]
    .dropna(subset=["is_open_weights", "year"])
    .groupby("year")["is_open_weights"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "open", "count": "total"})
)
ow_trend["pct_open"] = ow_trend["open"] / ow_trend["total"] * 100

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(ow_trend.index, ow_trend["pct_open"], alpha=0.18, color=PALETTE["Language"])
ax.plot(ow_trend.index, ow_trend["pct_open"], color=PALETTE["Language"], lw=2.5, marker="o", ms=6)
ax.set_title("Open-Weight Models as a Share of Annual Releases (2012–2025)", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("% of models with open weights")
ax.set_ylim(0, 100)

# Annotate 2023 inflection
if 2023 in ow_trend.index:
    pct23 = ow_trend.loc[2023, "pct_open"]
    ax.annotate(f"2023: {pct23:.0f}% open\n(Llama 2, Mistral era)",
                xy=(2023, pct23), xytext=(2019.5, pct23 + 10),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#374151", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
fig.tight_layout()
save(fig, "q3_open_weights_trend.png")


# ===========================================================================
# Q4  RESOURCE CLUSTERING
# ===========================================================================
print("Q4 — Resource clustering …")

CLUSTER_COLS = ["log_parameters", "log_compute", "log_data", "log_hw"]
q4 = df.dropna(subset=["log_parameters", "log_compute"]).copy().reset_index(drop=True)
q4["log_data"] = q4["log_data"].fillna(q4["log_data"].median())
q4["log_hw"]   = q4["log_hw"].fillna(q4["log_hw"].median())
q4_X = q4[CLUSTER_COLS].fillna(0).values

# PCA for 2D projection
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(StandardScaler().fit_transform(q4_X))
var_explained = pca.explained_variance_ratio_ * 100

# K-means with k=4 (elbow is around here for this data)
km = KMeans(n_clusters=4, random_state=42, n_init=20)
q4["cluster"] = km.fit_predict(q4_X)

# Label clusters by median parameter size
cluster_medians = q4.groupby("cluster")["log_parameters"].median().sort_values()
cluster_labels  = {
    cluster_medians.index[0]: "Small\n(research-scale)",
    cluster_medians.index[1]: "Mid-scale\n(production)",
    cluster_medians.index[2]: "Large\n(frontier-class)",
    cluster_medians.index[3]: "Frontier\n(true frontier)",
}
# If only 3 clusters make sense, adjust — either way, label by rank
rank_order = ["Small\n(research-scale)", "Mid-scale\n(production)",
              "Large\n(frontier-class)", "Frontier\n(true frontier)"]
for i, (cid, _) in enumerate(cluster_medians.items()):
    cluster_labels[cid] = rank_order[i]

q4["cluster_label"] = q4["cluster"].map(cluster_labels)

CLUSTER_COLORS = {
    rank_order[0]: "#94A3B8",
    rank_order[1]: "#16A34A",
    rank_order[2]: "#F59E0B",
    rank_order[3]: "#DC2626",
}

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle("AI Models by Resource Profile: Four Natural Tiers Emerge", fontsize=14, y=1.01)

# Left: PCA scatter coloured by cluster
ax = axes[0]
for lbl, grp in q4.groupby("cluster_label"):
    ax.scatter(coords[grp.index, 0], coords[grp.index, 1],
               color=CLUSTER_COLORS[lbl], alpha=0.5, s=20,
               edgecolors="white", linewidth=0.2, label=lbl)
ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance explained)")
ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance explained)")
ax.set_title("PCA Projection — Coloured by Cluster")
ax.legend(title="Tier", loc="upper right")

# Right: domain breakdown per cluster
ax = axes[1]
cross = q4.groupby(["cluster_label", "domain_grp"]).size().unstack(fill_value=0)
# Keep top 5 domains
top_domains = df["domain_grp"].value_counts().head(5).index.tolist()
cross = cross[[d for d in top_domains if d in cross.columns]]
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
dom_colors = [PALETTE.get(d, "#94A3B8") for d in cross_pct.columns]
cross_pct.plot.barh(ax=ax, color=dom_colors, edgecolor="white", linewidth=0.3)
ax.set_xlabel("% of cluster")
ax.set_title("Domain Composition by Tier")
ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

fig.tight_layout()
save(fig, "q4_resource_clusters.png")


# ===========================================================================
# Q6  LANGUAGE vs MULTIMODAL SCALING TRAJECTORIES
# ===========================================================================
print("Q6 — Language vs Multimodal trajectories …")

q6 = df[df["primary_domain"].isin(["Language", "Multimodal"])].dropna(
    subset=["log_parameters", "year"]).copy()
q6 = q6[q6["year"].between(2015, 2025)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Language vs Multimodal: Are They Converging?", fontsize=14, y=1.01)

domain_colors = {"Language": PALETTE["Language"], "Multimodal": PALETTE["Multimodal"]}

# Left panel: parameter growth scatter + per-domain trend lines
ax = axes[0]
for dom, grp in q6.groupby("primary_domain"):
    ax.scatter(grp["year"], grp["log_parameters"],
               color=domain_colors[dom], alpha=0.35, s=20,
               edgecolors="white", linewidth=0.2, label=dom)
    # Rolling median
    med = grp.groupby("year")["log_parameters"].median()
    ax.plot(med.index, med.values, color=domain_colors[dom], lw=2.8, zorder=5)

ax.set_xlabel("Year")
ax.set_ylabel("log₁₀(Parameters)")
ax.set_title("Parameter Growth: Language vs Multimodal")
ax.legend()

# Annotate convergence
ax.annotate("Gap narrows post-2022\nas multimodal models\nmatch language scale",
            xy=(2023, 11), xytext=(2018, 12.5), fontsize=8.5, fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="#374151", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))

# Right panel: model count per year
ax = axes[1]
count_by_year = (q6.groupby(["year", "primary_domain"])
                 .size().unstack(fill_value=0)
                 [["Language", "Multimodal"]])
count_by_year.plot.bar(ax=ax, color=[PALETTE["Language"], PALETTE["Multimodal"]],
                       edgecolor="white", linewidth=0.4, width=0.7)
ax.set_xlabel("Year")
ax.set_ylabel("New models released")
ax.set_title("Publication Rate: Language vs Multimodal")
ax.legend(["Language", "Multimodal"])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

fig.tight_layout()
save(fig, "q6_language_vs_multimodal.png")


# ===========================================================================
# Q7  WHAT PREDICTS NOTABILITY? — Random forest feature importance
# ===========================================================================
print("Q7 — Notability predictor …")

q7 = df.copy()
q7["is_notable"] = q7["is_notable"].fillna(False).astype(int)
q7["log_params_f"] = q7["log_parameters"].fillna(q7["log_parameters"].median())
q7["log_comp_f"]   = q7["log_compute"].fillna(q7["log_compute"].median())
q7["log_cost_f"]   = q7["log_cost"].fillna(q7["log_cost"].median())
q7["year_f"]       = q7["year"].fillna(q7["year"].median())
q7["is_open_f"]    = q7["is_open_weights"].fillna(False).astype(int)
q7["is_industry"]  = (q7["org_type"] == "Industry").astype(int)
q7["is_language"]  = (q7["primary_domain"] == "Language").astype(int)
q7["is_frontier"]  = q7["is_frontier"].fillna(False).astype(int)

FEAT7 = ["log_params_f", "log_comp_f", "log_cost_f", "year_f",
         "is_open_f", "is_industry", "is_language", "is_frontier"]
FEAT7_LABELS = ["log(Parameters)", "log(Compute)", "log(Training Cost)",
                "Publication Year", "Open Weights", "Industry Org",
                "Language Domain", "Frontier Model"]

X7 = q7[FEAT7].values
y7 = q7["is_notable"].values

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X7, y7)
cv_auc7 = cross_val_score(rf, X7, y7, cv=5, scoring="roc_auc").mean()

importances = pd.Series(rf.feature_importances_, index=FEAT7_LABELS).sort_values()

fig, ax = plt.subplots(figsize=(11, 6))
colors_imp = [PALETTE["Language"] if v > importances.median() else "#94A3B8"
              for v in importances.values]
bars = ax.barh(importances.index, importances.values, color=colors_imp,
               edgecolor="white", height=0.6)
ax.set_title("What Makes an AI Model Notable?\nRandom Forest Feature Importances", fontsize=13)
ax.set_xlabel("Feature importance (mean decrease in impurity)")

for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

ax.text(0.99, 0.03, f"CV AUC: {cv_auc7:.3f}", transform=ax.transAxes,
        ha="right", fontsize=9, color="#6B7280")

fig.tight_layout()
save(fig, "q7_notability_predictor.png")

# ===========================================================================
# Q5  COST REGRESSION — decomposing training cost drivers
# ===========================================================================
print("Q5 — Cost regression …")

q5 = df.dropna(subset=["log_cost"]).copy()
q5["log_comp_f"]  = q5["log_compute"].fillna(q5["log_compute"].median())
q5["log_hw_f"]    = q5["log_hw"].fillna(q5["log_hw"].median())
q5["year_f"]      = (q5["year"].fillna(q5["year"].median()) - 2010) / 10.0
q5["is_industry"] = (q5["org_type"] == "Industry").astype(int)

FEAT5  = ["log_comp_f", "log_hw_f", "year_f", "is_industry"]
LAB5   = ["log(Compute)", "log(HW Quantity)", "Year (decade)", "Industry org"]

from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

X5 = StandardScaler().fit_transform(q5[FEAT5].values)
y5 = q5["log_cost"].values

ridge = Ridge(alpha=1.0)
ridge.fit(X5, y5)
cv_r2 = cross_val_score(ridge, X5, y5, cv=5, scoring="r2").mean()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("The Economics of AI: What Drives Training Cost?", fontsize=14, y=1.01)

# Left: coefficients
coef5 = pd.Series(ridge.coef_, index=LAB5)
bar_colors5 = ["#DC2626" if c < 0 else "#16A34A" for c in coef5]
axes[0].barh(coef5.index, coef5.values, color=bar_colors5, edgecolor="white", height=0.55)
axes[0].axvline(0, color="#374151", lw=1.2)
axes[0].set_xlabel("Ridge coefficient (standardised features)")
axes[0].set_title(f"Cost Drivers (Ridge Regression, CV R²={cv_r2:.2f})")
for i, (idx, val) in enumerate(coef5.items()):
    axes[0].text(val + (0.01 if val >= 0 else -0.01), i,
                 f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right",
                 fontsize=9, fontweight="bold")

# Right: cost vs compute scatter, coloured by org type
ax = axes[1]
for org, grp in q5.groupby("org_type"):
    ax.scatter(grp["log_comp_f"], grp["log_cost"],
               color=PALETTE.get(org, "#94A3B8"), alpha=0.5, s=25,
               edgecolors="white", linewidth=0.2, label=org)
# Overall trend
mask = np.isfinite(q5["log_comp_f"]) & np.isfinite(q5["log_cost"])
slope, intercept, r, *_ = stats.linregress(q5.loc[mask, "log_comp_f"], q5.loc[mask, "log_cost"])
x_fit = np.linspace(q5["log_comp_f"].min(), q5["log_comp_f"].max(), 200)
ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=2, alpha=0.7)
ax.set_xlabel("log₁₀(Training Compute, FLOP)")
ax.set_ylabel("log₁₀(Training Cost, 2023 USD)")
ax.set_title(f"Cost vs Compute (R={r:.2f})")
ax.legend(title="Org type")

# Annotate headline: $388M GPT-4-scale
top_cost = q5.nlargest(1, "log_cost").iloc[0]
if pd.notna(top_cost["Model"]):
    ax.annotate(f"{top_cost['Model'][:18]}…\n${10**top_cost['log_cost']/1e6:.0f}M",
                xy=(top_cost["log_comp_f"], top_cost["log_cost"]),
                xytext=(-90, -30), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D1D5DB"))

fig.tight_layout()
save(fig, "q5_cost_regression.png")

# ===========================================================================
# Export CSVs for Tableau
# ===========================================================================
print("\nExporting Tableau CSVs …")

# Geography export
geo_cols = ["Model", "Organization", "primary_country", "primary_domain",
            "year", "org_type", "log_parameters", "log_compute",
            "is_frontier", "is_notable", "is_open_weights"]
geo = df[geo_cols].dropna(subset=["primary_country", "year"]).copy()
geo_path = os.path.join(EXPORT, "geo_by_year.csv")
geo.to_csv(geo_path, index=False)
print(f"  [saved] geo_by_year.csv  ({len(geo):,} rows)")

# Model count aggregated
geo_agg = (geo.groupby(["primary_country", "year", "primary_domain", "org_type"])
           .agg(model_count=("Model", "count"),
                frontier_count=("is_frontier", "sum"),
                notable_count=("is_notable", "sum"),
                open_count=("is_open_weights", "sum"))
           .reset_index())
geo_agg.to_csv(os.path.join(EXPORT, "geo_aggregated.csv"), index=False)
print(f"  [saved] geo_aggregated.csv  ({len(geo_agg):,} rows)")

# Frontier timeline
frontier_tl = df[df["is_frontier"]].dropna(subset=["year"]).sort_values("year")[
    ["Model", "Organization", "primary_country", "primary_domain", "year",
     "log_parameters", "log_compute", "log_cost", "org_type", "is_open_weights"]]
frontier_tl.to_csv(os.path.join(EXPORT, "frontier_timeline.csv"), index=False)
print(f"  [saved] frontier_timeline.csv  ({len(frontier_tl):,} rows)")

# Open weights over time
ow = (df[df["year"].between(2010, 2025)]
      .dropna(subset=["is_open_weights", "year", "primary_domain"])
      .groupby(["year", "primary_domain"])
      .agg(open=("is_open_weights", "sum"), total=("is_open_weights", "count"))
      .reset_index())
ow["pct_open"] = ow["open"] / ow["total"] * 100
ow.to_csv(os.path.join(EXPORT, "open_weights_by_year_domain.csv"), index=False)
print(f"  [saved] open_weights_by_year_domain.csv")

print("\nAll done.")
