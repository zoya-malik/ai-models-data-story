"""CPSC 573 Final Project -- Generate 10 annotated charts for the data cleaning report."""
import matplotlib
matplotlib.use("Agg")

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path so config imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from config import (
    ALL_MODELS_CSV, CLEANED_CSV, FIGURES_DIR, NUMERIC_COLS,
    DOMAIN_COLORS, setup_plot_style,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_plot_style()
os.makedirs(FIGURES_DIR, exist_ok=True)

RAW = pd.read_csv(ALL_MODELS_CSV)
CLEANED_EXISTS = os.path.exists(CLEANED_CSV)
if CLEANED_EXISTS:
    CLEANED = pd.read_csv(CLEANED_CSV)
else:
    CLEANED = None

def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")

def _primary_domain(val):
    """Extract the first domain from a comma-separated string."""
    if pd.isna(val):
        return "Unknown"
    return str(val).split(",")[0].strip()

def _primary_country(val):
    if pd.isna(val):
        return "Unknown"
    return str(val).split(",")[0].strip()

def _primary_org_cat(val):
    if pd.isna(val):
        return "Unknown"
    first = str(val).split(",")[0].strip()
    return first

RAW["_domain"] = RAW["Domain"].apply(_primary_domain)
RAW["_country"] = RAW["Country (of organization)"].apply(_primary_country)
RAW["_org_cat"] = RAW["Organization categorization"].apply(_primary_org_cat)
RAW["_year"] = pd.to_datetime(RAW["Publication date"], errors="coerce").dt.year

# ---------------------------------------------------------------------------
# 1. Nullity heatmap
# ---------------------------------------------------------------------------
def fig_01():
    print("Fig 01 -- Nullity heatmap")
    cols = [c for c in RAW.columns if not c.startswith("_")]
    null_matrix = RAW[cols].isnull().astype(int)
    # Subsample rows for visual clarity
    sample = null_matrix.sample(min(200, len(null_matrix)), random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(sample.T, cbar=False, cmap=["#e8f4f8", "#c0392b"],
                yticklabels=True, xticklabels=False, ax=ax, linewidths=0)
    ax.set_title("Missing Data Pattern Across AI Models Dataset", fontsize=15, fontweight="bold")
    ax.set_xlabel("Sampled Records (n=200)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)

    # Annotate high-nullity columns
    null_pct = RAW[cols].isnull().mean()
    high_null = null_pct[null_pct > 0.80].sort_values(ascending=False)
    for col_name, pct in high_null.head(5).items():
        idx = list(cols).index(col_name)
        ax.annotate(f"{pct:.0%} missing", xy=(sample.shape[0], idx + 0.5),
                    xytext=(sample.shape[0] + 15, idx + 0.5),
                    fontsize=7, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    # Also call out Training time and Cost specifically
    for label, col in [("Training time: 83% missing", "Training time (hours)"),
                       ("Cost: 93% missing", "Training compute cost (2023 USD)")]:
        if col in cols:
            idx = list(cols).index(col)
            ax.annotate(label, xy=(0, idx + 0.5),
                        xytext=(-60, idx + 0.5),
                        fontsize=7, fontweight="bold", color="#2c3e50",
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
                        ha="right")

    fig.tight_layout()
    _save(fig, "fig_01_nullity_heatmap.png")


# ---------------------------------------------------------------------------
# 2. Models per year
# ---------------------------------------------------------------------------
def fig_02():
    print("Fig 02 -- Models per year")
    df = RAW.dropna(subset=["_year"]).copy()
    df["_year"] = df["_year"].astype(int)
    # Limit to reasonable range
    df = df[(df["_year"] >= 1950) & (df["_year"] <= 2026)]

    top_domains = df["_domain"].value_counts().head(6).index.tolist()
    df["_domain_grp"] = df["_domain"].where(df["_domain"].isin(top_domains), "Other")

    ct = df.groupby(["_year", "_domain_grp"]).size().unstack(fill_value=0)
    color_map = {d: DOMAIN_COLORS.get(d, "#937860") for d in ct.columns}

    fig, ax = plt.subplots(figsize=(14, 7))
    ct.plot.bar(stacked=True, ax=ax, color=[color_map.get(c, "#aaa") for c in ct.columns],
                edgecolor="white", linewidth=0.3)
    ax.set_title("Number of AI Models Published per Year by Domain", fontsize=15, fontweight="bold")
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Model Count")
    ax.legend(title="Primary Domain", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Annotate inflection points
    if 2017 in ct.index:
        y2017 = ct.loc[2017].sum()
        tick_pos = list(ct.index).index(2017)
        ax.annotate("Transformer era\nbegins (2017)", xy=(tick_pos, y2017),
                    xytext=(tick_pos - 5, y2017 + 80),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    if 2023 in ct.index:
        y2023 = ct.loc[2023].sum()
        tick_pos = list(ct.index).index(2023)
        ax.annotate(f"2023 boom\n({int(y2023)} models)", xy=(tick_pos, y2023),
                    xytext=(tick_pos - 4, y2023 + 60),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, "fig_02_models_per_year.png")


# ---------------------------------------------------------------------------
# 3. Parameter growth over time
# ---------------------------------------------------------------------------
def fig_03():
    print("Fig 03 -- Parameter growth")
    df = RAW.dropna(subset=["Parameters", "Publication date"]).copy()
    df["_date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_log_params"] = np.log10(df["Parameters"].astype(float))
    df = df[df["_log_params"] > 0]

    top_domains = df["_domain"].value_counts().head(6).index.tolist()
    df["_domain_grp"] = df["_domain"].where(df["_domain"].isin(top_domains), "Other")

    fig, ax = plt.subplots(figsize=(14, 8))
    for dom in sorted(df["_domain_grp"].unique()):
        sub = df[df["_domain_grp"] == dom]
        color = DOMAIN_COLORS.get(dom, "#937860")
        ax.scatter(sub["_date"], sub["_log_params"], label=dom,
                   alpha=0.5, s=25, color=color, edgecolors="white", linewidth=0.3)

    # Trend line
    df["_date_num"] = (df["_date"] - df["_date"].min()).dt.days
    mask = np.isfinite(df["_log_params"]) & np.isfinite(df["_date_num"])
    if mask.sum() > 2:
        z = np.polyfit(df.loc[mask, "_date_num"], df.loc[mask, "_log_params"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["_date_num"].min(), df["_date_num"].max(), 100)
        dates_line = pd.to_timedelta(x_line, unit="D") + df["_date"].min()
        ax.plot(dates_line, p(x_line), "k--", lw=2, alpha=0.6, label="Trend")

    ax.set_title("Growth of Model Parameters Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Publication Date")
    ax.set_ylabel("log10(Parameters)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Annotate landmark models
    landmarks = {
        "GPT-3": "GPT-3",
        "GPT-4 ": "GPT-4",
        "Llama 2": "Llama 2",
        "BERT": "BERT",
        "AlexNet": "AlexNet",
    }
    annotated = 0
    for search, label in landmarks.items():
        match = df[df["Model"].str.strip().str.startswith(search, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            offset = (20, 15 + annotated * 18)
            ax.annotate(label, xy=(row["_date"], row["_log_params"]),
                        xytext=offset, textcoords="offset points",
                        fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))
            annotated += 1

    fig.tight_layout()
    _save(fig, "fig_03_parameter_growth.png")


# ---------------------------------------------------------------------------
# 4. Compute vs Parameters (log-log)
# ---------------------------------------------------------------------------
def fig_04():
    print("Fig 04 -- Compute vs Parameters")
    df = RAW.dropna(subset=["Training compute (FLOP)", "Parameters"]).copy()
    df["_log_flop"] = np.log10(df["Training compute (FLOP)"].astype(float))
    df["_log_params"] = np.log10(df["Parameters"].astype(float))
    df = df[(df["_log_flop"] > 0) & (df["_log_params"] > 0)]

    fig, ax = plt.subplots(figsize=(12, 8))
    for cat in sorted(df["_org_cat"].unique()):
        sub = df[df["_org_cat"] == cat]
        ax.scatter(sub["_log_params"], sub["_log_flop"], label=cat,
                   alpha=0.5, s=30, edgecolors="white", linewidth=0.3)

    ax.set_title("Training Compute vs Model Parameters (log-log)", fontsize=15, fontweight="bold")
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("log10(Training Compute FLOP)")
    ax.legend(title="Organization Type", fontsize=9)

    # Annotate the scaling relationship
    if len(df) > 10:
        mask = np.isfinite(df["_log_params"]) & np.isfinite(df["_log_flop"])
        z = np.polyfit(df.loc[mask, "_log_params"], df.loc[mask, "_log_flop"], 1)
        x_fit = np.linspace(df["_log_params"].min(), df["_log_params"].max(), 100)
        ax.plot(x_fit, np.poly1d(z)(x_fit), "k--", lw=2, alpha=0.5)
        ax.annotate(f"Scaling slope: {z[0]:.2f}", xy=(x_fit[60], np.poly1d(z)(x_fit[60])),
                    xytext=(30, -30), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_04_compute_vs_params.png")


# ---------------------------------------------------------------------------
# 5. Domain distribution
# ---------------------------------------------------------------------------
def fig_05():
    print("Fig 05 -- Domain distribution")
    counts = RAW["_domain"].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [DOMAIN_COLORS.get(d, "#937860") for d in counts.index]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title("Distribution of AI Models by Primary Domain", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Models")
    ax.set_ylabel("")

    # Annotate top domain
    ax.annotate(f"Language dominates\nwith {counts.iloc[0]} models ({counts.iloc[0]/len(RAW):.0%})",
                xy=(counts.iloc[0], len(counts) - 1),
                xytext=(counts.iloc[0] - 300, len(counts) - 4),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_05_domain_distribution.png")


# ---------------------------------------------------------------------------
# 6. Country distribution
# ---------------------------------------------------------------------------
def fig_06():
    print("Fig 06 -- Country distribution")
    counts = RAW["_country"].value_counts().head(15)
    # Shorten long names
    rename = {
        "United States of America": "USA",
        "United Kingdom of Great Britain and Northern Ireland": "UK",
        "Korea (Republic of)": "South Korea",
    }
    labels = [rename.get(c, c) for c in counts.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels[::-1], counts.values[::-1], color=sns.color_palette("Set2", len(counts))[::-1],
                   edgecolor="white")
    ax.set_title("Top 15 Countries by AI Model Count", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Models")

    # Annotate USA dominance
    ax.annotate(f"USA leads with {counts.iloc[0]} models",
                xy=(counts.iloc[0], len(counts) - 1),
                xytext=(counts.iloc[0] - 250, len(counts) - 4),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_06_country_distribution.png")


# ---------------------------------------------------------------------------
# 7. Open weights trend
# ---------------------------------------------------------------------------
def fig_07():
    print("Fig 07 -- Open weights trend")
    df = RAW.dropna(subset=["Open model weights?", "_year"]).copy()
    df["_year"] = df["_year"].astype(int)
    df = df[(df["_year"] >= 2010) & (df["_year"] <= 2026)]

    ct = pd.crosstab(df["_year"], df["Open model weights?"])
    # Ensure both columns exist
    for col in ["Yes", "No"]:
        if col not in ct.columns:
            ct[col] = 0

    fig, ax = plt.subplots(figsize=(12, 7))
    ct[["Yes", "No"]].plot.bar(stacked=True, ax=ax,
                                color=["#27ae60", "#e74c3c"],
                                edgecolor="white", linewidth=0.5)
    ax.set_title("Open vs Closed Model Weights Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Models")
    ax.legend(["Open Weights", "Closed Weights"], title="Weights Availability")

    # Annotate open-source surge
    if 2023 in ct.index:
        total = ct.loc[2023].sum()
        open_pct = ct.loc[2023].get("Yes", 0) / total if total > 0 else 0
        tick_pos = list(ct.index).index(2023)
        ax.annotate(f"2023: {open_pct:.0%} open",
                    xy=(tick_pos, total),
                    xytext=(tick_pos - 3, total + 30),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, "fig_07_open_weights_trend.png")


# ---------------------------------------------------------------------------
# 8. Cost distribution
# ---------------------------------------------------------------------------
def fig_08():
    print("Fig 08 -- Cost distribution")
    col = "Training compute cost (2023 USD)"
    df = RAW.dropna(subset=[col]).copy()
    df["_log_cost"] = np.log10(df[col].astype(float))
    df = df[np.isfinite(df["_log_cost"])]

    fig, ax = plt.subplots(figsize=(12, 7))
    # Split by org categorization if enough data
    for cat in sorted(df["_org_cat"].unique()):
        sub = df[df["_org_cat"] == cat]
        if len(sub) >= 3:
            ax.hist(sub["_log_cost"], bins=25, alpha=0.5, label=f"{cat} (n={len(sub)})", edgecolor="white")

    ax.set_title("Distribution of Training Cost (log-scale, 2023 USD)", fontsize=15, fontweight="bold")
    ax.set_xlabel("log10(Training Cost in 2023 USD)")
    ax.set_ylabel("Frequency")
    ax.legend(title="Organization Type", fontsize=9)

    median_cost = df["_log_cost"].median()
    ax.axvline(median_cost, color="black", linestyle="--", lw=1.5)
    ax.annotate(f"Median: ${10**median_cost:,.0f}",
                xy=(median_cost, ax.get_ylim()[1] * 0.8),
                xytext=(median_cost + 0.8, ax.get_ylim()[1] * 0.85),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_08_cost_distribution.png")


# ---------------------------------------------------------------------------
# 9. Cleaning comparison / data quality
# ---------------------------------------------------------------------------
def fig_09():
    print("Fig 09 -- Cleaning comparison / data quality")
    key_cols = ["Parameters", "Training compute (FLOP)", "Training dataset size (total)",
                "Training time (hours)", "Training compute cost (2023 USD)",
                "Domain", "Organization", "Country (of organization)", "Open model weights?"]
    key_cols = [c for c in key_cols if c in RAW.columns]

    raw_nulls = RAW[key_cols].isnull().sum()

    fig, ax = plt.subplots(figsize=(12, 7))

    if CLEANED_EXISTS and CLEANED is not None:
        cleaned_nulls = CLEANED[[c for c in key_cols if c in CLEANED.columns]].isnull().sum()
        x = np.arange(len(key_cols))
        w = 0.35
        ax.barh(x + w / 2, raw_nulls.values, w, label="Raw", color="#e74c3c", edgecolor="white")
        cleaned_vals = [cleaned_nulls.get(c, 0) for c in key_cols]
        ax.barh(x - w / 2, cleaned_vals, w, label="Cleaned", color="#27ae60", edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(key_cols)
        ax.legend()
        title = "Data Quality: Missing Values Before vs After Cleaning"
    else:
        ax.barh(key_cols, raw_nulls.values, color="#3498db", edgecolor="white")
        title = "Data Quality: Missing Values by Column"

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Missing Values")

    # Annotate worst column
    worst_col = raw_nulls.idxmax()
    worst_val = raw_nulls.max()
    worst_idx = list(key_cols).index(worst_col)
    ax.annotate(f"{worst_val} missing ({worst_val/len(RAW):.0%})",
                xy=(worst_val, worst_idx),
                xytext=(worst_val + 100, worst_idx + 1),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_09_cleaning_comparison.png")


# ---------------------------------------------------------------------------
# 10. Correlation heatmap
# ---------------------------------------------------------------------------
def fig_10():
    print("Fig 10 -- Correlation heatmap")
    num_cols = ["Parameters", "Training compute (FLOP)", "Training dataset size (total)",
                "Training time (hours)", "Training compute cost (2023 USD)",
                "Epochs", "Hardware quantity"]
    num_cols = [c for c in num_cols if c in RAW.columns]

    df = RAW[num_cols].copy()
    # Log-transform
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        pos = df[c] > 0
        df.loc[pos, c] = np.log10(df.loc[pos, c])
        df.loc[~pos, c] = np.nan

    corr = df.corr()

    short_names = {
        "Parameters": "Params",
        "Training compute (FLOP)": "Compute",
        "Training dataset size (total)": "Dataset Size",
        "Training time (hours)": "Train Time",
        "Training compute cost (2023 USD)": "Cost",
        "Epochs": "Epochs",
        "Hardware quantity": "HW Qty",
    }
    corr.index = [short_names.get(c, c) for c in corr.index]
    corr.columns = [short_names.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, square=True, ax=ax, linewidths=0.5,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Between Numeric Measures (log-transformed)", fontsize=15, fontweight="bold")

    # Find strongest off-diagonal correlation
    abs_corr = corr.abs().copy()
    vals = abs_corr.values.copy()
    np.fill_diagonal(vals, 0)
    max_idx = np.unravel_index(vals.argmax(), vals.shape)
    col1 = corr.columns[max_idx[1]]
    col2 = corr.index[max_idx[0]]
    val = corr.iloc[max_idx[0], max_idx[1]]
    ax.annotate(f"Strongest: {col1} vs {col2}\nr={val:.2f}",
                xy=(max_idx[1] + 0.5, max_idx[0] + 0.5),
                xytext=(max_idx[1] + 2, max_idx[0] - 1.5),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    fig.tight_layout()
    _save(fig, "fig_10_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Generating 10 visualizations for CPSC 573 Final Project")
    print("=" * 60)
    for fn in [fig_01, fig_02, fig_03, fig_04, fig_05, fig_06, fig_07, fig_08, fig_09, fig_10]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("=" * 60)
    print("Done.")
