"""
CPSC 573 Final Project -- Data Profiling Script
Produces comprehensive profiling statistics for the AI Models dataset.
"""

import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Add project root so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    DATA_DIR, CLEANED_DIR, REPORT_DIR, SECTIONS_DIR,
    ALL_MODELS_CSV, FRONTIER_CSV, LARGE_SCALE_CSV, NOTABLE_CSV,
    NUMERIC_COLS, CATEGORICAL_COLS, PROFILING_JSON,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_json(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj


def load_csv(path):
    """Load a CSV with sensible defaults."""
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# 1. Shape summary
# ---------------------------------------------------------------------------

def shape_summary(datasets: dict) -> dict:
    result = {}
    for name, df in datasets.items():
        result[name] = {"rows": len(df), "columns": len(df.columns)}
    return result


# ---------------------------------------------------------------------------
# 2. Nullity analysis
# ---------------------------------------------------------------------------

def nullity_analysis(df: pd.DataFrame) -> dict:
    total = len(df)
    records = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct_missing = round((1 - non_null / total) * 100, 2) if total > 0 else 0
        records.append({
            "column": col,
            "non_null": int(non_null),
            "pct_missing": pct_missing,
        })
    records.sort(key=lambda r: r["pct_missing"], reverse=True)
    worst = [r for r in records if r["pct_missing"] > 50]
    return {"per_column": records, "worst_columns": worst, "total_rows": total}


# ---------------------------------------------------------------------------
# 3. Classify dimensions vs measures
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> dict:
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    present_categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    other = [c for c in df.columns if c not in present_numeric and c not in present_categorical]
    return {
        "dimensions": present_categorical,
        "measures": present_numeric,
        "other": other,
    }


# ---------------------------------------------------------------------------
# 4. Descriptive statistics for numeric measures
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> dict:
    present = [c for c in NUMERIC_COLS if c in df.columns]
    result = {}
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            result[col] = {"count": 0}
            continue
        result[col] = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "log_mean": float(np.log10(s[s > 0]).mean()) if (s > 0).any() else None,
            "log_std": float(np.log10(s[s > 0]).std()) if (s > 0).any() else None,
        }
    return result


# ---------------------------------------------------------------------------
# 5. Frequency tables
# ---------------------------------------------------------------------------

def frequency_tables(df: pd.DataFrame) -> dict:
    result = {}

    # Top 15 organizations
    if "Organization" in df.columns:
        org_counts = df["Organization"].value_counts().head(15)
        result["top_15_organizations"] = {k: int(v) for k, v in org_counts.items()}

    # Domain breakdown
    if "Domain" in df.columns:
        dom = df["Domain"].value_counts()
        result["domain_breakdown"] = {k: int(v) for k, v in dom.items()}

    # Task breakdown (top 20 -- there are many tasks)
    if "Task" in df.columns:
        task = df["Task"].value_counts().head(20)
        result["task_breakdown_top20"] = {k: int(v) for k, v in task.items()}

    # Country breakdown
    country_col = "Country (of organization)"
    if country_col in df.columns:
        country = df[country_col].value_counts().head(15)
        result["country_breakdown"] = {k: int(v) for k, v in country.items()}

    # Confidence
    if "Confidence" in df.columns:
        conf = df["Confidence"].value_counts()
        result["confidence_breakdown"] = {k: int(v) for k, v in conf.items()}

    # Accessibility
    if "Model accessibility" in df.columns:
        acc = df["Model accessibility"].value_counts()
        result["model_accessibility"] = {k: int(v) for k, v in acc.items()}

    return result


# ---------------------------------------------------------------------------
# 6. Temporal profile
# ---------------------------------------------------------------------------

def temporal_profile(df: pd.DataFrame) -> dict:
    if "Publication date" not in df.columns:
        return {}

    dates = pd.to_datetime(df["Publication date"], errors="coerce")
    years = dates.dt.year.dropna().astype(int)
    models_per_year = years.value_counts().sort_index()
    mpy = {int(k): int(v) for k, v in models_per_year.items()}

    # Coverage change over time -- for key columns
    key_cols = [
        "Parameters", "Training compute (FLOP)",
        "Training dataset size (total)", "Training time (hours)",
        "Training compute cost (2023 USD)",
    ]
    coverage_by_year = {}
    df_with_year = df.copy()
    df_with_year["_year"] = dates.dt.year
    for col in key_cols:
        if col not in df.columns:
            continue
        cov = {}
        for yr, grp in df_with_year.groupby("_year"):
            if pd.isna(yr):
                continue
            total = len(grp)
            non_null = grp[col].notna().sum()
            cov[int(yr)] = round(non_null / total * 100, 1) if total > 0 else 0
        coverage_by_year[col] = cov

    return {
        "models_per_year": mpy,
        "column_coverage_by_year": coverage_by_year,
    }


# ---------------------------------------------------------------------------
# 7. Cross-file comparison
# ---------------------------------------------------------------------------

def cross_file_comparison(datasets: dict) -> dict:
    # Column overlap
    all_cols = {}
    for name, df in datasets.items():
        all_cols[name] = set(df.columns.tolist())

    union = set()
    for s in all_cols.values():
        union |= s
    intersection = union.copy()
    for s in all_cols.values():
        intersection &= s

    unique_per_file = {}
    for name, cols in all_cols.items():
        others = set()
        for n2, c2 in all_cols.items():
            if n2 != name:
                others |= c2
        unique_per_file[name] = sorted(cols - others)

    # Model overlap (by Model name)
    model_sets = {}
    for name, df in datasets.items():
        if "Model" in df.columns:
            model_sets[name] = set(df["Model"].dropna().str.strip().tolist())

    overlap = {}
    names = list(model_sets.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            common = model_sets[n1] & model_sets[n2]
            overlap[f"{n1} & {n2}"] = len(common)

    return {
        "shared_columns": sorted(intersection),
        "shared_column_count": len(intersection),
        "union_column_count": len(union),
        "unique_columns_per_file": {k: v for k, v in unique_per_file.items()},
        "model_overlap": overlap,
        "model_counts": {n: len(s) for n, s in model_sets.items()},
    }


# ---------------------------------------------------------------------------
# 8. Correlation matrix (log-transformed)
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame) -> dict:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")

    # Log-transform positive values
    log_sub = sub.copy()
    for c in log_sub.columns:
        pos = log_sub[c] > 0
        log_sub.loc[pos, c] = np.log10(log_sub.loc[pos, c])
        log_sub.loc[~pos, c] = np.nan

    corr = log_sub.corr()
    # Convert to nested dict
    corr_dict = {}
    for c1 in corr.columns:
        corr_dict[c1] = {}
        for c2 in corr.columns:
            v = corr.loc[c1, c2]
            corr_dict[c1][c2] = round(float(v), 3) if not pd.isna(v) else None

    return corr_dict


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------

def generate_markdown(results: dict, cleaned_results: dict = None) -> str:
    lines = []
    a = lines.append

    a("## 4. Data Profiling")
    a("")
    a("This section presents a comprehensive profile of the raw AI Models dataset prior to cleaning. "
      "The goal is to characterize the structure, completeness, distributions, and quality of the data "
      "that feeds into subsequent analysis.")
    a("")

    # 4.1 Dataset dimensions
    a("### 4.1 Dataset Dimensions")
    a("")
    a("| File | Rows | Columns |")
    a("|------|-----:|--------:|")
    for name, info in results["shape"].items():
        a(f"| {name} | {info['rows']:,} | {info['columns']} |")
    a("")
    a("The primary dataset (`all_ai_models.csv`) contains {:,} models across {} columns, ".format(
        results["shape"]["all_ai_models"]["rows"],
        results["shape"]["all_ai_models"]["columns"],
    ))
    a("making it the most comprehensive of the four files. The frontier, large-scale, and notable "
      "subsets are curated views with fewer rows but partially overlapping columns.")
    a("")

    # 4.2 Nullity analysis
    a("### 4.2 Missing-Value Analysis")
    a("")
    nullity = results["nullity"]
    a(f"Out of {nullity['total_rows']:,} rows, the per-column completeness varies dramatically:")
    a("")
    a("| Column | Non-Null | % Missing |")
    a("|--------|--------:|---------:|")
    for r in nullity["per_column"]:
        a(f"| {r['column']} | {r['non_null']:,} | {r['pct_missing']:.1f}% |")
    a("")
    a("**Worst columns (>50% missing):**")
    a("")
    for r in nullity["worst_columns"]:
        a(f"- **{r['column']}**: {r['pct_missing']:.1f}% missing")
    a("")
    a("The high missingness in training cost (~93%), training time (~83%), and dataset size (~59%) "
      "reflects the reality that these details are frequently unreported in AI research publications. "
      "This has important implications for any analysis relying on these fields.")
    a("")

    # 4.3 Column classification
    a("### 4.3 Column Classification")
    a("")
    cl = results["column_classification"]
    a("**Dimensions (categorical):** " + ", ".join(cl["dimensions"]))
    a("")
    a("**Measures (numeric):** " + ", ".join(cl["measures"]))
    a("")
    a(f"**Other columns:** {len(cl['other'])} columns including model name, authors, abstract, "
      "notes fields, and metadata.")
    a("")

    # 4.4 Descriptive statistics
    a("### 4.4 Descriptive Statistics for Numeric Measures")
    a("")
    a("All numeric measures exhibit extreme right skew, consistent with log-normal distributions "
      "spanning many orders of magnitude.")
    a("")
    a("| Measure | Count | Mean | Median | Std | Min | Max | Skew | Kurt |")
    a("|---------|------:|-----:|-------:|----:|----:|----:|-----:|-----:|")
    for col, st in results["descriptive_stats"].items():
        if st.get("count", 0) == 0:
            continue
        def fmt(v):
            if v is None:
                return "N/A"
            av = abs(v)
            if av == 0:
                return "0"
            if av >= 1e12:
                return f"{v:.2e}"
            if av >= 1e6:
                return f"{v:,.0f}"
            if av >= 100:
                return f"{v:,.1f}"
            if av >= 1:
                return f"{v:.2f}"
            return f"{v:.4f}"
        a(f"| {col} | {st['count']:,} | {fmt(st['mean'])} | {fmt(st['median'])} | "
          f"{fmt(st['std'])} | {fmt(st['min'])} | {fmt(st['max'])} | "
          f"{fmt(st['skewness'])} | {fmt(st['kurtosis'])} |")
    a("")
    a("The extreme skewness and kurtosis values confirm that raw-scale statistics are dominated "
      "by outliers. Log-scale analysis is essential for meaningful comparisons.")
    a("")

    # 4.5 Frequency tables
    a("### 4.5 Frequency Distributions")
    a("")

    freq = results["frequency_tables"]
    if "top_15_organizations" in freq:
        a("#### Top 15 Organizations by Model Count")
        a("")
        a("| Organization | Models |")
        a("|-------------|-------:|")
        for org, cnt in freq["top_15_organizations"].items():
            a(f"| {org} | {cnt} |")
        a("")

    if "domain_breakdown" in freq:
        a("#### Domain Breakdown")
        a("")
        a("| Domain | Models |")
        a("|--------|-------:|")
        total_models = sum(freq["domain_breakdown"].values())
        for dom, cnt in freq["domain_breakdown"].items():
            pct = cnt / total_models * 100
            a(f"| {dom} | {cnt} ({pct:.1f}%) |")
        a("")

    if "country_breakdown" in freq:
        a("#### Country Breakdown (Top 15)")
        a("")
        a("| Country | Models |")
        a("|---------|-------:|")
        for country, cnt in freq["country_breakdown"].items():
            a(f"| {country} | {cnt} |")
        a("")

    # 4.6 Temporal profile
    a("### 4.6 Temporal Patterns")
    a("")
    temp = results["temporal_profile"]
    if "models_per_year" in temp:
        a("#### Models Published Per Year")
        a("")
        a("| Year | Models |")
        a("|------|-------:|")
        for yr, cnt in sorted(temp["models_per_year"].items(), key=lambda x: int(x[0])):
            a(f"| {yr} | {cnt} |")
        a("")
        a("The dataset shows exponential growth in AI model publication, with a sharp "
          "acceleration from 2017 onward coinciding with the transformer revolution.")
        a("")

    if "column_coverage_by_year" in temp and temp["column_coverage_by_year"]:
        a("#### Column Coverage Over Time")
        a("")
        a("Coverage (% non-null) for key measures, by publication year:")
        a("")
        # Build a table with years as rows and columns as columns
        cov = temp["column_coverage_by_year"]
        col_names = list(cov.keys())
        all_years = sorted(set().union(*(set(v.keys()) for v in cov.values())))
        # Show only recent decades
        recent = [y for y in all_years if y >= 2010]
        if recent:
            header = "| Year | " + " | ".join(col_names) + " |"
            sep = "|------|" + "|".join(["------:" for _ in col_names]) + "|"
            a(header)
            a(sep)
            for yr in recent:
                vals = []
                for c in col_names:
                    v = cov[c].get(yr, None)
                    vals.append(f"{v}%" if v is not None else "N/A")
                a(f"| {yr} | " + " | ".join(vals) + " |")
            a("")
            a("Coverage generally improves over time, though cost and training time remain "
              "poorly reported even for recent models.")
            a("")

    # 4.7 Cross-file comparison
    a("### 4.7 Cross-File Comparison")
    a("")
    xf = results["cross_file"]
    a(f"- **Shared columns across all 4 files:** {xf['shared_column_count']}")
    a(f"- **Total unique columns (union):** {xf['union_column_count']}")
    a("")
    if xf.get("unique_columns_per_file"):
        a("**Unique columns per file** (not found in other files):")
        a("")
        for fname, cols in xf["unique_columns_per_file"].items():
            if cols:
                a(f"- `{fname}`: {', '.join(cols)}")
        a("")
    if xf.get("model_overlap"):
        a("**Model overlap between files:**")
        a("")
        a("| Pair | Shared Models |")
        a("|------|-------------:|")
        for pair, cnt in xf["model_overlap"].items():
            a(f"| {pair} | {cnt:,} |")
        a("")

    # 4.8 Correlation matrix
    a("### 4.8 Correlation Analysis (Log-Transformed)")
    a("")
    a("Pearson correlations computed on log10-transformed values for positive entries:")
    a("")
    corr = results["correlation_matrix"]
    cols_order = list(corr.keys())
    if cols_order:
        # Abbreviate column names for table readability
        short = {}
        for c in cols_order:
            s = c.replace("Training ", "Tr. ").replace("compute ", "comp. ").replace(" (FLOP)", "").replace("(total)", "").replace("(2023 USD)", "$").replace("(hours)", "h").replace("dataset size", "data").replace("(MFU)", "MFU").replace("(HFU)", "HFU").replace("power draw (W)", "power")
            short[c] = s
        header = "| | " + " | ".join(short[c] for c in cols_order) + " |"
        sep = "|---|" + "|".join(["---:" for _ in cols_order]) + "|"
        a(header)
        a(sep)
        for c1 in cols_order:
            vals = []
            for c2 in cols_order:
                v = corr[c1].get(c2)
                vals.append(f"{v:.2f}" if v is not None else "N/A")
            a(f"| {short[c1]} | " + " | ".join(vals) + " |")
        a("")
        a("Parameters, training compute, and dataset size show strong positive correlations on "
          "log scale, confirming well-known scaling relationships in AI research.")
        a("")

    # 4.9 Notable findings
    a("### 4.9 Key Findings and Data Quality Issues")
    a("")
    a("1. **Severe missingness in cost and resource columns**: Training cost is missing for ~93% of "
      "models, training time for ~83%, and dataset size for ~59%. These gaps limit quantitative "
      "analysis of resource requirements.")
    a("")
    a("2. **Extreme value ranges**: Numeric measures span many orders of magnitude (e.g., parameters "
      "range from single digits to hundreds of billions). Log-transformation is essential for "
      "meaningful statistical analysis.")
    a("")
    a("3. **US dominance**: The United States is the leading country of origin for AI models, "
      "followed by China and the United Kingdom.")
    a("")
    a("4. **Language models dominate**: The Language domain accounts for the largest share of models, "
      "reflecting the NLP focus of recent AI research.")
    a("")
    a("5. **Exponential growth**: Model publication rates have grown exponentially, with the "
      "post-2017 transformer era seeing the steepest increase.")
    a("")
    a("6. **Column schema differences**: The four CSV files share a common core of columns but "
      "each has unique fields, requiring careful alignment for cross-file analysis.")
    a("")

    # Cleaned data comparison if available
    if cleaned_results:
        a("### 4.10 Raw vs. Cleaned Data Comparison")
        a("")
        raw_shape = results["shape"]["all_ai_models"]
        cln_shape = cleaned_results["shape"]["all_ai_models_cleaned"]
        a(f"- **Raw**: {raw_shape['rows']:,} rows, {raw_shape['columns']} columns")
        a(f"- **Cleaned**: {cln_shape['rows']:,} rows, {cln_shape['columns']} columns")
        a(f"- **Rows removed during cleaning**: {raw_shape['rows'] - cln_shape['rows']:,}")
        a("")
        # Compare nullity
        raw_null = {r["column"]: r["pct_missing"] for r in results["nullity"]["per_column"]}
        cln_null = {r["column"]: r["pct_missing"] for r in cleaned_results["nullity"]["per_column"]}
        common_cols = set(raw_null.keys()) & set(cln_null.keys())
        improved = [(c, raw_null[c], cln_null[c]) for c in common_cols
                     if cln_null[c] < raw_null[c] - 0.5]
        if improved:
            improved.sort(key=lambda x: x[1] - x[2], reverse=True)
            a("**Columns with improved completeness after cleaning:**")
            a("")
            a("| Column | Raw % Missing | Cleaned % Missing | Improvement |")
            a("|--------|-------------:|------------------:|------------:|")
            for col, raw_pct, cln_pct in improved[:15]:
                a(f"| {col} | {raw_pct:.1f}% | {cln_pct:.1f}% | {raw_pct - cln_pct:.1f}pp |")
            a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading datasets...")
    datasets_raw = {
        "all_ai_models": load_csv(ALL_MODELS_CSV),
        "frontier_ai_models": load_csv(FRONTIER_CSV),
        "large_scale_ai_models": load_csv(LARGE_SCALE_CSV),
        "notable_ai_models": load_csv(NOTABLE_CSV),
    }

    df = datasets_raw["all_ai_models"]
    print(f"  all_ai_models: {df.shape}")

    results = {}

    print("1. Shape summary...")
    results["shape"] = shape_summary(datasets_raw)

    print("2. Nullity analysis...")
    results["nullity"] = nullity_analysis(df)

    print("3. Column classification...")
    results["column_classification"] = classify_columns(df)

    print("4. Descriptive statistics...")
    results["descriptive_stats"] = descriptive_stats(df)

    print("5. Frequency tables...")
    results["frequency_tables"] = frequency_tables(df)

    print("6. Temporal profile...")
    results["temporal_profile"] = temporal_profile(df)

    print("7. Cross-file comparison...")
    results["cross_file"] = cross_file_comparison(datasets_raw)

    print("8. Correlation matrix...")
    results["correlation_matrix"] = correlation_matrix(df)

    # Check for cleaned data
    cleaned_results = None
    cleaned_path = os.path.join(CLEANED_DIR, "all_ai_models_cleaned.csv")
    if os.path.exists(cleaned_path):
        print("Found cleaned data -- profiling for comparison...")
        df_clean = load_csv(cleaned_path)
        cleaned_results = {
            "shape": {"all_ai_models_cleaned": {"rows": len(df_clean), "columns": len(df_clean.columns)}},
            "nullity": nullity_analysis(df_clean),
            "descriptive_stats": descriptive_stats(df_clean),
        }
        results["cleaned_comparison"] = cleaned_results

    # Write JSON
    print(f"Writing {PROFILING_JSON}...")
    os.makedirs(os.path.dirname(PROFILING_JSON), exist_ok=True)
    with open(PROFILING_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_json(results), f, indent=2, default=str)

    # Write markdown
    md_path = os.path.join(SECTIONS_DIR, "04_profiling.md")
    print(f"Writing {md_path}...")
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    md = generate_markdown(results, cleaned_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("Done. Profiling complete.")


if __name__ == "__main__":
    main()
