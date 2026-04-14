"""
CPSC 573 Final Project - Data Cleaning Script
Cleans and transforms the Epoch AI Models dataset.
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from collections import Counter

# Add scripts dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_MODELS_CSV, FRONTIER_CSV, LARGE_SCALE_CSV, NOTABLE_CSV,
    CLEANED_CSV, CLEANING_LOG, CLEANED_DIR, SECTIONS_DIR,
    NUMERIC_COLS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cleaning_log = []

def log_step(step: str, before: int, after: int, details: str):
    entry = {"step": step, "before": before, "after": after, "details": details}
    cleaning_log.append(entry)
    print(f"  [{step}] {before} -> {after}  |  {details}")


def primary_value(series: pd.Series) -> pd.Series:
    """Return the first comma-separated token, stripped."""
    return series.fillna("").astype(str).str.split(",").str[0].str.strip().replace("", np.nan)


def deduplicate_csv_field(val):
    """Remove duplicate tokens in a comma-separated string, preserving order."""
    if pd.isna(val):
        return val
    tokens = [t.strip() for t in str(val).split(",")]
    seen = set()
    unique = []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return ",".join(unique) if unique else np.nan


def primary_org_type(val):
    """Derive a single org_type from the (possibly comma-separated) categorization.
    Uses majority vote; ties broken by first occurrence."""
    if pd.isna(val) or str(val).strip() == "":
        return np.nan
    tokens = [t.strip() for t in str(val).split(",") if t.strip()]
    if not tokens:
        return np.nan
    counts = Counter(tokens)
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(SECTIONS_DIR, exist_ok=True)

    print("Loading datasets...")
    df = pd.read_csv(ALL_MODELS_CSV)
    notable = pd.read_csv(NOTABLE_CSV)
    frontier = pd.read_csv(FRONTIER_CSV)
    large_scale = pd.read_csv(LARGE_SCALE_CSV)

    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"  all_ai_models: {initial_rows} rows x {initial_cols} cols")

    # -----------------------------------------------------------------------
    # 1. Strip whitespace / newlines from all string columns
    # -----------------------------------------------------------------------
    print("\n1. Stripping whitespace and newlines...")
    str_cols = df.select_dtypes(include="object").columns
    newline_count = 0
    for col in str_cols:
        mask = df[col].astype(str).str.contains(r"[\n\r]", regex=True, na=False)
        newline_count += mask.sum()
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan, "NaN": np.nan})

    # Also strip the helper datasets
    for aux in [notable, frontier, large_scale]:
        for col in aux.select_dtypes(include="object").columns:
            aux[col] = aux[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            aux[col] = aux[col].replace({"nan": np.nan, "": np.nan, "NaN": np.nan})

    log_step("strip_whitespace", initial_rows, len(df),
             f"Cleaned {len(str_cols)} string columns; fixed {newline_count} cells with embedded newlines")

    # -----------------------------------------------------------------------
    # 2. Deduplicate model names
    # -----------------------------------------------------------------------
    print("\n2. Deduplicating models...")
    before_dedup = len(df)
    dup_mask = df.duplicated(subset=["Model"], keep=False)
    dup_count = dup_mask.sum()

    if dup_count > 0:
        # For each set of duplicates, keep the row with fewest NaN values
        df["_null_count"] = df.isnull().sum(axis=1)
        df = df.sort_values("_null_count").drop_duplicates(subset=["Model"], keep="first")
        df = df.drop(columns=["_null_count"])

    after_dedup = len(df)
    log_step("deduplicate", before_dedup, after_dedup,
             f"Removed {before_dedup - after_dedup} duplicate entries (kept most complete record)")

    # -----------------------------------------------------------------------
    # 3. Clean country duplicates (e.g., "Australia,Australia" -> "Australia")
    # -----------------------------------------------------------------------
    print("\n3. Cleaning country duplicates...")
    country_col = "Country (of organization)"
    before_country = df[country_col].nunique()
    df[country_col] = df[country_col].apply(deduplicate_csv_field)
    after_country = df[country_col].nunique()
    log_step("clean_country_duplicates", before_country, after_country,
             f"Deduplicated country values from {before_country} to {after_country} unique values")

    # Also clean Organization categorization the same way
    org_cat_col = "Organization categorization"
    df[org_cat_col] = df[org_cat_col].apply(deduplicate_csv_field)

    # -----------------------------------------------------------------------
    # 4. Split comma-separated fields -> primary_* columns
    # -----------------------------------------------------------------------
    print("\n4. Splitting comma-separated fields...")
    df["primary_domain"] = primary_value(df["Domain"])
    df["primary_task"] = primary_value(df["Task"])
    df["primary_country"] = primary_value(df[country_col])
    log_step("split_csv_fields", len(df), len(df),
             f"Created primary_domain ({df['primary_domain'].notna().sum()} non-null), "
             f"primary_task ({df['primary_task'].notna().sum()} non-null), "
             f"primary_country ({df['primary_country'].notna().sum()} non-null)")

    # -----------------------------------------------------------------------
    # 5. Parse dates
    # -----------------------------------------------------------------------
    print("\n5. Parsing dates...")
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    parsed_dates = df["Publication date"].notna().sum()
    df["year"] = df["Publication date"].dt.year.astype("Int64")
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    log_step("parse_dates", len(df), len(df),
             f"Parsed {parsed_dates} of {len(df)} dates successfully; "
             f"created year ({df['year'].notna().sum()} non-null) and decade columns")

    # -----------------------------------------------------------------------
    # 6. Type conversions (numeric columns)
    # -----------------------------------------------------------------------
    print("\n6. Converting numeric columns...")
    numeric_details = []
    for col in NUMERIC_COLS:
        if col in df.columns:
            before_nn = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_nn = df[col].notna().sum()
            numeric_details.append(f"{col}: {after_nn} non-null")
    log_step("type_conversion", len(df), len(df),
             f"Converted {len(numeric_details)} columns to numeric: " +
             "; ".join(numeric_details))

    # -----------------------------------------------------------------------
    # 7. Merge data from other CSVs
    # -----------------------------------------------------------------------
    print("\n7. Merging data from auxiliary CSVs...")

    # 7a. is_frontier flag
    frontier_names = set(frontier["Model"].dropna().unique())
    df["is_frontier"] = df["Model"].isin(frontier_names)
    frontier_count = df["is_frontier"].sum()

    # 7b. is_notable flag
    notable_names = set(notable["Model"].dropna().unique())
    df["is_notable"] = df["Model"].isin(notable_names)
    notable_count = df["is_notable"].sum()

    # 7c. Bring in Organization categorization from notable where missing in main
    notable_org_map = notable.dropna(subset=[org_cat_col]).set_index("Model")[org_cat_col].to_dict()
    mask_missing_org = df[org_cat_col].isna()
    filled_org = 0
    for idx in df.index[mask_missing_org]:
        model = df.at[idx, "Model"]
        if model in notable_org_map:
            df.at[idx, org_cat_col] = notable_org_map[model]
            filled_org += 1

    # 7d. Append models that exist in notable/large_scale but NOT in all_ai_models
    all_model_names = set(df["Model"].dropna().unique())

    missing_notable = notable[~notable["Model"].isin(all_model_names)].copy()
    missing_large = large_scale[~large_scale["Model"].isin(all_model_names | set(missing_notable["Model"].dropna()))].copy()

    # Align columns before concat
    missing_notable["is_notable"] = True
    missing_notable["is_frontier"] = missing_notable["Model"].isin(frontier_names)
    missing_large["is_notable"] = False
    missing_large["is_frontier"] = missing_large["Model"].isin(frontier_names)

    before_merge = len(df)
    df = pd.concat([df, missing_notable, missing_large], ignore_index=True, sort=False)
    appended = len(df) - before_merge

    # Fill is_notable and is_frontier for newly appended rows that may be NaN
    df["is_frontier"] = df["is_frontier"].fillna(False).astype(bool)
    df["is_notable"] = df["is_notable"].fillna(False).astype(bool)

    log_step("merge_auxiliary", before_merge, len(df),
             f"Flagged {frontier_count} frontier, {notable_count} notable models; "
             f"filled {filled_org} missing Organization categorizations; "
             f"appended {appended} models from notable/large_scale CSVs")

    # Re-do primary fields for newly appended rows
    df["primary_domain"] = primary_value(df["Domain"])
    df["primary_task"] = primary_value(df["Task"])
    df["primary_country"] = primary_value(df[country_col])
    df[country_col] = df[country_col].apply(deduplicate_csv_field)
    df[org_cat_col] = df[org_cat_col].apply(deduplicate_csv_field)

    # Re-parse dates for appended rows
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df["year"] = df["Publication date"].dt.year.astype("Int64")
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")

    # Re-coerce numeric for appended rows
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------------------------
    # 8. Create derived columns
    # -----------------------------------------------------------------------
    print("\n8. Creating derived columns...")

    # log_parameters
    mask_params = df["Parameters"].notna() & (df["Parameters"] > 0)
    df["log_parameters"] = np.nan
    df.loc[mask_params, "log_parameters"] = np.log10(df.loc[mask_params, "Parameters"])

    # log_compute
    compute_col = "Training compute (FLOP)"
    mask_compute = df[compute_col].notna() & (df[compute_col] > 0)
    df["log_compute"] = np.nan
    df.loc[mask_compute, "log_compute"] = np.log10(df.loc[mask_compute, compute_col])

    # is_open_weights
    df["is_open_weights"] = df["Open model weights?"].str.lower().str.strip() == "yes"

    # org_type (primary)
    df["org_type"] = df[org_cat_col].apply(primary_org_type)

    derived_details = (
        f"log_parameters ({df['log_parameters'].notna().sum()} non-null), "
        f"log_compute ({df['log_compute'].notna().sum()} non-null), "
        f"is_open_weights ({df['is_open_weights'].sum()} True), "
        f"org_type ({df['org_type'].notna().sum()} non-null)"
    )
    log_step("derived_columns", len(df), len(df), f"Created: {derived_details}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    final_rows = len(df)
    final_cols = len(df.columns)
    print(f"\nFinal dataset: {final_rows} rows x {final_cols} cols")

    log_step("final_summary", initial_rows, final_rows,
             f"Final dataset has {final_rows} rows and {final_cols} columns "
             f"(started with {initial_rows} rows and {initial_cols} cols)")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\nSaving outputs...")
    df.to_csv(CLEANED_CSV, index=False)
    print(f"  Saved cleaned CSV to {CLEANED_CSV}")

    with open(CLEANING_LOG, "w", encoding="utf-8") as f:
        json.dump(cleaning_log, f, indent=2, default=str)
    print(f"  Saved cleaning log to {CLEANING_LOG}")

    # -----------------------------------------------------------------------
    # Generate markdown report section
    # -----------------------------------------------------------------------
    report_path = os.path.join(SECTIONS_DIR, "03_cleaning.md")
    generate_report(report_path, df)
    print(f"  Saved report section to {report_path}")

    print("\nDone!")


def generate_report(path: str, df: pd.DataFrame):
    """Write the cleaning report section as markdown."""
    log = {e["step"]: e for e in cleaning_log}

    md = []
    md.append("## 3. Data Cleaning and Transformation\n")
    md.append("This section describes the cleaning and transformation steps applied to the Epoch AI Models dataset "
              "to produce a unified, analysis-ready dataset.\n")

    md.append("### 3.1 Raw Data Overview\n")
    md.append(f"The primary dataset (`all_ai_models.csv`) contained **{log['strip_whitespace']['before']:,}** rows "
              f"and **57** columns. Three supplementary datasets were also used: "
              f"`notable_ai_models.csv` (981 rows), `frontier_ai_models.csv` (137 rows), and "
              f"`large_scale_ai_models.csv` (491 rows).\n")

    md.append("### 3.2 Cleaning Steps\n")

    # 1 - whitespace
    e = log["strip_whitespace"]
    md.append(f"**Step 1: Whitespace and newline removal.** {e['details']}. "
              f"Model names and other text fields contained embedded `\\n` characters and excess whitespace "
              f"that were normalized to single spaces.\n")

    # 2 - dedup
    e = log["deduplicate"]
    md.append(f"**Step 2: Deduplication.** {e['details']}. "
              f"Duplicate model names were identified, and for each set of duplicates, "
              f"the record with the fewest missing values was retained.\n")

    # 3 - country
    e = log["clean_country_duplicates"]
    md.append(f"**Step 3: Country deduplication.** {e['details']}. "
              f"Values such as \"Australia,Australia\" were collapsed to \"Australia\" while preserving "
              f"genuinely multi-country entries like \"China,United States of America\".\n")

    # 4 - split
    e = log["split_csv_fields"]
    md.append(f"**Step 4: Primary value extraction.** {e['details']}. "
              f"For multi-valued fields (Domain, Task, Country), the first value was extracted into "
              f"`primary_domain`, `primary_task`, and `primary_country` columns to facilitate grouping and visualization.\n")

    # 5 - dates
    e = log["parse_dates"]
    md.append(f"**Step 5: Date parsing.** {e['details']}. "
              f"The `Publication date` column was converted to datetime format, and `year` and `decade` columns "
              f"were derived for temporal analysis.\n")

    # 6 - numeric
    e = log["type_conversion"]
    md.append(f"**Step 6: Numeric type conversion.** {e['details']}.\n")

    # 7 - merge
    e = log["merge_auxiliary"]
    md.append(f"**Step 7: Auxiliary dataset merging.** {e['details']}. "
              f"Boolean indicators `is_frontier` and `is_notable` were added. "
              f"Organization categorization data was backfilled from the notable models dataset where missing in the primary dataset. "
              f"Models present in the supplementary datasets but absent from the primary dataset were appended.\n")

    # 8 - derived
    e = log["derived_columns"]
    md.append(f"**Step 8: Derived columns.** {e['details']}. "
              f"`log_parameters` and `log_compute` provide log10-transformed scales suitable for visualization. "
              f"`is_open_weights` is a boolean indicator derived from the \"Open model weights?\" field. "
              f"`org_type` captures the primary organization type (Industry, Academia, Government, or Research collective) "
              f"using a majority-vote approach on multi-valued categorizations.\n")

    # Summary
    e = log["final_summary"]
    md.append("### 3.3 Cleaning Summary\n")
    md.append(f"{e['details']}. "
              f"The cleaned dataset is saved as `all_ai_models_cleaned.csv` and a machine-readable log of all "
              f"cleaning steps is available in `cleaning_log.json`.\n")

    # Table (no trailing \n so join doesn't create blank lines between rows)
    md.append("| Step | Description | Before | After | Details |")
    md.append("|------|------------|--------|-------|---------|")
    for entry in cleaning_log:
        md.append(f"| {entry['step']} | see above | {entry['before']:,} | {entry['after']:,} | {entry['details'][:80]}{'...' if len(entry['details']) > 80 else ''} |")
    md.append("")  # trailing newline

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
