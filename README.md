# 70 Years of AI: A Data Story

An end-to-end data analysis and visualization project examining 3,305 AI models from 1950 to 2025 using the [Epoch AI public models dataset](https://epoch.ai/data/ai-models).

**Live blog post →** [zoya-malik.github.io/ai-models-data-story](https://zoya-malik.github.io/ai-models-data-story)

---

## Four Research Questions

**Q1 — Volume & Scale:** How has AI model output evolved from 1950 to 2025, and has the relationship between model scale and training compute remained consistent over time?

**Q2 — Geography:** Which countries produce AI models, and is geographic concentration at the frontier increasing or spreading over time?

**Q3 — Openness:** What factors most strongly predict whether a model releases its weights publicly, and how has the open vs. closed balance shifted since 2012?

**Q4 — Resource Tiers & Impact:** Do AI models cluster into distinct resource tiers, or exist on a smooth spectrum — and what factors best predict a model's notable impact on the field?

---

## Repository Structure

```
docs/               Blog post (GitHub Pages site)
  index.html        Main data story
  figures/          All visualizations
  CPSC_573_Data_Cleaning_Report.pdf

scripts/            Python analysis pipeline
  clean_data.py     Data cleaning & augmentation
  profile_data.py   Exploratory data profiling
  analysis.py       All research question analyses
  visualizations.py Figure generation
  gephi_export.py   Network graph export
  config.py         Shared configuration

data/
  cleaned/          Cleaned dataset (3,305 rows, 76 columns)
  gephi/            Network nodes & edges CSVs
  tableau_exports/  CSVs used in Tableau interactive
```

---

## Key Findings

- Model output grew from ~10/year (pre-2017) to **944 in 2024 alone**, with a Transformer-driven inflection in 2017
- Scaling laws **fractured post-2020** — compute and parameters decoupled as efficiency techniques matured
- US share of all models fell from 60% → 38% (2018–2024), but the US still holds **two-thirds of frontier releases**
- Country of origin is a genuine predictor of domain specialization (χ²(81) = 1,297, p < 0.001, Cramér's V = 0.23)
- **Organization type** is the single strongest predictor of closed weights — stronger than model size or compute
- Llama 2 spawned **55 documented derivatives**; open-source models became infrastructure for the field

---

## Tools & Methods

| Tool | Purpose |
|------|---------|
| Python (pandas, scikit-learn, scipy, networkx) | Data cleaning, analysis, visualization |
| Tableau | Interactive geography visualization |
| Gephi / NetworkX | Model lineage network (3,305 nodes, 722 edges) |
| Matplotlib / Seaborn | Static figures |

---

## Data Source

Epoch AI — *Tracking the Training of Large AI Models*  
[epoch.ai/data/ai-models](https://epoch.ai/data/ai-models)

Large raw source files are excluded from this repo. The cleaned dataset (`data/cleaned/all_ai_models_cleaned.csv`) is included.

---

*CPSC 573 — Data Visualization · University of Calgary · April 2026*  
*Zoya Malik · zmalik9692@gmail.com*
