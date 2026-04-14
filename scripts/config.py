"""Shared configuration for CPSC 573 Final Project - AI Models Data Cleaning Report."""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
REPORT_DIR = os.path.join(BASE_DIR, "report")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
SECTIONS_DIR = os.path.join(REPORT_DIR, "sections")

# Raw data files
ALL_MODELS_CSV = os.path.join(DATA_DIR, "all_ai_models.csv")
FRONTIER_CSV = os.path.join(DATA_DIR, "frontier_ai_models.csv")
LARGE_SCALE_CSV = os.path.join(DATA_DIR, "large_scale_ai_models.csv")
NOTABLE_CSV = os.path.join(DATA_DIR, "notable_ai_models.csv")

# Cleaned data output
CLEANED_CSV = os.path.join(CLEANED_DIR, "all_ai_models_cleaned.csv")
CLEANING_LOG = os.path.join(CLEANED_DIR, "cleaning_log.json")
PROFILING_JSON = os.path.join(DATA_DIR, "profiling_results.json")

# Key column groups
NUMERIC_COLS = [
    "Parameters",
    "Training compute (FLOP)",
    "Training dataset size (total)",
    "Training time (hours)",
    "Training compute cost (2023 USD)",
    "Epochs",
    "Hardware quantity",
    "Citations",
    "Hardware utilization (MFU)",
    "Hardware utilization (HFU)",
    "Training power draw (W)",
]

CATEGORICAL_COLS = [
    "Domain",
    "Task",
    "Organization",
    "Country (of organization)",
    "Confidence",
    "Model accessibility",
    "Organization categorization",
    "Frontier model",
    "Open model weights?",
    "Approach",
    "Training hardware",
    "Numerical format",
]

# Visualization style
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = sns.color_palette("Set2", 10)
DOMAIN_COLORS = {
    "Language": "#4C72B0",
    "Vision": "#DD8452",
    "Multimodal": "#55A868",
    "Audio": "#C44E52",
    "Games": "#8172B3",
    "Other": "#937860",
    "Drawing": "#DA8BC3",
    "Speech": "#8C8C8C",
    "Robotics": "#CCB974",
    "Biology": "#64B5CD",
}

def setup_plot_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })
