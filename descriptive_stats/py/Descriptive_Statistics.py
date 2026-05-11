# Descriptive Statistics & Percentage Breakdowns
# Per-item distributions, segment crosstabs, and Likert-friendly visualizations.
# Companion to Analysis_Exploration.py. See README.md for methodology overview.

# Imports & wave selection
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent           # descriptive_stats/py
ANALYSIS_DIR = HERE.parent                       # descriptive_stats
ROOT = ANALYSIS_DIR.parent                       # project root

WAVE_RE = re.compile(r"UXR_Survey_Results_(\d{4}-\d{2})\.xlsx$")

def discover_waves(root):
    return sorted(m.group(1) for f in root.glob("UXR_Survey_Results_*.xlsx")
                  if (m := WAVE_RE.search(f.name)))

parser = argparse.ArgumentParser(description="Per-wave descriptive statistics")
parser.add_argument("--wave", default=None,
                    help="YYYY-MM wave to analyze. Defaults to latest available.")
args, _ = parser.parse_known_args()

available = discover_waves(ROOT)
if not available:
    raise SystemExit(
        f"No UXR_Survey_Results_YYYY-MM.xlsx files found in {ROOT}.\n"
        f"Place a wave file matching UXR_Survey_Results_<YYYY-MM>.xlsx at the "
        f"project root and re-run."
    )
WAVE = args.wave or available[-1]
if WAVE not in available:
    raise SystemExit(f"Wave {WAVE} not found. Available: {available}")

DATA = ROOT / f"UXR_Survey_Results_{WAVE}.xlsx"
OUT = ANALYSIS_DIR / "csv" / WAVE
CHARTS = ANALYSIS_DIR / "png" / WAVE
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
print(f"Wave: {WAVE}\nData: {DATA}\nOut:  {OUT}\n")

# Load & align scales
df = pd.read_excel(DATA).iloc[1:].reset_index(drop=True)  # drop SurveyMonkey header row
N = len(df)
print(f"N = {N}")

ITEMS = {
    14: ("Ease of Use", "Experience"),
    15: ("Discovery", "Experience"),
    16: ("Emotional Experience", "Experience"),
    17: ("Personalization", "Experience"),
    18: ("Social Connection", "Experience"),
    19: ("Visual Experience", "Experience"),
    20: ("Experience Value", "CPV"),
    21: ("Feature Value", "CPV"),
    22: ("Participation Safety", "Trust & Safety"),
    23: ("Enforcement Balance", "Trust & Safety"),
}
USAGE_COL, VRCPLUS_COL = 24, 25

# Favorability axis: 1=most unfavorable, 5=most favorable. Used for the standard
# Likert items and Participation Safety. Enforcement Balance handled separately.
FAVORABILITY_ORDER_AGREE = [
    "Strongly disagree", "Disagree", "Neither agree nor disagree",
    "Agree", "Strongly agree",
]
FAVORABILITY_ORDER_EASE = [
    "Very difficult", "Somewhat difficult", "Neutral",
    "Somewhat easy", "Very easy",
]
ENFORCEMENT_ORDER = [
    "Far too little enforcement (harmful behavior often goes unchecked)",
    "Slightly too little enforcement",
    "About the right level",
    "Slightly too much enforcement",
    "Far too much enforcement (content or users are restricted unnecessarily)",
]
ENFORCEMENT_LABELS = [
    "Far too little", "Slightly too little", "About right",
    "Slightly too much", "Far too much",
]

# Standard 5-point favorability labels for the 9 monotonic items
FAVORABILITY_LABELS = ["Most negative", "Negative", "Neutral", "Positive", "Most positive"]

# Build a tidy distribution table — % per response option, by item

def build_distribution(col_idx, label):
    """Return a dict: item, n, %distribution by favorability bucket, T2B, B2B, mean."""
    series = df.iloc[:, col_idx].dropna()
    if col_idx in (14, 15, 16, 17, 18, 19, 20, 21):
        order = FAVORABILITY_ORDER_AGREE
    elif col_idx == 22:
        order = FAVORABILITY_ORDER_EASE
    else:  # col 23 — handled by the dedicated function below
        return None
    counts = series.value_counts().reindex(order).fillna(0).astype(int)
    pct = (counts / counts.sum() * 100).round(1)
    numeric = (pd.Series(range(1, 6), index=order)).reindex(series.values).reset_index(drop=True)
    mean = float(numeric.mean())
    median = float(numeric.median())
    return {
        "item": label,
        "n": int(counts.sum()),
        FAVORABILITY_LABELS[0]: pct.iloc[0],
        FAVORABILITY_LABELS[1]: pct.iloc[1],
        FAVORABILITY_LABELS[2]: pct.iloc[2],
        FAVORABILITY_LABELS[3]: pct.iloc[3],
        FAVORABILITY_LABELS[4]: pct.iloc[4],
        "B2B_pct": float((pct.iloc[0] + pct.iloc[1]).round(1)),
        "Neutral_pct": float(pct.iloc[2]),
        "T2B_pct": float((pct.iloc[3] + pct.iloc[4]).round(1)),
        "Mean": round(mean, 2),
        "Median": median,
    }

monotonic = [build_distribution(i, ITEMS[i][0]) for i in range(14, 23)]
dist_df = pd.DataFrame(monotonic)
dist_df.insert(1, "Pillar", [ITEMS[i + 14][1] for i in range(len(monotonic))])
print("\nPer-item percentage distribution (favorability axis):")
print(dist_df.to_string(index=False))
dist_df.to_csv(OUT / "descriptives_distribution.csv", index=False)

# Enforcement Balance — Goldilocks, separate handling
ef = df.iloc[:, 23].dropna()
ef_counts = ef.value_counts().reindex(ENFORCEMENT_ORDER).fillna(0).astype(int)
ef_pct = (ef_counts / ef_counts.sum() * 100).round(1)
ef_table = pd.DataFrame({
    "Response": ENFORCEMENT_LABELS,
    "n": ef_counts.values,
    "%": ef_pct.values,
})
ef_table.loc[len(ef_table)] = ["Want MORE enforcement (subtotal)",
                                int(ef_counts.iloc[0] + ef_counts.iloc[1]),
                                round(float(ef_pct.iloc[0] + ef_pct.iloc[1]), 1)]
ef_table.loc[len(ef_table)] = ["About right",
                                int(ef_counts.iloc[2]),
                                round(float(ef_pct.iloc[2]), 1)]
ef_table.loc[len(ef_table)] = ["Want LESS enforcement (subtotal)",
                                int(ef_counts.iloc[3] + ef_counts.iloc[4]),
                                round(float(ef_pct.iloc[3] + ef_pct.iloc[4]), 1)]
print("\nEnforcement Balance distribution:")
print(ef_table.to_string(index=False))
ef_table.to_csv(OUT / "descriptives_enforcement.csv", index=False)

# Segmentation question distributions
usage_dist = df.iloc[:, USAGE_COL].value_counts(normalize=True).mul(100).round(1)
vrcplus_dist = df.iloc[:, VRCPLUS_COL].value_counts(normalize=True).mul(100).round(1)
print("\nUsage Frequency distribution (%):")
print(usage_dist.to_string())
print("\nVRC+ Subscription distribution (%):")
print(vrcplus_dist.to_string())

usage_dist.to_csv(OUT / "descriptives_usage.csv", header=["%"])
vrcplus_dist.to_csv(OUT / "descriptives_vrcplus.csv", header=["%"])

# T2B / B2B leaderboard chart — what's working, what's not
plot_df = dist_df.sort_values("T2B_pct", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(plot_df))
ax.barh(y - 0.2, plot_df["T2B_pct"], height=0.4, color="#2ca02c", label="Top-2-Box (positive)")
ax.barh(y + 0.2, plot_df["B2B_pct"], height=0.4, color="#d62728", label="Bottom-2-Box (negative)")
for i, row in plot_df.iterrows():
    ax.text(row["T2B_pct"] + 1, i - 0.2, f'{row["T2B_pct"]}%', va="center", fontsize=8)
    ax.text(row["B2B_pct"] + 1, i + 0.2, f'{row["B2B_pct"]}%', va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(plot_df["item"])
ax.set_xlabel("% of respondents")
ax.set_title("What's working / what's not — Top-2-Box vs Bottom-2-Box per item")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "descr_t2b_b2b_leaderboard.png", dpi=120)
plt.close(fig)

# Diverging stacked bar chart — standard Likert visualization
# Each item shows full distribution with neutral centered at 0
fig, ax = plt.subplots(figsize=(11, 5))
items_order = dist_df.sort_values("T2B_pct", ascending=True)["item"].tolist()
plot_df = dist_df.set_index("item").loc[items_order]

neg_strong = -plot_df["Most negative"]
neg = -plot_df["Negative"]
neu = plot_df["Neutral_pct"] / 2  # split neutral around 0
pos = plot_df["Positive"]
pos_strong = plot_df["Most positive"]

y = np.arange(len(plot_df))
ax.barh(y, neg_strong, color="#67000d", label="Most negative", left=-plot_df["Negative"] - plot_df["Neutral_pct"] / 2)
ax.barh(y, neg, color="#d62728", label="Negative", left=-plot_df["Neutral_pct"] / 2)
ax.barh(y, plot_df["Neutral_pct"], color="#cccccc", label="Neutral", left=-plot_df["Neutral_pct"] / 2)
ax.barh(y, pos, color="#67c067", label="Positive", left=plot_df["Neutral_pct"] / 2)
ax.barh(y, pos_strong, color="#1a7a1a", label="Most positive", left=plot_df["Neutral_pct"] / 2 + pos)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(plot_df.index)
ax.set_xlabel("← Negative   |   Positive →   (% of respondents)")
ax.set_title("How respondents answered each question (sorted by Top-2-Box ascending)")
ax.legend(loc="lower right", ncol=5, fontsize=8, bbox_to_anchor=(1.0, -0.18))
ax.grid(axis="x", alpha=0.3)
ax.set_xticks(np.arange(-100, 101, 25))
ax.set_xticklabels([f"{abs(x)}%" for x in np.arange(-100, 101, 25)])
fig.tight_layout()
fig.savefig(CHARTS / "descr_diverging_stacked.png", dpi=120)
plt.close(fig)

# Enforcement Balance — directional bar chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#67000d", "#d62728", "#1a7a1a", "#5a8fce", "#1f4ea1"]
ax.bar(ENFORCEMENT_LABELS, ef_pct.values, color=colors)
for i, v in enumerate(ef_pct.values):
    ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9)
ax.set_ylabel("% of respondents")
ax.set_title("Enforcement Balance — what does the right level look like?\n"
             f"{round(float(ef_pct.iloc[0] + ef_pct.iloc[1]), 1)}% want more · "
             f"{round(float(ef_pct.iloc[2]), 1)}% about right · "
             f"{round(float(ef_pct.iloc[3] + ef_pct.iloc[4]), 1)}% want less")
ax.set_ylim(0, max(ef_pct.values) * 1.2)
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
fig.savefig(CHARTS / "descr_enforcement_balance.png", dpi=120)
plt.close(fig)

# T2B by segment — Usage Frequency
USAGE_BINS = {
    "Every day": "High", "A few times a week": "High",
    "Once a week": "Medium", "A few times a month": "Medium",
    "Once a month": "Low",
}
df["Usage Bin"] = df.iloc[:, USAGE_COL].map(lambda v: USAGE_BINS.get(v, "Low"))

def t2b_by_segment(seg_col_name, monotonic_cols):
    rows = []
    for col_idx, (label, _) in {k: ITEMS[k] for k in monotonic_cols}.items():
        series = df.iloc[:, col_idx]
        if col_idx in range(14, 22):
            top_set = {"Agree", "Strongly agree"}
        else:  # col 22
            top_set = {"Somewhat easy", "Very easy"}
        is_top = series.isin(top_set)
        for seg, mask in df.groupby(seg_col_name).groups.items():
            sub = is_top.loc[mask]
            rows.append({"item": label, "segment": seg, "n": len(sub),
                         "T2B_pct": round(float(sub.mean()) * 100, 1)})
    return pd.DataFrame(rows)

mono_cols = list(range(14, 23))
usage_t2b = t2b_by_segment("Usage Bin", mono_cols)
usage_pivot = usage_t2b.pivot(index="item", columns="segment", values="T2B_pct").reindex(
    [ITEMS[i][0] for i in range(14, 23)])[["High", "Medium", "Low"]]
print("\nT2B % by Usage Bin:")
print(usage_pivot)
usage_pivot.to_csv(OUT / "descriptives_t2b_by_usage.csv")

vrcplus_t2b = t2b_by_segment(df.columns[VRCPLUS_COL], mono_cols)
vrcplus_pivot = vrcplus_t2b.pivot(index="item", columns="segment", values="T2B_pct").reindex(
    [ITEMS[i][0] for i in range(14, 23)])
print("\nT2B % by VRC+:")
print(vrcplus_pivot)
vrcplus_pivot.to_csv(OUT / "descriptives_t2b_by_vrcplus.csv")

# Segment heatmap (T2B by item × segment)
seg_combined = pd.concat([
    usage_pivot.add_prefix("Usage: "),
    vrcplus_pivot.rename(columns=lambda c: "VRC+: " + ("Yes" if "Yes" in c else "No")),
], axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(seg_combined, annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=20, vmax=90, cbar_kws={"label": "Top-2-Box %"}, ax=ax)
ax.set_title("Top-2-Box % by item × segment")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(CHARTS / "descr_segment_heatmap.png", dpi=120)
plt.close(fig)

# T2B by Tenure (months since activation)
# Tenure source: Custom Data 2 (col 9), numeric months 0–97. Buckets per spec.

# Bucket tenure and compute T2B per item
TENURE_COL = 9  # Custom Data 2 — months since activation
TENURE_BUCKETS = ["0", "1-3", "4-12", "13-24", "25+"]

def to_tenure_bucket(months):
    if pd.isna(months):
        return None
    m = int(months)
    if m == 0: return "0"
    if m <= 3: return "1-3"
    if m <= 12: return "4-12"
    if m <= 24: return "13-24"
    return "25+"

df["Tenure Bucket"] = df.iloc[:, TENURE_COL].map(to_tenure_bucket)
tenure_n = df["Tenure Bucket"].value_counts().reindex(TENURE_BUCKETS).fillna(0).astype(int)
print("\nTenure bucket sizes:")
print(tenure_n)

tenure_t2b = t2b_by_segment("Tenure Bucket", mono_cols)
tenure_pivot = tenure_t2b.pivot(
    index="item", columns="segment", values="T2B_pct",
).reindex([ITEMS[i][0] for i in range(14, 23)])[TENURE_BUCKETS]
print("\nT2B % by Tenure Bucket:")
print(tenure_pivot)
tenure_pivot.to_csv(OUT / "descriptives_t2b_by_tenure.csv")

# Tenure heatmap — same style as the existing segment heatmap
col_labels_with_n = [f"{b}\n(n={tenure_n[b]})" for b in TENURE_BUCKETS]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.heatmap(tenure_pivot, annot=True, fmt=".0f", cmap="RdYlGn",
            vmin=20, vmax=90, cbar_kws={"label": "Top-2-Box %"},
            xticklabels=col_labels_with_n, ax=ax)
ax.set_title("Top-2-Box % by item × Tenure (months since activation)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig(CHARTS / "descr_t2b_by_tenure.png", dpi=120)
plt.close(fig)

# Enforcement Balance × Usage — focused segmentation analysis
# Goal: surface whether the Goldilocks distribution shifts with engagement.

# Enforcement Balance by Usage — granular table (all 8 usage levels)
USAGE_ORDER_FULL = [
    "Every day", "A few times a week", "Once a week",
    "A few times a month", "Once a month", "Less than once a month",
    "I’ve only used it once or twice", "I have never used it",
]
ef_by_usage_full = pd.crosstab(
    df.iloc[:, USAGE_COL], df.iloc[:, 23], normalize="index",
).reindex(USAGE_ORDER_FULL).dropna(how="all")
ef_by_usage_full = (ef_by_usage_full[ENFORCEMENT_ORDER] * 100).round(1)
ef_by_usage_full.columns = ENFORCEMENT_LABELS
ef_by_usage_full["n"] = df.iloc[:, USAGE_COL].value_counts().reindex(
    ef_by_usage_full.index).fillna(0).astype(int)
ef_by_usage_full["want_more"] = (ef_by_usage_full["Far too little"]
                                  + ef_by_usage_full["Slightly too little"]).round(1)
ef_by_usage_full["want_less"] = (ef_by_usage_full["Slightly too much"]
                                  + ef_by_usage_full["Far too much"]).round(1)
ef_by_usage_full["net_more_minus_less"] = (ef_by_usage_full["want_more"]
                                            - ef_by_usage_full["want_less"]).round(1)
print("\nEnforcement Balance by Usage Frequency (granular):")
print(ef_by_usage_full)
ef_by_usage_full.to_csv(OUT / "descriptives_enforcement_by_usage_full.csv")

# Chi-square test of independence
from scipy.stats import chi2_contingency
contingency = pd.crosstab(df["Usage Bin"], df.iloc[:, 23])[ENFORCEMENT_ORDER]
chi2, p_val, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test (Usage Bin × Enforcement Balance):")
print(f"  chi2 = {chi2:.2f}, df = {dof}, p = {p_val:.4f}")
n_total = contingency.values.sum()
cramer_v = float(np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1))))
print(f"  Cramér's V (effect size) = {cramer_v:.3f}")

# Diverging stacked bar chart — Enforcement Balance by Usage Bin
USAGE_BIN_ORDER = ["Low", "Medium", "High"]  # bottom to top in chart
ef_by_bin = pd.crosstab(df["Usage Bin"], df.iloc[:, 23], normalize="index")
ef_by_bin = (ef_by_bin[ENFORCEMENT_ORDER] * 100).reindex(USAGE_BIN_ORDER)
ef_by_bin.columns = ENFORCEMENT_LABELS
bin_n = df["Usage Bin"].value_counts().reindex(USAGE_BIN_ORDER)

far_too_much = ef_by_bin["Far too much"]
slightly_too_much = ef_by_bin["Slightly too much"]
about_right = ef_by_bin["About right"]
slightly_too_little = ef_by_bin["Slightly too little"]
far_too_little = ef_by_bin["Far too little"]

half_about = about_right / 2
fig, ax = plt.subplots(figsize=(10, 4.5))
y = np.arange(len(ef_by_bin))
ax.barh(y, -far_too_much, color="#67000d",
        left=-slightly_too_much - half_about, label="Far too much")
ax.barh(y, -slightly_too_much, color="#fb6a4a",
        left=-half_about, label="Slightly too much")
ax.barh(y, about_right, color="#cccccc", left=-half_about, label="About right")
ax.barh(y, slightly_too_little, color="#9ecae1",
        left=half_about, label="Slightly too little")
ax.barh(y, far_too_little, color="#08306b",
        left=half_about + slightly_too_little, label="Far too little")
ax.axvline(0, color="black", linewidth=0.8)

# Annotate net direction next to bars
for i, bin_name in enumerate(USAGE_BIN_ORDER):
    want_more = float(slightly_too_little.iloc[i] + far_too_little.iloc[i])
    want_less = float(slightly_too_much.iloc[i] + far_too_much.iloc[i])
    ax.text(half_about.iloc[i] + slightly_too_little.iloc[i] + far_too_little.iloc[i] + 2,
            i, f"want more: {want_more:.0f}%", va="center", fontsize=8,
            color="#08306b", fontweight="bold")
    ax.text(-(slightly_too_much.iloc[i] + far_too_much.iloc[i] + half_about.iloc[i] + 2),
            i, f"want less: {want_less:.0f}%", va="center", ha="right", fontsize=8,
            color="#67000d", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels([f"{b}\n(n={bin_n[b]})" for b in USAGE_BIN_ORDER])
ax.set_xlabel("← Want LESS enforcement   |   Want MORE enforcement →")
ax.set_title("Enforcement Balance by Usage Frequency")
ax.set_xlim(-50, 75)
ax.set_xticks(np.arange(-50, 76, 25))
ax.set_xticklabels([f"{abs(x)}%" for x in np.arange(-50, 76, 25)])
ax.legend(loc="lower right", ncol=5, fontsize=8, bbox_to_anchor=(1.0, -0.22))
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "descr_enforcement_by_usage_diverging.png", dpi=120)
plt.close(fig)

# Net "want more - want less" by full usage frequency (the gradient view)
ef_focus = ef_by_usage_full[ef_by_usage_full["n"] >= 30].copy()
fig, ax = plt.subplots(figsize=(9, 4.5))
xs = np.arange(len(ef_focus))
ax.bar(xs, ef_focus["want_more"], color="#08306b",
       label="Want MORE enforcement", alpha=0.85)
ax.bar(xs, -ef_focus["want_less"], color="#67000d",
       label="Want LESS enforcement", alpha=0.85)
for i, (idx, row) in enumerate(ef_focus.iterrows()):
    ax.text(i, row["want_more"] + 1.5, f"{row['want_more']:.0f}%",
            ha="center", fontsize=9, color="#08306b", fontweight="bold")
    ax.text(i, -row["want_less"] - 2.5, f"{row['want_less']:.0f}%",
            ha="center", fontsize=9, color="#67000d", fontweight="bold")
    ax.text(i, 0, f"{row['About right']:.0f}% about right",
            ha="center", va="center", fontsize=8, color="#444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([f"{idx}\n(n={ef_focus.loc[idx, 'n']})"
                    for idx in ef_focus.index], fontsize=9)
ax.set_ylabel("% wanting change ←")
ax.set_title("Direction of dissatisfaction by Usage Frequency\n"
             "(Bins with n<30 omitted for stability)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "descr_enforcement_direction_by_usage.png", dpi=120)
plt.close(fig)

# Print summary
print("\n" + "=" * 60)
print("Files written")
print("=" * 60)
print(f"  Tables: {OUT}")
print(f"  Charts: {CHARTS}")
for f in sorted(OUT.glob("descriptives_*.csv")):
    print(f"    - {f.name}")
for f in sorted(CHARTS.glob("descr_*.png")):
    print(f"    - charts/{f.name}")