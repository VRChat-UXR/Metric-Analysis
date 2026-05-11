# %% [markdown]
# Cross-wave multivariate comparison
# Reads each wave's `results.json` (produced by Analysis_Exploration.py) and tracks:
#   - CFA fit indices (RMSEA, CFI, SRMR, χ²) over time
#   - Cronbach's α (Experience block) over time
#   - Cluster proportions (re-ranked by overall positivity so labels are stable)
#   - Cluster mean profiles per wave (does the shape drift?)
# Produces:
#   - csv/comparisons/fit_indices_by_wave.csv
#   - csv/comparisons/cluster_proportions_by_wave.csv
#   - csv/comparisons/alpha_by_wave.csv
#   - png/comparisons/fit_indices_trend.png
#   - png/comparisons/alpha_trend.png
#   - png/comparisons/cluster_proportions_trend.png
#   - png/comparisons/cluster_profile_drift.png
#
# Single-wave behavior: writes the snapshot tables and exits without producing
# trend charts (logs that ≥2 waves are required).
#
# Run prerequisite: each wave's Analysis_Exploration.py must have been run
# already (so results.json exists in cfa_efa_analysis/csv/<wave>/).

# %% Imports
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
ROOT = ANALYSIS_DIR.parent
CSV_DIR = ANALYSIS_DIR / "csv"
OUT = CSV_DIR / "comparisons"
CHARTS = ANALYSIS_DIR / "png" / "comparisons"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

WAVE_RE = re.compile(r"^\d{4}-\d{2}$")

# %% Discover waves with results.json
waves = []
for sub in sorted(CSV_DIR.iterdir()):
    if sub.is_dir() and WAVE_RE.match(sub.name) and (sub / "results.json").exists():
        waves.append(sub.name)
print(f"Waves with results.json: {waves}")
if not waves:
    raise SystemExit(
        "No wave subdirectories with results.json found.\n"
        "Run Analysis_Exploration.py for each wave first."
    )

# %% Extract per-wave metrics
ITEMS_ORDER = [
    "Ease of Use", "Discovery", "Emotional Experience", "Personalization",
    "Social Connection", "Visual Experience", "Experience Value",
    "Feature Value", "Participation Safety", "Enforcement Satisfaction",
]
CLUSTER_LABELS = ["Detractors", "Neutrals", "Enthusiasts"]


def rank_clusters_by_positivity(profiles):
    """Re-rank cluster IDs by mean overall positivity so labels are stable across waves.
    Returns dict mapping original cluster ID -> stable label.
    """
    means = {}
    for row in profiles:
        cluster_id = row["Cluster"]
        item_vals = [row[i] for i in ITEMS_ORDER if i in row]
        means[cluster_id] = float(np.mean(item_vals))
    # Sort ascending: lowest = Detractors, middle = Neutrals, highest = Enthusiasts
    sorted_ids = sorted(means.keys(), key=lambda c: means[c])
    if len(sorted_ids) == 3:
        return dict(zip(sorted_ids, CLUSTER_LABELS))
    # Generic fallback for k != 3
    return {c: f"Tier {i + 1}" for i, c in enumerate(sorted_ids)}


fit_rows, alpha_rows, cluster_rows, profile_rows = [], [], [], []
for w in waves:
    r = json.loads((CSV_DIR / w / "results.json").read_text())
    f = r["cfa_fit"]
    fit_rows.append({
        "wave": w,
        "n": r["n_respondents"],
        "chi2": round(f["chi2"], 2),
        "chi2_dof": int(f["chi2_dof"]),
        "RMSEA": round(f["RMSEA"], 4),
        "CFI": round(f["CFI"], 4),
        "SRMR": round(f["SRMR"], 4),
        "AIC": round(f["AIC"], 2),
        "BIC": round(f["BIC"], 2),
        "verdict": r["cfa_verdict"][:30] + "…",
    })
    alpha_rows.append({
        "wave": w,
        "experience_alpha": round(r["experience_cronbach_alpha"], 3),
        "ci_low": r["experience_cronbach_ci"][0],
        "ci_high": r["experience_cronbach_ci"][1],
    })
    profiles = r["cluster_profiles"]
    label_map = rank_clusters_by_positivity(profiles)
    n_total = sum(p["n"] for p in profiles)
    for p in profiles:
        label = label_map[p["Cluster"]]
        cluster_rows.append({
            "wave": w, "cluster_label": label,
            "n": int(p["n"]), "pct": round(100 * p["n"] / n_total, 1),
        })
        profile_rows.append({
            "wave": w, "cluster_label": label,
            **{item: p[item] for item in ITEMS_ORDER if item in p},
        })

fit_df = pd.DataFrame(fit_rows)
alpha_df = pd.DataFrame(alpha_rows)
cluster_df = pd.DataFrame(cluster_rows)
profile_df = pd.DataFrame(profile_rows)
fit_df.to_csv(OUT / "fit_indices_by_wave.csv", index=False)
alpha_df.to_csv(OUT / "alpha_by_wave.csv", index=False)
cluster_df.to_csv(OUT / "cluster_proportions_by_wave.csv", index=False)
profile_df.to_csv(OUT / "cluster_profiles_by_wave.csv", index=False)

print("\nFit indices by wave:")
print(fit_df)
print("\nCronbach's alpha (Experience block) by wave:")
print(alpha_df)
print("\nCluster proportions by wave:")
print(cluster_df.pivot(index="wave", columns="cluster_label", values="pct"))

if len(waves) < 2:
    print("\nOnly one wave available — skipping trend charts. "
          "Re-run after the next wave's Analysis_Exploration.py finishes.")
    raise SystemExit(0)

# %% Fit indices trend
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
THRESHOLDS = {"RMSEA": (0.08, "lower is better"),
              "CFI": (0.90, "higher is better"),
              "SRMR": (0.08, "lower is better")}
for ax, idx in zip(axes, ["RMSEA", "CFI", "SRMR"]):
    ax.plot(fit_df["wave"], fit_df[idx], "o-", color="#1f4ea1")
    for x, y in zip(fit_df["wave"], fit_df[idx]):
        ax.text(x, y, f"  {y:.3f}", va="center", fontsize=8)
    threshold, note = THRESHOLDS[idx]
    ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8,
               label=f"Threshold ({threshold})")
    ax.set_title(f"{idx} ({note})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("CFA fit indices over time", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "fit_indices_trend.png", dpi=120)
plt.close(fig)

# %% Cronbach's α trend (Experience block)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(alpha_df["wave"], alpha_df["experience_alpha"], "o-", color="#2ca02c", linewidth=2)
ax.fill_between(alpha_df["wave"], alpha_df["ci_low"], alpha_df["ci_high"],
                color="#2ca02c", alpha=0.2, label="95% CI")
for x, y in zip(alpha_df["wave"], alpha_df["experience_alpha"]):
    ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
ax.axhline(0.7, color="grey", linestyle="--", label="Acceptable (0.70)")
ax.axhline(0.8, color="grey", linestyle=":", label="Good (0.80)")
ax.set_ylabel("Cronbach's α (Experience block)")
ax.set_title("Experience composite reliability over time")
ax.set_ylim(0.5, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(CHARTS / "alpha_trend.png", dpi=120)
plt.close(fig)

# %% Cluster proportions over time (stacked area)
pivot = cluster_df.pivot(index="wave", columns="cluster_label", values="pct")
pivot = pivot.reindex(columns=[c for c in CLUSTER_LABELS if c in pivot.columns])
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {"Detractors": "#d62728", "Neutrals": "#999999", "Enthusiasts": "#2ca02c"}
ax.stackplot(pivot.index, [pivot[c].values for c in pivot.columns],
             labels=pivot.columns, colors=[colors.get(c, "#999") for c in pivot.columns])
for i, w in enumerate(pivot.index):
    cum = 0
    for c in pivot.columns:
        v = pivot.loc[w, c]
        ax.text(i, cum + v / 2, f"{c}\n{v:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        cum += v
ax.set_ylabel("% of respondents")
ax.set_title("Cluster (satisfaction tier) proportions over time")
ax.set_ylim(0, 100)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(CHARTS / "cluster_proportions_trend.png", dpi=120)
plt.close(fig)

# %% Cluster profile drift — overlay each cluster's shape per wave
fig, axes = plt.subplots(1, len([c for c in CLUSTER_LABELS if c in pivot.columns]),
                          figsize=(14, 4), sharey=True)
if not isinstance(axes, np.ndarray):
    axes = [axes]
present_clusters = [c for c in CLUSTER_LABELS if c in pivot.columns]
for ax, cluster in zip(axes, present_clusters):
    sub = profile_df[profile_df["cluster_label"] == cluster]
    for _, row in sub.iterrows():
        item_vals = [row[item] for item in ITEMS_ORDER if item in row]
        ax.plot(ITEMS_ORDER, item_vals, "o-", label=row["wave"], alpha=0.75)
    ax.set_title(cluster, fontsize=11)
    ax.set_ylim(1, 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks(range(len(ITEMS_ORDER)))
    ax.set_xticklabels(ITEMS_ORDER, rotation=45, ha="right", fontsize=8)
fig.suptitle("Cluster profile drift across waves (shape comparison)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "cluster_profile_drift.png", dpi=120)
plt.close(fig)

# %% Summary
print("\n" + "=" * 60)
print("Files written")
print("=" * 60)
print(f"  Tables: {OUT}")
for f in sorted(OUT.glob("*.csv")):
    print(f"    - {f.name}")
print(f"  Charts: {CHARTS}")
for f in sorted(CHARTS.glob("*.png")):
    print(f"    - {f.name}")
