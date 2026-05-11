# %% [markdown]
# Cross-wave descriptive comparison
# Discovers all UXR_Survey_Results_YYYY-MM.xlsx files, computes per-wave
# T2B/B2B per item and the Enforcement Balance distribution, then produces:
#   - csv/comparisons/t2b_by_wave.csv  (long format: item, wave, T2B, B2B, n)
#   - csv/comparisons/t2b_deltas.csv   (consecutive-wave deltas + two-prop z-test)
#   - csv/comparisons/enforcement_direction_by_wave.csv
#   - png/comparisons/t2b_trend_per_item.png
#   - png/comparisons/t2b_change_magnitude.png  (latest delta sorted)
#   - png/comparisons/t2b_slope_chart.png  (clean for 2-wave)
#   - png/comparisons/enforcement_direction_trend.png
#
# Single-wave behavior: writes the per-wave snapshot tables and exits before
# producing comparison charts (logs that ≥2 waves are required).

# %% Imports
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
ROOT = ANALYSIS_DIR.parent
OUT = ANALYSIS_DIR / "csv" / "comparisons"
CHARTS = ANALYSIS_DIR / "png" / "comparisons"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

WAVE_RE = re.compile(r"UXR_Survey_Results_(\d{4}-\d{2})\.xlsx$")
waves = sorted(m.group(1) for f in ROOT.glob("UXR_Survey_Results_*.xlsx")
               if (m := WAVE_RE.search(f.name)))
print(f"Waves discovered: {waves}")
if not waves:
    raise SystemExit("No UXR_Survey_Results_*.xlsx files found.")

# %% Item map (kept in sync with per-wave scripts)
ITEMS = {
    14: "Ease of Use", 15: "Discovery", 16: "Emotional Experience",
    17: "Personalization", 18: "Social Connection", 19: "Visual Experience",
    20: "Experience Value", 21: "Feature Value", 22: "Participation Safety",
}
ITEM_ORDER = list(ITEMS.values())
LIKERT_TOP = {"Agree", "Strongly agree"}
LIKERT_BOT = {"Strongly disagree", "Disagree"}
SAFETY_TOP = {"Somewhat easy", "Very easy"}
SAFETY_BOT = {"Very difficult", "Somewhat difficult"}
ENFORCEMENT_ORDER = [
    "Far too little enforcement (harmful behavior often goes unchecked)",
    "Slightly too little enforcement",
    "About the right level",
    "Slightly too much enforcement",
    "Far too much enforcement (content or users are restricted unnecessarily)",
]
ENFORCEMENT_LABELS = ["Far too little", "Slightly too little", "About right",
                      "Slightly too much", "Far too much"]


def per_wave_metrics(wave):
    """Load one wave's xlsx and return long-format T2B/B2B + enforcement direction."""
    df = pd.read_excel(ROOT / f"UXR_Survey_Results_{wave}.xlsx").iloc[1:].reset_index(drop=True)
    rows = []
    for col_idx, label in ITEMS.items():
        s = df.iloc[:, col_idx]
        top = SAFETY_TOP if col_idx == 22 else LIKERT_TOP
        bot = SAFETY_BOT if col_idx == 22 else LIKERT_BOT
        n = int(s.notna().sum())
        rows.append({
            "wave": wave, "item": label,
            "n": n,
            "T2B_count": int(s.isin(top).sum()),
            "B2B_count": int(s.isin(bot).sum()),
            "T2B_pct": round(s.isin(top).mean() * 100, 2),
            "B2B_pct": round(s.isin(bot).mean() * 100, 2),
        })
    ef_counts = df.iloc[:, 23].value_counts().reindex(ENFORCEMENT_ORDER).fillna(0).astype(int)
    ef_pct = (ef_counts / ef_counts.sum() * 100).round(1)
    ef_row = {"wave": wave, "n": int(ef_counts.sum())}
    for label, p in zip(ENFORCEMENT_LABELS, ef_pct.values):
        ef_row[label] = float(p)
    ef_row["want_more"] = round(float(ef_pct.iloc[0] + ef_pct.iloc[1]), 1)
    ef_row["about_right"] = round(float(ef_pct.iloc[2]), 1)
    ef_row["want_less"] = round(float(ef_pct.iloc[3] + ef_pct.iloc[4]), 1)
    return pd.DataFrame(rows), ef_row


# %% Aggregate across waves
all_rows, ef_rows = [], []
for w in waves:
    items_df, ef_row = per_wave_metrics(w)
    all_rows.append(items_df)
    ef_rows.append(ef_row)
trend = pd.concat(all_rows, ignore_index=True)
ef_trend = pd.DataFrame(ef_rows)
trend.to_csv(OUT / "t2b_by_wave.csv", index=False)
ef_trend.to_csv(OUT / "enforcement_direction_by_wave.csv", index=False)
print("\nT2B/B2B by wave (long format):")
print(trend)
print("\nEnforcement direction by wave:")
print(ef_trend)

if len(waves) < 2:
    print("\nOnly one wave available — skipping wave-over-wave comparison charts.")
    print("When the next wave's xlsx is added to the project root, re-run this script.")
    raise SystemExit(0)

# %% Wave-over-wave deltas + significance (two-proportion z-test on T2B)
def two_prop_z(p1, n1, p2, n2):
    """Two-proportion z-test on T2B counts. Returns (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_pool = (p1 + p2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 / n1 - p2 / n2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p_value)


deltas = []
for prev, curr in zip(waves, waves[1:]):
    a = trend[trend["wave"] == prev].set_index("item")
    b = trend[trend["wave"] == curr].set_index("item")
    for item in ITEM_ORDER:
        ra, rb = a.loc[item], b.loc[item]
        z, p = two_prop_z(rb["T2B_count"], rb["n"], ra["T2B_count"], ra["n"])
        deltas.append({
            "from_wave": prev, "to_wave": curr, "item": item,
            "T2B_prev": ra["T2B_pct"], "T2B_curr": rb["T2B_pct"],
            "delta_pp": round(rb["T2B_pct"] - ra["T2B_pct"], 2),
            "B2B_prev": ra["B2B_pct"], "B2B_curr": rb["B2B_pct"],
            "B2B_delta_pp": round(rb["B2B_pct"] - ra["B2B_pct"], 2),
            "z": round(z, 2), "p_value": round(p, 4),
            "sig_05": "*" if p < 0.05 else "",
        })
deltas_df = pd.DataFrame(deltas)
deltas_df.to_csv(OUT / "t2b_deltas.csv", index=False)
print("\nWave-over-wave T2B deltas with significance test:")
print(deltas_df)

# %% Trend chart: T2B per item across waves (multipanel)
n_items = len(ITEM_ORDER)
ncols = 3
nrows = int(np.ceil(n_items / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), sharey=True)
axes = axes.flatten()
for i, item in enumerate(ITEM_ORDER):
    ax = axes[i]
    sub = trend[trend["item"] == item].sort_values("wave")
    ax.plot(sub["wave"], sub["T2B_pct"], "o-", color="#2ca02c", label="T2B")
    ax.plot(sub["wave"], sub["B2B_pct"], "s--", color="#d62728", label="B2B", alpha=0.7)
    for x, y in zip(sub["wave"], sub["T2B_pct"]):
        ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=8, color="#2ca02c")
    ax.set_title(item, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)
for j in range(n_items, len(axes)):
    axes[j].axis("off")
fig.suptitle("Top-2-Box / Bottom-2-Box trend per item", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(CHARTS / "t2b_trend_per_item.png", dpi=120)
plt.close(fig)

# %% Slope chart (2-wave) / line chart (3+ waves)
fig, ax = plt.subplots(figsize=(8, 6))
for item in ITEM_ORDER:
    sub = trend[trend["item"] == item].sort_values("wave")
    ax.plot(sub["wave"], sub["T2B_pct"], "o-", label=item, alpha=0.85)
    ax.text(sub["wave"].iloc[-1], sub["T2B_pct"].iloc[-1], f"  {item}",
            va="center", fontsize=8)
ax.set_ylabel("Top-2-Box %")
ax.set_title("Item-level T2B across waves")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "t2b_slope_chart.png", dpi=120)
plt.close(fig)

# %% Latest-wave change magnitude (sorted bar)
latest_delta = deltas_df[deltas_df["to_wave"] == waves[-1]].copy()
latest_delta = latest_delta.sort_values("delta_pp")
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#d62728" if d < 0 else "#2ca02c" for d in latest_delta["delta_pp"]]
y = np.arange(len(latest_delta))
bars = ax.barh(y, latest_delta["delta_pp"], color=colors)
for b, row in zip(bars, latest_delta.itertuples()):
    sig = " *" if row.sig_05 else ""
    ax.text(b.get_width() + (0.4 if b.get_width() >= 0 else -0.4), b.get_y() + b.get_height() / 2,
            f"{row.delta_pp:+.1f}{sig}", va="center",
            ha="left" if b.get_width() >= 0 else "right", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(latest_delta["item"])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(f"Δ T2B (pp), {latest_delta['from_wave'].iloc[0]} → {latest_delta['to_wave'].iloc[0]}")
ax.set_title("Wave-over-wave T2B change by item   (* = p < .05)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "t2b_change_magnitude.png", dpi=120)
plt.close(fig)

# %% Enforcement Direction trend
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ef_trend["wave"], ef_trend["want_more"], "o-",
        color="#08306b", label="Want MORE enforcement", linewidth=2)
ax.plot(ef_trend["wave"], ef_trend["about_right"], "o-",
        color="#5a8fce", label="About right")
ax.plot(ef_trend["wave"], ef_trend["want_less"], "o-",
        color="#67000d", label="Want LESS enforcement")
for col, color in [("want_more", "#08306b"), ("about_right", "#5a8fce"),
                   ("want_less", "#67000d")]:
    for x, y in zip(ef_trend["wave"], ef_trend[col]):
        ax.text(x, y + 1.5, f"{y:.0f}%", ha="center", fontsize=9, color=color)
ax.set_ylabel("% of respondents")
ax.set_title("Enforcement Direction over time")
ax.set_ylim(0, max(ef_trend[["want_more", "about_right", "want_less"]].max()) * 1.2)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(CHARTS / "enforcement_direction_trend.png", dpi=120)
plt.close(fig)

# %% Print summary
print("\n" + "=" * 60)
print("Files written")
print("=" * 60)
print(f"  Tables: {OUT}")
for f in sorted(OUT.glob("*.csv")):
    print(f"    - {f.name}")
print(f"  Charts: {CHARTS}")
for f in sorted(CHARTS.glob("*.png")):
    print(f"    - {f.name}")
