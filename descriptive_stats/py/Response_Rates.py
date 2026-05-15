# Response Rates by Tenure
# Joins the survey send list (uxr_monthly_survey_<YYYY-MM-DD>.csv) to the
# response file (UXR_Survey_Results_<YYYY-MM>.xlsx) on user_id and reports
# response rates per tenure bucket. Two views:
#   - Headline: usable responses (all items answered) divided by users sent.
#   - Funnel: started / screened / usable counts and rates per bucket, so
#     drop-off shape across tenure is visible.

# Imports & wave selection
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent           # descriptive_stats/py
ANALYSIS_DIR = HERE.parent                       # descriptive_stats
ROOT = ANALYSIS_DIR.parent                       # project root

WAVE_RE = re.compile(r"UXR_Survey_Results_(\d{4}-\d{2})\.xlsx$")
SENT_RE = re.compile(r"uxr_monthly_survey_(\d{4}-\d{2}-\d{2})\.csv$")

def discover_waves(root):
    return sorted(m.group(1) for f in root.glob("UXR_Survey_Results_*.xlsx")
                  if (m := WAVE_RE.search(f.name)))

def find_sent_file(root, wave):
    # Sent-list filenames include the send date (YYYY-MM-DD). Pick the
    # latest file whose date starts with the wave's YYYY-MM prefix.
    candidates = sorted(
        (m.group(1), f) for f in root.glob("uxr_monthly_survey_*.csv")
        if (m := SENT_RE.search(f.name)) and m.group(1).startswith(wave)
    )
    if not candidates:
        raise SystemExit(
            f"No sent-list file uxr_monthly_survey_{wave}-*.csv found in {root}."
        )
    return candidates[-1][1]

parser = argparse.ArgumentParser(description="Response rates by tenure for a given wave")
parser.add_argument("--wave", default=None,
                    help="YYYY-MM wave. Defaults to latest available.")
args, _ = parser.parse_known_args()

available = discover_waves(ROOT)
if not available:
    raise SystemExit(
        f"No UXR_Survey_Results_YYYY-MM.xlsx files found in {ROOT}."
    )
WAVE = args.wave or available[-1]
if WAVE not in available:
    raise SystemExit(f"Wave {WAVE} not found. Available: {available}")

RESULTS_FILE = ROOT / f"UXR_Survey_Results_{WAVE}.xlsx"
SENT_FILE = find_sent_file(ROOT, WAVE)
OUT = ANALYSIS_DIR / "csv" / WAVE
CHARTS = ANALYSIS_DIR / "png" / WAVE
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
print(f"Wave: {WAVE}")
print(f"Sent: {SENT_FILE.name}")
print(f"Results: {RESULTS_FILE.name}")
print(f"Out: {OUT}\n")

# Tenure buckets (months since activation)
TENURE_BUCKETS = ["0", "1-3", "4-12", "13-24", "25+"]

def bucket_tenure(months):
    if months == 0:
        return "0"
    if 1 <= months <= 3:
        return "1-3"
    if 4 <= months <= 12:
        return "4-12"
    if 13 <= months <= 24:
        return "13-24"
    return "25+"

# Load send list — the denominator universe
sent = pd.read_csv(SENT_FILE)
sent["bucket"] = sent["tenure"].apply(bucket_tenure)
print(f"Sent: {len(sent):,} users")

# Load results, drop the SurveyMonkey "Response" header row
df = pd.read_excel(RESULTS_FILE).iloc[1:].reset_index(drop=True)

USER_ID_COL = 8
SCREENING_COL = 13         # early-survey screening column
LIKERT_COLS = list(range(14, 24))

# Build the three respondent ID sets that define the funnel stages.
# Usable is "all 10 Likert items answered" — sufficient for response-rate
# purposes (SurveyMonkey forces select-one on each Likert column).
ids_started = set(df.iloc[:, USER_ID_COL].dropna())
screened_mask = df.iloc[:, SCREENING_COL].astype(str).str.startswith("Yes")
ids_screened = set(df.loc[screened_mask].iloc[:, USER_ID_COL].dropna())
usable_mask = df.iloc[:, LIKERT_COLS].notna().all(axis=1)
ids_usable = set(df.loc[usable_mask].iloc[:, USER_ID_COL].dropna())

print(f"Funnel totals — started: {len(ids_started)}  screened: {len(ids_screened)}  "
      f"usable: {len(ids_usable)}")

# Sanity check: every responding user_id should exist in the send list.
orphans = ids_started - set(sent["user_id"])
if orphans:
    print(f"WARNING: {len(orphans)} respondent IDs not found in send list — excluded from rates.")

# Aggregate by tenure bucket, using the send list as the canonical tenure source.
def per_bucket_counts(id_set):
    return (sent.assign(hit=sent["user_id"].isin(id_set))
                .groupby("bucket")["hit"].sum()
                .reindex(TENURE_BUCKETS, fill_value=0))

sent_n = sent.groupby("bucket").size().reindex(TENURE_BUCKETS, fill_value=0)
started_n = per_bucket_counts(ids_started)
screened_n = per_bucket_counts(ids_screened)
usable_n = per_bucket_counts(ids_usable)

def add_total(df_, count_cols):
    totals = {c: df_[c].sum() for c in count_cols}
    totals["tenure_bucket"] = "TOTAL"
    return pd.concat([df_, pd.DataFrame([totals])], ignore_index=True)

# Headline table: usable response rate by tenure bucket
headline = pd.DataFrame({
    "tenure_bucket": TENURE_BUCKETS,
    "sent": sent_n.values,
    "usable_responses": usable_n.values,
})
headline = add_total(headline, ["sent", "usable_responses"])
headline["response_rate_pct"] = (headline["usable_responses"] / headline["sent"] * 100).round(2)
headline.to_csv(OUT / "response_rates_by_tenure.csv", index=False)
print("\n=== Usable response rate by tenure ===")
print(headline.to_string(index=False))

# Funnel table: counts and rates at each drop-off stage
funnel = pd.DataFrame({
    "tenure_bucket": TENURE_BUCKETS,
    "sent": sent_n.values,
    "started": started_n.values,
    "screened": screened_n.values,
    "usable": usable_n.values,
})
funnel = add_total(funnel, ["sent", "started", "screened", "usable"])
for stage in ["started", "screened", "usable"]:
    funnel[f"{stage}_pct"] = (funnel[stage] / funnel["sent"] * 100).round(2)
funnel.to_csv(OUT / "response_rate_funnel_by_tenure.csv", index=False)
print("\n=== Funnel by tenure (counts and rate as % of sent) ===")
print(funnel.to_string(index=False))

# Chart 1: headline — usable response rate by tenure bucket
plot_df = headline.iloc[:-1]  # drop TOTAL row for per-bucket plot
overall_rate = headline.iloc[-1]["response_rate_pct"]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(plot_df["tenure_bucket"], plot_df["response_rate_pct"],
              color="#3b78b3", edgecolor="white")
for bar, sent_count, resp_count in zip(
    bars, plot_df["sent"], plot_df["usable_responses"]
):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08,
            f"{h:.2f}%\n({int(resp_count):,}/{int(sent_count):,})",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Tenure (months since activation)")
ax.set_ylabel("Usable response rate (%)")
ax.set_title(f"Wave {WAVE} — Response rate by tenure (usable responses / sent) — overall {overall_rate:.2f}%")
ax.set_ylim(0, max(plot_df["response_rate_pct"]) * 1.30)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS / "response_rates_by_tenure.png", dpi=150)
plt.close()

# Chart 2: funnel — grouped bars for started / screened / usable rates per bucket
plot_df = funnel.iloc[:-1]
stages = [
    ("started_pct", "Started", "#9ecae1"),
    ("screened_pct", "Screened", "#3182bd"),
    ("usable_pct", "Usable", "#08519c"),
]
x = np.arange(len(plot_df))
width = 0.27

fig, ax = plt.subplots(figsize=(11, 6))
for i, (col, label, color) in enumerate(stages):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, plot_df[col], width, label=label,
                  color=color, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.07,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(sent: {int(n):,})"
                    for b, n in zip(plot_df["tenure_bucket"], plot_df["sent"])])
ax.set_xlabel("Tenure (months since activation)")
ax.set_ylabel("Rate (% of users sent)")
ax.set_title(f"Wave {WAVE} — Response funnel by tenure")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(plot_df["started_pct"]) * 1.25)
plt.tight_layout()
plt.savefig(CHARTS / "response_rate_funnel_by_tenure.png", dpi=150)
plt.close()

# Chart 3: tenure distribution of usable responses (share of N=690)
total_usable = int(usable_n.sum())
share_pct = (usable_n / total_usable * 100)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(TENURE_BUCKETS, share_pct.values, color="#08519c", edgecolor="white")
for bar, count in zip(bars, usable_n.values):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
            f"{h:.1f}%\n(n={int(count):,})",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Tenure (months since activation)")
ax.set_ylabel("Share of usable responses (%)")
ax.set_title(f"Wave {WAVE} — Tenure distribution of usable responses (N={total_usable:,})")
ax.set_ylim(0, max(share_pct.values) * 1.20)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(CHARTS / "usable_responses_tenure_distribution.png", dpi=150)
plt.close()

# Matrix: usable response rate by tenure × platform
# Uses the send list as the canonical source for both tenure and platform
# (the send list is the denominator universe; col 11 of the response file
# only sees the numerator and would bias the platform mix).
PLATFORMS = ["Standalone VR", "PCVR", "Desktop", "Mobile"]

sent["usable"] = sent["user_id"].isin(ids_usable)
# unstack can leave NaN where a (bucket, platform) combo has no rows, and
# reindex's fill_value only fills *new* index/column entries — so .fillna(0)
# is the actual zero-fill we want before casting to int.
sent_cell = (sent.groupby(["bucket", "platform"]).size()
                 .unstack("platform")
                 .reindex(index=TENURE_BUCKETS, columns=PLATFORMS)
                 .fillna(0).astype(int))
usable_cell = (sent.groupby(["bucket", "platform"])["usable"].sum()
                   .unstack("platform")
                   .reindex(index=TENURE_BUCKETS, columns=PLATFORMS)
                   .fillna(0).astype(int))
rate_cell = (usable_cell / sent_cell.replace(0, np.nan) * 100).round(2)

# Add column and row totals (marginal rates) for the table view.
sent_cell["TOTAL"] = sent_cell.sum(axis=1)
usable_cell["TOTAL"] = usable_cell.sum(axis=1)
sent_cell.loc["TOTAL"] = sent_cell.sum(axis=0)
usable_cell.loc["TOTAL"] = usable_cell.sum(axis=0)
rate_full = (usable_cell / sent_cell.replace(0, np.nan) * 100).round(2)

rate_full.to_csv(OUT / "response_rates_by_tenure_platform.csv")
sent_cell.to_csv(OUT / "response_rates_by_tenure_platform_sent.csv")
usable_cell.to_csv(OUT / "response_rates_by_tenure_platform_usable.csv")

print("\n=== Usable response rate (%) by tenure × platform ===")
print(rate_full.to_string())

# Chart 4: heatmap of response rate, tenure × platform
fig, ax = plt.subplots(figsize=(9.5, 5.5))
data = rate_cell.values.astype(float)
im = ax.imshow(data, aspect="auto", cmap="Blues",
               vmin=np.nanmin(data), vmax=np.nanmax(data))

ax.set_xticks(np.arange(len(PLATFORMS)))
ax.set_xticklabels(PLATFORMS)
ax.set_yticks(np.arange(len(TENURE_BUCKETS)))
ax.set_yticklabels(TENURE_BUCKETS)
ax.set_xlabel("Platform")
ax.set_ylabel("Tenure (months since activation)")
ax.set_title(f"Wave {WAVE} — Usable response rate by tenure × platform")

# Cell annotations: rate%, with usable/sent counts beneath.
threshold = np.nanmin(data) + (np.nanmax(data) - np.nanmin(data)) * 0.55
for i, bucket in enumerate(TENURE_BUCKETS):
    for j, plat in enumerate(PLATFORMS):
        rate = rate_cell.loc[bucket, plat]
        n_sent = int(sent_cell.loc[bucket, plat])
        n_usable = int(usable_cell.loc[bucket, plat])
        if n_sent == 0 or np.isnan(rate):
            label = "—"
            color = "#666666"
        else:
            label = f"{rate:.1f}%\n({n_usable}/{n_sent})"
            color = "white" if rate >= threshold else "#1a1a1a"
        ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("Response rate (%)")
plt.tight_layout()
plt.savefig(CHARTS / "response_rates_by_tenure_platform.png", dpi=150)
plt.close()

print(f"\nWrote:")
print(f"  {OUT / 'response_rates_by_tenure.csv'}")
print(f"  {OUT / 'response_rate_funnel_by_tenure.csv'}")
print(f"  {OUT / 'response_rates_by_tenure_platform.csv'}")
print(f"  {OUT / 'response_rates_by_tenure_platform_sent.csv'}")
print(f"  {OUT / 'response_rates_by_tenure_platform_usable.csv'}")
print(f"  {CHARTS / 'response_rates_by_tenure.png'}")
print(f"  {CHARTS / 'response_rate_funnel_by_tenure.png'}")
print(f"  {CHARTS / 'usable_responses_tenure_distribution.png'}")
print(f"  {CHARTS / 'response_rates_by_tenure_platform.png'}")
