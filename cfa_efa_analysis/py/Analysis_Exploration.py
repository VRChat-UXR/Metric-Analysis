# %% [markdown]
# Monthly Metrics Survey — per-wave multivariate analysis
# Three pillars (Experience, CPV, Trust & Safety) over 10 Likert-scale items.
# See README.md for a methodology overview.

# %% Imports & config
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sklearn 1.6 renamed force_all_finite -> ensure_all_finite; factor_analyzer 0.5.1
# still uses the old name. Translate the kwarg before factor_analyzer imports it.
import sklearn.utils.validation as _skv
_orig_check_array = _skv.check_array
def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)
_skv.check_array = _check_array_compat

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import factor_analyzer.factor_analyzer as _fa_mod
_fa_mod.check_array = _check_array_compat

from kmedoids import KMedoids
from semopy import Model, calc_stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

import argparse
import re

HERE = Path(__file__).resolve().parent           # cfa_efa_analysis/py
ANALYSIS_DIR = HERE.parent                       # cfa_efa_analysis
ROOT = ANALYSIS_DIR.parent                       # project root

WAVE_RE = re.compile(r"UXR_Survey_Results_(\d{4}-\d{2})\.xlsx$")

def discover_waves(root):
    return sorted(m.group(1) for f in root.glob("UXR_Survey_Results_*.xlsx")
                  if (m := WAVE_RE.search(f.name)))

parser = argparse.ArgumentParser(description="Per-wave EFA/CFA/clustering analysis")
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
RESULTS = {}  # collects outputs to dump as JSON for the report writers

# %% Load data
df_raw = pd.read_excel(DATA)
print(f"Raw shape: {df_raw.shape}")

# Item column indices (per analysis plan)
EXPERIENCE_IDX = [14, 15, 16, 17, 18, 19]
CPV_IDX = [20, 21]
SAFETY_IDX = 22
ENFORCEMENT_IDX = 23
USAGE_IDX = 24
VRCPLUS_IDX = 25

ITEM_LABELS = {
    14: "Ease of Use",
    15: "Discovery",
    16: "Emotional Experience",
    17: "Personalization",
    18: "Social Connection",
    19: "Visual Experience",
    20: "Experience Value",
    21: "Feature Value",
    22: "Participation Safety",
    23: "Enforcement Satisfaction",
}
PILLAR_OF = {
    **{i: "Experience" for i in EXPERIENCE_IDX},
    **{i: "CPV" for i in CPV_IDX},
    22: "Trust & Safety",
    23: "Trust & Safety",
}

# %% Encode Likert + Goldilocks items
LIKERT_MAP = {
    "Strongly disagree": 1, "Disagree": 2, "Neither agree nor disagree": 3,
    "Agree": 4, "Strongly agree": 5,
}
SAFETY_MAP = {
    "Very difficult": 1, "Somewhat difficult": 2, "Neutral": 3,
    "Somewhat easy": 4, "Very easy": 5,
}
ENFORCEMENT_FOLDED = {
    "Far too little enforcement (harmful behavior often goes unchecked)": 1,
    "Slightly too little enforcement": 3,
    "About the right level": 5,
    "Slightly too much enforcement": 3,
    "Far too much enforcement (content or users are restricted unnecessarily)": 1,
}
ENFORCEMENT_DIRECTION = {
    "Far too little enforcement (harmful behavior often goes unchecked)": -2,
    "Slightly too little enforcement": -1,
    "About the right level": 0,
    "Slightly too much enforcement": 1,
    "Far too much enforcement (content or users are restricted unnecessarily)": 2,
}

data = pd.DataFrame(index=df_raw.index)
for idx in EXPERIENCE_IDX + CPV_IDX:
    data[ITEM_LABELS[idx]] = df_raw.iloc[:, idx].map(LIKERT_MAP)
data[ITEM_LABELS[22]] = df_raw.iloc[:, 22].map(SAFETY_MAP)
data[ITEM_LABELS[23]] = df_raw.iloc[:, 23].map(ENFORCEMENT_FOLDED)
data["Enforcement Direction"] = df_raw.iloc[:, 23].map(ENFORCEMENT_DIRECTION)
data["Usage Frequency"] = df_raw.iloc[:, 24]
data["VRC+"] = df_raw.iloc[:, 25]

# Drop the SurveyMonkey header row (row 0) which contains "Response" labels
data = data.dropna(subset=list(ITEM_LABELS.values())).reset_index(drop=True)
print(f"After dropping header / NA: {data.shape}")

# %% Data quality checks
print("\nMissing per item:")
print(data[list(ITEM_LABELS.values())].isna().sum())
print("\nValue ranges per item:")
print(data[list(ITEM_LABELS.values())].agg(["min", "max"]).T)
RESULTS["n_respondents"] = int(len(data))

# Collapse usage frequency into 3 bins (per plan §C)
USAGE_BINS = {
    "Every day": "High", "A few times a week": "High",
    "Once a week": "Medium", "A few times a month": "Medium",
    "Once a month": "Low",
}
data["Usage Bin"] = data["Usage Frequency"].map(lambda v: USAGE_BINS.get(v, "Low"))
print("\nUsage Bin counts:", data["Usage Bin"].value_counts().to_dict())
print("VRC+ counts:", data["VRC+"].value_counts().to_dict())

ITEMS = list(ITEM_LABELS.values())  # 10 items in canonical order
X = data[ITEMS].astype(float)

# %% Brief sanity descriptives (not the headline — Liz already has these)
desc = X.agg(["mean", "median", "std"]).T.round(2)
desc["T2B_pct"] = ((X >= 4).mean() * 100).round(1)
desc["B2B_pct"] = ((X <= 2).mean() * 100).round(1)
print("\nItem descriptives:")
print(desc)
desc.to_csv(OUT / "item_descriptives.csv")
RESULTS["descriptives"] = desc.reset_index().rename(columns={"index": "item"}).to_dict(orient="records")

# %% [markdown]
# Section 7 — Exploratory Factor Analysis
# Goal: see whether the proposed 3-pillar (or 9-sub-category) structure
# is empirically supported by the response patterns.

# %% EFA — KMO, Bartlett, parallel analysis, extraction
print("\n" + "=" * 60)
print("EFA")
print("=" * 60)
chi_sq, bartlett_p = calculate_bartlett_sphericity(X)
kmo_per_item, kmo_total = calculate_kmo(X)
print(f"Bartlett chi2 = {chi_sq:.1f}, p = {bartlett_p:.4g}")
print(f"KMO total = {kmo_total:.3f}")
RESULTS["efa_kmo"] = float(kmo_total)
RESULTS["efa_bartlett_p"] = float(bartlett_p)

# Parallel analysis: compare observed eigenvalues to those from random data
fa_unrot = FactorAnalyzer(n_factors=len(ITEMS), rotation=None)
fa_unrot.fit(X)
observed_eigs, _ = fa_unrot.get_eigenvalues()

n_iter = 100
random_eigs = np.zeros((n_iter, len(ITEMS)))
for i in range(n_iter):
    random_data = np.random.normal(size=X.shape)
    fa_rand = FactorAnalyzer(n_factors=len(ITEMS), rotation=None)
    fa_rand.fit(random_data)
    eigs, _ = fa_rand.get_eigenvalues()
    random_eigs[i] = eigs
parallel_threshold = np.percentile(random_eigs, 95, axis=0)
n_factors = int(np.sum(observed_eigs > parallel_threshold))
print(f"Observed eigenvalues: {np.round(observed_eigs, 2)}")
print(f"Parallel-analysis 95th-pctile: {np.round(parallel_threshold, 2)}")
print(f"=> Retain {n_factors} factors")
RESULTS["efa_n_factors"] = n_factors
RESULTS["efa_observed_eigs"] = [float(v) for v in observed_eigs]

# Scree + parallel plot
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(1, len(ITEMS) + 1)
ax.plot(xs, observed_eigs, "o-", label="Observed")
ax.plot(xs, parallel_threshold, "s--", color="grey", label="Parallel (95th pctile)")
ax.axhline(1, color="lightgrey", linestyle=":", label="Kaiser (1.0)")
ax.set_xlabel("Factor")
ax.set_ylabel("Eigenvalue")
ax.set_title("Scree plot with parallel analysis")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "efa_scree.png", dpi=120)
plt.close(fig)

# Extract with oblimin rotation (factors expected to correlate)
fa = FactorAnalyzer(n_factors=max(n_factors, 2), rotation="oblimin", method="ml")
fa.fit(X)
loadings = pd.DataFrame(
    fa.loadings_, index=ITEMS,
    columns=[f"F{i+1}" for i in range(fa.loadings_.shape[1])],
)
print("\nFactor loadings (oblimin):")
print(loadings.round(2))
loadings.to_csv(OUT / "efa_loadings.csv")
RESULTS["efa_loadings"] = loadings.round(3).reset_index().rename(
    columns={"index": "item"}).to_dict(orient="records")

communalities = pd.Series(fa.get_communalities(), index=ITEMS, name="communality")
print("\nCommunalities:")
print(communalities.round(2))
RESULTS["efa_communalities"] = communalities.round(3).to_dict()

# Loading heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(loadings.round(2), annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title(f"EFA loadings — {fa.loadings_.shape[1]} factors, oblimin")
fig.tight_layout()
fig.savefig(CHARTS / "efa_loadings_heatmap.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Section 8 — Confirmatory Factor Analysis on the 3-pillar model
# Specify Experience (6 items), CPV (2), T&S (2 — Participation Safety + folded Enforcement).

# %% CFA — 3-pillar model
print("\n" + "=" * 60)
print("CFA — 3-pillar model")
print("=" * 60)
# Standardize so CFA fits the correlation matrix — keeps the heterogeneous
# Enforcement Satisfaction variance from inflating absolute residuals (SRMR).
cfa_data = pd.DataFrame(StandardScaler().fit_transform(X), columns=[
    "EaseUse", "Discovery", "Emotional", "Personalization",
    "SocialConn", "Visual", "ExpValue", "FeatValue", "PartSafety", "EnforceSat",
])
model_spec = """
Experience =~ EaseUse + Discovery + Emotional + Personalization + SocialConn + Visual
CPV =~ ExpValue + FeatValue
TS =~ PartSafety + EnforceSat
"""
cfa = Model(model_spec)
cfa.fit(cfa_data, obj="MLW")
stats_df = calc_stats(cfa)
print("\nFit indices:")
print(stats_df.T)

# semopy 2.x doesn't include SRMR — compute it from observed vs implied cov
S = np.cov(cfa_data.values, rowvar=False, bias=False)
Sigma = cfa.calc_sigma()[0]
diag_S = np.sqrt(np.diag(S))
k = S.shape[0]
sq_resid = 0.0
n_elem = 0
for i in range(k):
    for j in range(i + 1):
        r = (S[i, j] - Sigma[i, j]) / (diag_S[i] * diag_S[j])
        sq_resid += r ** 2
        n_elem += 1
srmr_value = float(np.sqrt(sq_resid / n_elem))
print(f"SRMR (manual): {srmr_value:.4f}")

fit_row = stats_df.iloc[0].to_dict()
fit_clean = {
    "chi2": float(fit_row.get("chi2", np.nan)),
    "chi2_dof": float(fit_row.get("DoF", np.nan)),
    "chi2_p": float(fit_row.get("chi2 p-value", np.nan)),
    "RMSEA": float(fit_row.get("RMSEA", np.nan)),
    "CFI": float(fit_row.get("CFI", np.nan)),
    "SRMR": srmr_value,
    "AIC": float(fit_row.get("AIC", np.nan)),
    "BIC": float(fit_row.get("BIC", np.nan)),
}
print(f"\nKey indices: {fit_clean}")
RESULTS["cfa_fit"] = fit_clean

cfa_estimates = cfa.inspect()
print("\nStandardized estimates:")
print(cfa_estimates)
cfa_estimates.to_csv(OUT / "cfa_estimates.csv", index=False)

# Decision-tree rule on which model is supported (per plan §B)
def cfa_verdict(fit):
    rmsea, cfi, srmr = fit["RMSEA"], fit["CFI"], fit["SRMR"]
    rel_ok = rmsea < 0.08 and cfi > 0.90  # relative-fit indices
    abs_ok = srmr < 0.08                  # absolute-fit residuals
    if rel_ok and abs_ok:
        return "ACCEPT — 3-pillar model fits adequately"
    if rel_ok and not abs_ok:
        return ("MARGINAL — relative fit (RMSEA, CFI) is acceptable but absolute "
                "residuals (SRMR) are high; model captures the correlation structure "
                "but local misfit remains, likely driven by T&S scale heterogeneity")
    if rmsea < 0.10 and cfi > 0.85:
        return "MARGINAL — fit is borderline; report with caveats"
    return "REJECT — fit indices below conventional thresholds; revisit item set or model"

verdict = cfa_verdict(fit_clean)
print(f"\nVerdict: {verdict}")
RESULTS["cfa_verdict"] = verdict

# %% [markdown]
# Section 10 — K-means clustering (user personas)

# %% K-means — k selection
print("\n" + "=" * 60)
print("K-means clustering")
print("=" * 60)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

ks = list(range(2, 7))
inertias = []
silhouettes = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_std)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_std, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

RESULTS["kmeans_k_search"] = [
    {"k": int(k), "inertia": float(i), "silhouette": float(s)}
    for k, i, s in zip(ks, inertias, silhouettes)
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(ks, inertias, "o-")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow")
axes[1].plot(ks, silhouettes, "o-", color="orange")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette"); axes[1].set_title("Silhouette")
fig.tight_layout()
fig.savefig(CHARTS / "kmeans_k_selection.png", dpi=120)
plt.close(fig)

# Pick k by best silhouette among k>=3 (k=2 typically wins on positive-Likert
# data because it splits happy/unhappy — uninteresting for personas)
candidates = [(k, s) for k, s in zip(ks, silhouettes) if k >= 3]
best_k = max(candidates, key=lambda t: t[1])[0]
print(f"\nSelected k = {best_k} (best silhouette among k>=3)")
RESULTS["kmeans_chosen_k"] = int(best_k)

km_final = KMeans(n_clusters=best_k, n_init=50, random_state=42)
data["Cluster"] = km_final.fit_predict(X_std)

# %% Cluster profiles
profile = data.groupby("Cluster")[ITEMS].mean().round(2)
profile["n"] = data.groupby("Cluster").size()
print("\nCluster profiles (mean by item):")
print(profile)
profile.to_csv(OUT / "cluster_profiles.csv")
RESULTS["cluster_profiles"] = profile.reset_index().to_dict(orient="records")

# Profile shape: deviation from grand mean (so personas are distinguished by
# *what* they care about, not just overall positivity)
grand_mean = X.mean()
shape = (data.groupby("Cluster")[ITEMS].mean() - grand_mean).round(2)
print("\nCluster shape (deviation from grand mean):")
print(shape)
shape.to_csv(OUT / "cluster_shape.csv")
RESULTS["cluster_shape"] = shape.reset_index().to_dict(orient="records")

# Parallel-coordinates chart of cluster shape
fig, ax = plt.subplots(figsize=(10, 5))
for c, row in shape.iterrows():
    ax.plot(ITEMS, row.values, "o-", label=f"Cluster {c} (n={profile.loc[c, 'n']})")
ax.axhline(0, color="grey", linestyle=":")
ax.set_ylabel("Deviation from grand mean")
ax.set_title("Cluster profile shapes")
ax.legend(loc="best")
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(CHARTS / "cluster_profiles_shape.png", dpi=120)
plt.close(fig)

# %% K-medoids sensitivity check
print("\nK-medoids sensitivity check")
from scipy.spatial.distance import pdist, squareform
dist = squareform(pdist(X_std, metric="euclidean"))
kmd = KMedoids(n_clusters=best_k, method="fasterpam", random_state=42)
kmd_labels = kmd.fit(dist).labels_
ari = adjusted_rand_score(data["Cluster"], kmd_labels)
print(f"Adjusted Rand Index (k-means vs k-medoids): {ari:.3f}")
RESULTS["kmeans_kmedoids_ari"] = float(ari)

# %% [markdown]
# Section 11 — CART decision tree mapping segments to clusters

# %% CART decision tree
print("\n" + "=" * 60)
print("CART decision tree: Usage + VRC+ -> Cluster")
print("=" * 60)
seg = pd.get_dummies(
    data[["Usage Frequency", "VRC+"]],
    drop_first=False,
).astype(int)
tree = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=20)
tree.fit(seg, data["Cluster"])
tree_acc = tree.score(seg, data["Cluster"])
print(f"Tree training accuracy: {tree_acc:.3f}")
print(export_text(tree, feature_names=list(seg.columns)))
RESULTS["cart_accuracy"] = float(tree_acc)
RESULTS["cart_rules"] = export_text(tree, feature_names=list(seg.columns))

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(tree, feature_names=list(seg.columns),
          class_names=[f"C{c}" for c in sorted(data["Cluster"].unique())],
          filled=True, ax=ax, fontsize=8)
fig.tight_layout()
fig.savefig(CHARTS / "cart_decision_tree.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Section 12 — Selection-bias diagnostics (cluster × segment composition)

# %% Cluster composition by segment
usage_by_cluster = pd.crosstab(data["Cluster"], data["Usage Bin"], normalize="index").round(3) * 100
vrcplus_by_cluster = pd.crosstab(data["Cluster"], data["VRC+"], normalize="index").round(3) * 100
print("\n% Usage Bin by Cluster:")
print(usage_by_cluster)
print("\n% VRC+ by Cluster:")
print(vrcplus_by_cluster)
RESULTS["usage_by_cluster"] = usage_by_cluster.reset_index().to_dict(orient="records")
RESULTS["vrcplus_by_cluster"] = vrcplus_by_cluster.reset_index().to_dict(orient="records")

# Overall usage skew (for the bias callout)
usage_total = (data["Usage Bin"].value_counts(normalize=True) * 100).round(1).to_dict()
print(f"\nOverall usage distribution: {usage_total}")
RESULTS["usage_distribution"] = usage_total

fig, ax = plt.subplots(figsize=(8, 4))
usage_by_cluster.plot(kind="bar", stacked=True, ax=ax,
                     color=["#2ca02c", "#ff7f0e", "#d62728"])
ax.set_ylabel("% within cluster")
ax.set_title("Cluster composition by Usage Bin")
ax.legend(title="Usage", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(CHARTS / "cluster_by_usage.png", dpi=120)
plt.close(fig)

# %% [markdown]
# Section 13 — Enforcement Balance directional deep-dive

# %% Enforcement Direction by segment
ed_label = data["Enforcement Direction"].map({
    -2: "Far too little", -1: "Slightly too little", 0: "About right",
    1: "Slightly too much", 2: "Far too much",
})
overall_dir = ed_label.value_counts(normalize=True).reindex(
    ["Far too little", "Slightly too little", "About right",
     "Slightly too much", "Far too much"]
).fillna(0).round(3) * 100
print("\nEnforcement Balance — overall %:")
print(overall_dir)
RESULTS["enforcement_overall"] = overall_dir.to_dict()

dir_by_usage = pd.crosstab(data["Usage Bin"], ed_label, normalize="index").round(3) * 100
dir_by_vrcplus = pd.crosstab(data["VRC+"], ed_label, normalize="index").round(3) * 100
print("\nEnforcement Balance by Usage Bin (%):")
print(dir_by_usage)
print("\nEnforcement Balance by VRC+ (%):")
print(dir_by_vrcplus)
RESULTS["enforcement_by_usage"] = dir_by_usage.reset_index().to_dict(orient="records")
RESULTS["enforcement_by_vrcplus"] = dir_by_vrcplus.reset_index().to_dict(orient="records")

cat_order = ["Far too little", "Slightly too little", "About right",
             "Slightly too much", "Far too much"]
fig, ax = plt.subplots(figsize=(8, 4))
dir_by_usage[cat_order].plot(
    kind="bar", stacked=True, ax=ax,
    color=["#67000d", "#cb181d", "#a1d99b", "#6baed6", "#08519c"],
)
ax.set_ylabel("%"); ax.set_title("Enforcement Balance — by Usage Bin")
ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(CHARTS / "enforcement_by_usage.png", dpi=120)
plt.close(fig)

# %% Cronbach's alpha — Experience block sanity check
import pingouin as pg
exp_items = [ITEM_LABELS[i] for i in EXPERIENCE_IDX]
alpha = pg.cronbach_alpha(data[exp_items])
print(f"\nExperience block Cronbach's alpha: {alpha[0]:.3f} "
      f"(95% CI {alpha[1][0]:.3f}–{alpha[1][1]:.3f})")
RESULTS["experience_cronbach_alpha"] = float(alpha[0])
RESULTS["experience_cronbach_ci"] = [float(alpha[1][0]), float(alpha[1][1])]

# %% Dump RESULTS for the report writers
with open(OUT / "results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nWrote {OUT / 'results.json'}")
print("All charts under:", CHARTS)
