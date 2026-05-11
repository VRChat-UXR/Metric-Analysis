# Monthly Metrics Survey — Analysis Code

A Python pipeline for analyzing VRChat's Monthly Metrics Survey, a UXR program tracking three aggregate health metrics — **Experience**, **CPV** (Customer Perceived Value), and **Trust & Safety** — across 10 Likert-scale items. The code covers descriptive Likert analysis, exploratory and confirmatory factor analysis (EFA / CFA), k-means clustering with a k-medoids stability check, CART decision trees for segment-to-cluster mapping, and cross-wave trend comparison.

This repository is the **analysis component only**. Raw survey responses, generated outputs, and per-wave narrative reports are produced and held privately by the VRChat UXR team; they are intentionally excluded from this repo via `.gitignore`.

---

## What this code does

Two analysis streams, each designed to be re-executed for every monthly wave.

### `descriptive_stats/`
Per-item Likert distributions, top-2-box / bottom-2-box leaderboards, diverging stacked-bar visualizations, segmentation crosstabs (Usage Frequency, VRC+, account tenure), a Goldilocks-aware Enforcement Balance breakdown, and (across waves) two-proportion z-tests on T2B deltas.

- `py/Descriptive_Statistics.py` — single-wave descriptives.
- `py/Wave_Comparison.py` — cross-wave deltas, significance, trend charts.

### `cfa_efa_analysis/`
The multivariate model.

- **EFA** with KMO, Bartlett's test of sphericity, parallel analysis for factor count, and oblimin-rotated loadings.
- **CFA** on the proposed 3-pillar model via `semopy`, with RMSEA, CFI, manually-computed SRMR, and AIC/BIC.
- **K-means clustering** with silhouette-based k selection (forced k ≥ 3, see *Methodology rationale*).
- **K-medoids sensitivity check** via Adjusted Rand Index between k-means and FasterPAM.
- **CART decision tree** mapping segmentation variables to cluster membership.
- **Cronbach's α** on the multi-item Experience block.
- **Selection-bias diagnostics** (cluster × Usage × VRC+ composition).

Files:
- `py/Analysis_Exploration.py` — full EFA + CFA + cluster pipeline against a single wave; emits `results.json` for cross-wave aggregation.
- `py/Wave_Comparison.py` — reads each wave's `results.json` and tracks fit indices, α, cluster proportions, and cluster profile drift over time.

---

## Survey schema

One survey wave = one Excel file at the project root, named `UXR_Survey_Results_<YYYY-MM>.xlsx`. Row 0 is a SurveyMonkey header row and is dropped on load.

### Item map

| Col | Pillar | Sub-category | Scale |
|---|---|---|---|
| 14 | Experience | Ease of Use | 5-pt Likert (Strongly Disagree → Strongly Agree) |
| 15 | Experience | Discovery | 5-pt Likert |
| 16 | Experience | Emotional Experience | 5-pt Likert |
| 17 | Experience | Personalization | 5-pt Likert |
| 18 | Experience | Social Connection | 5-pt Likert |
| 19 | Experience | Visual Experience | 5-pt Likert |
| 20 | CPV | Experience Value | 5-pt Likert |
| 21 | CPV | Feature Value | 5-pt Likert |
| 22 | T&S | Participation Safety | 5-pt favorability (Very Difficult → Very Easy) |
| 23 | T&S | Enforcement Balance | 5-pt **Goldilocks** — middle is optimal |
| 24 | — | Usage Frequency (segmentation) | 8-level ordinal |
| 25 | — | VRC+ subscription (segmentation) | binary |

SurveyMonkey passthrough fields (account-level metadata, not survey responses):
- Col 8: VRChat user ID — never read or processed by the pipeline.
- Col 9: Tenure in months since activation (numeric; canonical tenure source).
- Col 10: Tenure pre-bucketed (Excel mangles one bucket — prefer col 9).
- Col 11: Primary platform.

### Encoding decisions

- Standard Likert items (cols 14–21): Strongly Disagree = 1 → Strongly Agree = 5.
- Participation Safety (col 22): Very Difficult = 1 → Very Easy = 5.
- Enforcement Balance (col 23) is **non-monotonic** — both extremes are unfavorable — so it's encoded **twice**:
  - **Folded** (used in EFA, CFA, clustering, correlations): Far too little = 1, Slightly too little = 3, About right = 5, Slightly too much = 3, Far too much = 1.
  - **Directional** (used in narrative reporting only): Far too little = −2 ... Far too much = +2.

The folded score is never averaged into a generic "happiness" composite with the other items — the underlying construct is qualitatively different from a positive-negative agreement scale.

---

## Methodology rationale

**Why a folded encoding for the Goldilocks item.** Multivariate methods (EFA, CFA, k-means) require a monotonic favorability axis. The raw 5-point order ("Far too little → Far too much") is non-monotonic — both extremes are bad — so the response options are remapped so 5 = optimal and 1 = either extreme. The directional version is preserved separately because under-enforcement vs over-enforcement implies very different product responses, and a folded score erases that distinction.

**Why CFA reports both relative and absolute fit.** RMSEA and CFI capture how well the proposed factor structure reproduces the correlation pattern (relative fit). SRMR captures absolute residual fit. A model can pass relative fit and fail absolute fit when local heterogeneity — e.g., a sub-construct with a different scale wording — inflates residuals. Reporting both prevents a "good RMSEA / CFI" headline from masking residual misfit.

**Why k ≥ 3 in k-means.** With 9 of 10 items positively worded, k = 2 reliably produces a "happy / unhappy" split — diagnostic of acquiescence bias and ceiling effects, not differentiated user types. Forcing k ≥ 3 surfaces any structure beyond overall positivity.

**Why a k-medoids sensitivity check.** k-means assumes spherical clusters in Euclidean space; k-medoids (FasterPAM) is robust to that assumption. The Adjusted Rand Index between the two clusterings flags soft- vs hard-edged cluster boundaries — when ARI is moderate, cluster membership should be reported as a tendency rather than a categorical label.

**Why Cronbach's α only on the Experience block.** The other two pillars have only two indicators each, where Spearman-Brown is more appropriate; T&S additionally has scale heterogeneity (favorability + Goldilocks-folded) that violates α's assumptions.

---

## Running the pipeline

The data file (`UXR_Survey_Results_<YYYY-MM>.xlsx`) is **not** in this repo and never will be — it contains user-level survey responses. A properly-named wave file must be placed at the project root before running.

### Setup
This project uses [Poetry](https://python-poetry.org/) with an in-project virtualenv at `.venv/`.

```
python -m poetry install
python -m poetry lock     # only after dependency changes
```

Python 3.11–3.14 supported.

### Per-wave analysis (re-run for each new wave)
```
python -m poetry run python cfa_efa_analysis/py/Analysis_Exploration.py
python -m poetry run python descriptive_stats/py/Descriptive_Statistics.py
```

Both scripts auto-detect the latest `UXR_Survey_Results_*.xlsx` at the project root. Pin a specific wave with `--wave YYYY-MM`.

### Cross-wave comparison
```
python -m poetry run python cfa_efa_analysis/py/Wave_Comparison.py
python -m poetry run python descriptive_stats/py/Wave_Comparison.py
```

Both gracefully handle the single-wave case — they write snapshot tables and skip trend charts.

### Output locations (all gitignored)
- `cfa_efa_analysis/csv/<wave>/`, `cfa_efa_analysis/png/<wave>/`
- `descriptive_stats/csv/<wave>/`, `descriptive_stats/png/<wave>/`
- `*/csv/comparisons/`, `*/png/comparisons/` — cross-wave artifacts.

Output directories are created on demand. Nothing under `csv/`, `png/`, or `md/` is committed.

---

## Library quirks worth knowing

- **`factor_analyzer` 0.5.x uses sklearn's deprecated `force_all_finite` kwarg.** `Analysis_Exploration.py` monkey-patches `sklearn.utils.validation.check_array` at import time to translate it to `ensure_all_finite`. Don't remove the shim until `factor_analyzer` fixes upstream.
- **`semopy` 2.x doesn't include SRMR by default.** SRMR is computed manually from observed vs implied covariance. CFA inputs are standardized so SRMR is on a correlation-residual scale.
- **`kmedoids`** is used in place of `scikit-learn-extra`, which doesn't build on Python 3.14.
- **`pingouin`** for Cronbach's α and nonparametric tests.
- **`openpyxl`** is required for `pd.read_excel` on `.xlsx`.

---

## Project structure

```
.
├── README.md                                  ← this file
├── .gitignore
├── pyproject.toml, poetry.lock, poetry.toml   ← Poetry env spec
│
├── cfa_efa_analysis/
│   └── py/
│       ├── Analysis_Exploration.py            ← per-wave EFA / CFA / clusters / CART
│       └── Wave_Comparison.py                 ← cross-wave fit / α / cluster trends
│
└── descriptive_stats/
    └── py/
        ├── Descriptive_Statistics.py          ← per-wave descriptives
        └── Wave_Comparison.py                 ← cross-wave deltas / sig tests / trends
```

The `csv/`, `png/`, and `md/` subdirectories under each analysis folder, along with the wave `.xlsx` files at the project root, are generated or authored locally and are gitignored.

---

## Adding a new wave

1. Drop `UXR_Survey_Results_<YYYY-MM>.xlsx` at the project root (local only — never committed).
2. Run both per-wave scripts.
3. Run both cross-wave scripts to refresh trend artifacts.

---

© 2026 VRChat. All rights reserved.
