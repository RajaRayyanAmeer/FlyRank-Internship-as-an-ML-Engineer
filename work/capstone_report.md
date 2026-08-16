# Capstone Report — Refresh and Content Opportunity Scoring

- **Author:** Raja Rayyan Ameer
- **Lane:** Refresh and Content Opportunity Scoring
- **Repo:** https://github.com/RajaRayyanAmeer/FlyRank-Internship-as-an-ML-Engineer
- **Date:** 2026-08-16

## 1. Problem Framing

**Unit of analysis:** one row = one content item (page), identified by pseudonymous
`content_id`. **Output:** a ranked score (0–100) plus reason codes and a suggested action
(`refresh`, `refresh_and_review_ctr`, `expand_and_refresh`, `monitor`). **Action a human
takes:** a content reviewer works down the ranked queue weekly, starting at rank 1, and
manually verifies/executes the suggested action. **Cost of a wrong call:** a false positive
wastes reviewer time on a page that didn't need attention; a false negative lets a genuinely
declining page go unreviewed another cycle — asymmetric but bounded, since a human always
verifies before acting. **Why ML helps:** with 30,000+ pages and no human able to review them
all, a ranked, explainable shortlist turns an unbounded backlog into a tractable weekly list —
and a learned model captures interaction effects (e.g. position × CTR × freshness together)
that a single hand-rule can't.

## 2. Data Safety

Used only `content_refresh_anonymized.csv` (30,000 rows, 32 pseudonymized clients). Excluded
by the release itself: client names, domains, URLs, titles, keywords, raw queries,
credentials. **Deliberately excluded from features:** `trend_direction` and `trend_pct` — both
are the source of the label (`is_declining_label = trend_direction == "down"`) and would leak
the answer directly into the input. `content_id`/`client_id` are used only for
grouping/deduplication and the client-holdout split — never as model features. No
client-identifying detail appears anywhere in `work/`.

## 3. Baseline

A transparent, weighted hand-rule: `0.40 × visibility_score + 0.30 × freshness_risk_score +
0.25 × position_opportunity_score + 0.05 × depth_gap_score`, each component a percentile rank
of an observable signal. It's a fair comparison because it's evaluated on the **same
client-holdout split and the same metric** as every model. Baseline Precision@50 = **0.240**
(ROC AUC 0.627, avg. precision 0.468).

## 4. Model / Analysis

Three classifiers compared: logistic regression, decision tree, random forest — chosen because
this is a binary classification/ranking problem (declining vs. not) where tree ensembles
typically handle mixed numeric/categorical, non-linear signals well, while logistic regression
serves as an interpretable linear reference point. **Feature list:**
`MODEL_NUMERIC_FEATURES`/`MODEL_CATEGORICAL_FEATURES` (`scripts/ml_utils.py`) — log-transformed
impressions/clicks/sessions, `avg_position`, `ctr`, `engagement_rate`, `scroll_rate`, content
age/freshness fields, `word_count`/`char_count`, tiers, `content_type`, `main_intent`,
`competition_level`. **Left out on purpose:** `trend_direction`, `trend_pct` (leakage),
`content_id`/`client_id` (identifiers, not signal), `provider_used`/`model_used` (per data
dictionary, explicitly not model features). **Target, one sentence:** predict whether a
content item's impressions declined more than 20% month-over-month (`is_declining_label`), as
a proxy for "needs review now."

## 5. Evaluation

**Split:** client-holdout — ~20% of clients withheld entirely so no client's pages appear in
both train and test, with a stratified row-holdout fallback if a random client split can't
preserve both classes. This tests cross-client generalization, not memorized per-client
baselines. **Base rate:** 54.2% of rows are labeled declining — precision@50 must be read
against this; a naive "always predict declining" rule would already score high on raw
accuracy, which is why Precision@50 and lift-over-baseline (not accuracy alone) are the
reported metrics.

| Model | ROC AUC | Avg. Precision | Precision@50 |
|---|---:|---:|---:|
| baseline_rules | 0.627 | 0.468 | 0.240 |
| logistic_regression | 0.700 | 0.522 | 0.400 |
| decision_tree | 0.742 | 0.575 | 0.540 |
| random_forest | 0.750 | 0.618 | **0.740** |

**Error analysis:** Random Forest's ROC AUC (0.750) is only modestly above the baseline's
(0.627), but Precision@50 shows a much larger gap (0.740 vs 0.240) — the model is specifically
better at getting the *top-ranked* items right, which is what the reviewer actually consumes,
more than it is better at overall discrimination across the full population.

## 6. Interpretation

Top features by importance: `days_with_impressions` (0.158), `log_impressions_90d` (0.129),
`avg_position` (0.109), `content_age_days` (0.095) — visibility and position dominate over
content-depth signals like `word_count` (0.040). In plain words: the model leans most on
*how consistently and how visibly* a page has been showing up in search, not on how long the
article is. This matches the signal audit: staleness alone and word count alone were **MIXED**
signals (no clean, monotonic relationship with decline), while CTR-relative-to-position was
**CONFIRMED**. The negative result — word count not mattering much on its own — is itself
useful: it says "make it longer" is not, by itself, a safe refresh recommendation.

## 7. Recommendation

1. High-confidence `refresh_and_review_ctr` pages first — visible, well-positioned, under-clicked.
2. `expand_and_refresh` — small group, thin content with real demand.
3. `refresh` — declining-with-demand or stale visible pages.
4. `monitor` — no action this cycle.

A FlyRank editor opens the ranked queue Monday morning, works the top ~50 rows (matching the
Precision@50 metric choice), verifies each page manually, and acts per the suggested action.
**Confidence:** high for the top-ranked CTR/decline segment (backed by a confirmed signal and a
3x precision lift); low-to-moderate for any single-signal explanation (staleness, length)
in isolation. **Limits:** this is decision support, not an automated publish/unpublish trigger,
and is not evidence of causal refresh impact or algorithmic prediction.

## 8. Reproducibility

```bash
git clone <this-repo-url>
cd flyrank-ml-internship-starter
pip install -r requirements.txt
python scripts/run_all.py
```

Random seed: `42` (fixed throughout `scripts/03_train_model.py`). Environment: per
`requirements.txt` (pandas, numpy, scikit-learn, matplotlib, reportlab, duckdb,
huggingface_hub) — random forest's third-decimal Precision@50 is documented as
version-sensitive (~0.68–0.74 range); the ~3x lift over baseline is the stable claim.
Full notebook trail: `work/notebooks/w01_research_question.ipynb` through
`w07_action_playbook.ipynb`, `w04_signal_audit.ipynb`, `capstone.ipynb`.

---

> **Claims checklist:** observed / measured / directional / decision-support language
> everywhere · base rate (54.2%) reported next to Precision@50 · no causal claims · no
> "predicted Google's algorithm" · no client-identifying details · numbers match a fresh
> re-run of `scripts/run_all.py`.
