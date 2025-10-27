# Archive: Datasets Chaos v1 (Invalid)

This dataset was archived on 10/27/2025 as part of the strategic thesis pivot.

**Reason:** Failed non-linearity validation. The Random Forest model underperformed Linear Regression by 1.82% (RF vs Linear Gain: -1.82%), failing to meet the >5% improvement target required to justify the use of advanced ML and XAI (SHAP).

**Status:** Invalid for thesis demonstration. Moved to legacy to preserve history.

**Root Cause:** Insufficient interaction strength, age-dependent variance, and pathway correlations in the data generation process.

**Next Steps:** Implementation of Issue #50 to generate datasets_chaos_v2 with stronger non-linear signals.

**See:** `docs/PROJECT_STATUS_OCT_2025.md` and `docs/PIVOT.md`.
