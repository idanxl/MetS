Model training and evaluation pipeline for survival prediction of metabolic syndrome.

The analysis is performed separately for the low-risk (met0) and high-risk
(met12) cohorts and includes three survival models: Random Survival Forest
(RSF), Gradient Boosting Survival Analysis (GBSA), and Cox proportional hazards.

Pipeline:
1. Split patients into training and held-out test sets.
2. Create two independent 10,000-patient training samples.
3. Tune preliminary models using four-fold group cross-validation.
4. Train preliminary models and rank features using SHAP (RSF/GBSA)
   or absolute penalized Cox coefficients.
5. Tune parsimonious models using the top 50 features and the second sample.
6. Train the final parsimonious models on the full training set.

Patient-level grouping is maintained throughout data splitting and
cross-validation to prevent observations from the same patient appearing
in both training and validation/test sets.
