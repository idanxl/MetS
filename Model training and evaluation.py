"""
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
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from lifelines import CoxPHFitter
import optuna
import shap
import joblib
import time

#######################################
## Machine learning data organization #
#######################################

# Patient-level 80/20 train-test split
for PREFIX in ["met0", "met12"]:

    path = fr"{PATH_TO_ML_DF}"
    df = pd.read_parquet(path)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(df, groups=df['patient_id'])
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df.to_parquet(f"train_{PREFIX}.parquet", index = False)
    test_df.to_parquet(f"test_{PREFIX}.parquet", index = False)

# Create two independent 10,000-patient samples from the training set
for PREFIX in ["met0", "met12"]:

    path = fr"train_{PREFIX}.parquet"
    df = pd.read_parquet(path)

    # Sample 1 for preliminary hyperparameter tuning and feature selection
    sample_patients1 = (
        df['patient_id']
            .drop_duplicates()
            .sample(n=10000, random_state=42)
    )
    df_sample1 = df[
        df['patient_id'].isin(sample_patients1)
    ].copy()


    df_sample1.to_parquet(f"train_{PREFIX}_10K_sample1.parquet", index = False)

    remaining_patients = (
        df['patient_id']
            .drop_duplicates()
            .loc[lambda x: ~x.isin(sample_patients1)]
    )
    
    # Sample 2 for parsimonious-model hyperparameter tuning
    sample2_patients = remaining_patients.sample(
        n=10000,
        random_state=43
    )

    df_sample2 = df[
        df['patient_id'].isin(sample2_patients)
    ].copy()

    df_sample2.to_parquet(f"train_{PREFIX}_10K_sample2.parquet", index = False)

#######################################
######## Cox data organization ########
#######################################

# Patient-level 80/20 train-test split
for PREFIX in ["met0", "met12"]:

    path = fr"{PATH_TO_COX_DF}"
    df = pd.read_parquet(path)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(df, groups=df['patient_id'])
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df.to_parquet(f"train_cox_{PREFIX}.parquet", index = False)
    test_df.to_parquet(f"test_cox_{PREFIX}.parquet", index = False)

# Create two independent 10,000-patient samples from the training set
for PREFIX in ["met0", "met12"]:

    path = fr"train_cox_{PREFIX}.parquet"
    df = pd.read_parquet(path)

    # Sample 1 for preliminary hyperparameter tuning and feature selection
    sample_patients1 = (
        df['patient_id']
            .drop_duplicates()
            .sample(n=10000, random_state=42)
    )
    df_sample1 = df[
        df['patient_id'].isin(sample_patients1)
    ].copy()


    df_sample1.to_parquet(f"train_cox_{PREFIX}_10K_sample1.parquet", index = False)

    remaining_patients = (
        df['patient_id']
            .drop_duplicates()
            .loc[lambda x: ~x.isin(sample_patients1)]
    )
    
    # Sample 2 for parsimonious-model hyperparameter tuning
    sample2_patients = remaining_patients.sample(
        n=10000,
        random_state=43
    )

    df_sample2 = df[
        df['patient_id'].isin(sample2_patients)
    ].copy()

    df_sample2.to_parquet(f"train_cox_{PREFIX}_10K_sample2.parquet", index = False)


#######################################
######## Hyperparameter tuning ########
#######################################


def objective_preliminary(trial, model_type="RSF"):
    """
    Optuna objective for preliminary RSF or GBSA hyperparameter tuning.

    Evaluates each parameter configuration using four-fold GroupKFold
    cross-validation and returns the mean Harrell C-index.
    """
    start = time.time()

    # RSF
    if model_type=="RSF":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 250, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 250),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.2, 0.9),
        }
        model_class = RandomSurvivalForest

    # GBSA
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 250, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 250),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "subsample": trial.suggest_float("subsample", 0.2, 0.9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
        model_class = GradientBoostingSurvivalAnalysis

    kf = GroupKFold(n_splits=4)
    c_indexes = []

    # Four-fold patient-grouped cross-validation
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y, groups)):

        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y[train_index], y[val_index]

        if model_type == "RSF":
            model = model_class(**params, random_state=42, n_jobs=-1)
        else:
            model = model_class(**params, random_state=42)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        c_index = concordance_index_censored(y_val["event"], y_val["duration_years"], preds)[0]
        c_indexes.append(c_index)

    avg_cindex = np.mean(c_indexes)
    end = time.time()
    duration = end - start
    trial.set_user_attr("duration_seconds", duration)

    return avg_cindex

def objective_cox_preliminary(trial):
    """
    Optuna objective for preliminary penalized Cox model tuning.

    Penalizer strength and L1 ratio are evaluated using four-fold
    patient-grouped cross-validation with concordance index as the objective.
    """
    start = time.time()
    penalizer = trial.suggest_float("penalizer", 0.01, 0.2, log=True)
    l1_ratio = trial.suggest_categorical("l1_ratio", [0.0, 0.25, 0.75, 1.0])
    
    kf = GroupKFold(n_splits=4)
    c_indexes = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(train_df, groups=groups)):
        
        fold_train = train_df.iloc[train_index].copy()
        fold_val = train_df.iloc[val_index].copy()

        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        model.fit(
            fold_train,
            duration_col="duration_years",
            event_col="event",
            show_progress=False
        )
        c_index = model.score(fold_val, scoring_method="concordance_index")

        c_indexes.append(float(c_index))


    avg_cindex = float(np.mean(c_indexes))
    end = time.time()
    duration = end - start
    trial.set_user_attr("duration_seconds", duration)

    return avg_cindex


# Tune preliminary RSF and GBSA models on sample 1
for PREFIX in ["met0", "met12"]:

    path = fr"train_{PREFIX}_10K_sample1.parquet"
    df = pd.read_parquet(path)
    groups = df['patient_id'].values
    X = df.drop(columns=["event", "duration_years", "duration_days", "patient_id","MetSsum_before_start"])
    y = Surv.from_dataframe("event", "duration_years", df)


    # GBSA
    study_gbsa = optuna.create_study(direction="maximize")
    with tqdm(total=20, desc="GBSA Progress") as pbar:
        study_gbsa.optimize(
            lambda trial: objective_preliminary(trial, model_type="GBSA"),
            n_trials=20,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    df_gbsa = study_gbsa.trials_dataframe()
    df_gbsa.to_csv(f"HPT_GBSA_{PREFIX}.csv", index=False)


    # RSF
    study_rsf = optuna.create_study(direction="maximize")
    with tqdm(total=20, desc="RSF Progress") as pbar:
        study_rsf.optimize(
            lambda trial: objective_preliminary(trial, model_type="RSF"),
            n_trials=20,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    df_rsf = study_rsf.trials_dataframe()
    df_rsf.to_csv(f"HPT_RSF_{PREFIX}.csv", index=False)


# Tune preliminary Cox models on sample 1
for prefix in ["met0", "met12"]:

    train_df = pd.read_parquet(fr"train_cox_{prefix}_10K_sample1.parquet")
    groups = train_df["patient_id"].copy()

    drop_cols = [
        "duration_days",
        "patient_id",
        "MetSsum_before_start"
    ]

    train_df = train_df.drop(columns=drop_cols)
    
    # Run Optuna hyperparameter search
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective_cox_preliminary, n_trials = 20, gc_after_trial = True)

    # Save all Optuna trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"HPT_COX_{prefix}.csv", index=False, encoding="utf-8")


#######################################
########## Preliminary models #########
#######################################

# Fit preliminary models using tuned parameters and derive feature rankings
PREFIX = [ "met0",
          "met12"]

# Hyperparameters selected from preliminary tuning
model_configs = {
     "met0": {
         "RSF_met0": RandomSurvivalForest(
             n_estimators=57,
             max_depth=24,
             min_samples_split=29,
             min_samples_leaf=49,
             max_features=0.596,
             bootstrap=True,
             max_samples=0.814,
             n_jobs=3,
             random_state=42,
             verbose=1
         ),
         "GBSA_met0": GradientBoostingSurvivalAnalysis(
             n_estimators=90,
             max_depth=6,
             min_samples_split=92,
             min_samples_leaf=72,
             max_features=0.582,
             subsample=0.569,
             learning_rate=0.137,
             random_state=42,
             verbose=1
         )
     },

    "met12": {
        "RSF_met12": RandomSurvivalForest(
            n_estimators=33,
            max_depth=37,
            min_samples_split=60,
            min_samples_leaf=21,
            max_features=0.167,
            bootstrap=False,
            n_jobs=3,
            random_state=42,
            verbose=1
        ),
        "GBSA_met12": GradientBoostingSurvivalAnalysis(
            n_estimators=86,
            max_depth=4,
            min_samples_split=244,
            min_samples_leaf=138,
            max_features=0.758,
            subsample=0.756,
            learning_rate=0.116,
            random_state=42,
            verbose=1
         )
     }
}

# Train preliminary RSF/GBSA models and calculate SHAP feature importance
for prefix in PREFIX:

    path = rf"train_{prefix}_10K_sample1.parquet"
    df = pd.read_parquet(path)

    X = df.drop(columns=["event", "duration_years", "duration_days", "patient_id", "MetSsum_before_start"])
    y = Surv.from_dataframe("event", "duration_years", df)
    y_strat = df["event"].astype(int)
    groups = df["patient_id"]

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=51
    )

    train_idx, val_idx = next(
        splitter.split(X, y_strat, groups=groups)
    )

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_train = y[train_idx]

    models = model_configs[prefix]

    for model_name, model_base in models.items():

        model = clone(model_base)
        model.fit(X_train, y_train)


        # Background sample for the SHAP explainer
        X_background = X_train.sample(
            n=min(50, len(X_train)),
            random_state=42
        )

        # Observations used for SHAP value estimation
        X_explain = X_val.sample(
            n=min(1200, len(X_val)),
            random_state=42
        )

        explainer = shap.Explainer(
            model.predict,
            X_background,
            algorithm="permutation",
            seed=42
        )

        n_permutations = 3
        max_evals = n_permutations * (2 * X_explain.shape[1] + 1)

        shap_values = explainer(
            X_explain,
            max_evals=max_evals
        )
        
        # Global feature importance: mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        shap_importance = pd.DataFrame({
            "feature": X_explain.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(
            "mean_abs_shap",
            ascending=False
        )
        shap_importance["rank"] = range(1, len(shap_importance) + 1)

        shap_importance.to_csv(f"shap_importance_{model_name}.csv", index=False)

        # Save preliminary model, SHAP values, and feature ranking
        joblib.dump(model, f"model_{model_name}.joblib")
        joblib.dump(shap_values, f"shap_values_{model_name}.joblib")        
        
        
# Hyperparameters selected from preliminary Cox tuning
COX_PARAMS = {
    "met0": {
        "penalizer": 0.050,
        "l1_ratio": 0.0
    },
    "met12": {
        "penalizer": 0.023,
        "l1_ratio": 0.25
    }
}

# Fit preliminary Cox models and rank features by absolute coefficient magnitude
for prefix in COX_PARAMS:

    df = pd.read_parquet(fr"train_cox_{prefix}_10K_sample1.parquet")

    groups = df["patient_id"]
    y_strat = df["event"].astype(int)

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=51
    )

    train_idx, val_idx = next(
        splitter.split(df, y_strat, groups=groups)
    )

    train_df = df.iloc[train_idx].copy()

    drop_cols = [
    "duration_days",
    "patient_id",
    "MetSsum_before_start"
    ]

    train_df = train_df.drop(columns=drop_cols)

    # Fit preliminary Cox model
    cox_params = COX_PARAMS[prefix]

    cph = CoxPHFitter(
        penalizer=cox_params["penalizer"],
        l1_ratio=cox_params["l1_ratio"]
    )

    cph.fit(
        train_df,
        duration_col="duration_years",
        event_col="event"
    )
    cox_summary = cph.summary.copy()
    # Rank predictors by absolute penalized coefficient magnitude
    cox_summary["abs_coef"] = cox_summary["coef"].abs()

    cox_summary = (
        cox_summary
        .sort_values("abs_coef", ascending=False)
        .reset_index(names="variable")
    )

    cox_summary.to_csv(
        f"cox_importance_{prefix}.csv",
        index=False
    )
    joblib.dump(cph, f"model_cox_{prefix}.joblib")


#######################################
## Parsimonious hyperparameter tuning #
#######################################

def objective_parsimonious(trial, model_type="RSF"):
    """
    Optuna objective for parsimonious RSF or GBSA hyperparameter tuning.

    Models are evaluated using the 50 highest-ranked features and four-fold
    patient-grouped cross-validation. The mean Harrell C-index is optimized.
    """
    start = time.time()
    #   RSF
    if model_type=="RSF":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 250, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 250),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_samples": trial.suggest_float("max_samples", 0.2, 0.9),
        }
        model_class = RandomSurvivalForest

    #    GBSA
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 250, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 250),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "subsample": trial.suggest_float("subsample", 0.2, 0.9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
        model_class = GradientBoostingSurvivalAnalysis

    kf = GroupKFold(n_splits=4)
    c_indexes = []

    # Four-fold patient-grouped cross-validation
    for train_index, val_index in kf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y[train_index], y[val_index]

        if model_type == "RSF":
            model = model_class(**params, random_state=42, n_jobs=-1)
        else:
            model = model_class(**params, random_state=42)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        c_index = concordance_index_censored(y_val["event"], y_val["duration_years"], preds)[0]
        c_indexes.append(c_index)

    avg_cindex = np.mean(c_indexes)
    std_cindex = np.std(c_indexes)

    trial.set_user_attr("c_index_mean", avg_cindex)
    trial.set_user_attr("c_index_std", std_cindex)
    trial.set_user_attr("params", params)
    end = time.time()
    duration = end - start
    trial.set_user_attr("duration_seconds", duration)

    return avg_cindex



for PREFIX in [ "met0",
                "met12"]:
    path = fr"train_{PREFIX}_10K_sample2.parquet"
    df = pd.read_parquet(path)

    y = Surv.from_dataframe("event", "duration_years", df)
    groups = df['patient_id'].values

    # Select the 50 highest-ranked GBSA features from preliminary SHAP analysis
    path = rf"shap_importance_GBSA_{PREFIX}.csv"

    feature_df = pd.read_csv(path)

    TOP50_FEATURES = (
        feature_df
        .sort_values("mean_abs_shap", ascending=False)
        .head(50)["feature"]
        .dropna()
        .tolist()
    )

    X = df[TOP50_FEATURES].copy()
    study_gbsa = optuna.create_study(direction="maximize")
    with tqdm(total=20, desc="GBSA Progress") as pbar:
        study_gbsa.optimize(
            lambda trial: objective_parsimonious(trial, model_type="GBSA"),
            n_trials=20,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    df_gbsa = study_gbsa.trials_dataframe()
    df_gbsa.to_csv(f"HPT_Pars_GBSA_{PREFIX}.csv", index=False)

    # Select the 50 highest-ranked RSF features from preliminary SHAP analysis
    path = rf"shap_importance_RSF_{PREFIX}.csv"

    feature_df = pd.read_csv(path)

    TOP50_FEATURES = (
        feature_df
        .sort_values("mean_abs_shap", ascending=False)
        .head(50)["feature"]
        .dropna()
        .tolist()
    )

    X = df[TOP50_FEATURES].copy()
    study_rsf = optuna.create_study(direction="maximize")
    with tqdm(total=20, desc="RSF Progress") as pbar:
        study_rsf.optimize(
            lambda trial: objective_parsimonious(trial, model_type="RSF"),
            n_trials=20,
            callbacks=[lambda study, trial: pbar.update(1)]
        )
    df_rsf = study_rsf.trials_dataframe()
    df_rsf.to_csv(f"HPT_Pars_RSF_{PREFIX}.csv", index=False)


def objective_cox_parsimonious(trial):
    """
    Optuna objective for parsimonious Cox model hyperparameter tuning.

    The top 50 preliminary Cox features are evaluated using four-fold
    patient-grouped cross-validation and concordance index.
    """
    start = time.time()
    penalizer = trial.suggest_float("penalizer", 0.01, 0.2, log=True)
    l1_ratio = trial.suggest_categorical("l1_ratio", [0.0, 0.25, 0.75, 1.0])

    kf = GroupKFold(n_splits=4)
    c_indexes = []

    for train_index, val_index in kf.split(train_df, groups=groups):
        fold_train = train_df.iloc[train_index].copy()
        fold_val = train_df.iloc[val_index].copy()

        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        model.fit(
            fold_train,
            duration_col="duration_years",
            event_col="event",
            show_progress=False
        )
        c_index = model.score(fold_val, scoring_method="concordance_index")

        c_indexes.append(float(c_index))


    avg_cindex = float(np.mean(c_indexes))
    end = time.time()
    duration = end - start
    trial.set_user_attr("duration_seconds", duration)

    return avg_cindex


# Cox hyperparameter tuning
for prefix in ["met0", "met12"]:

    train_df = pd.read_parquet(fr"train_cox_{prefix}_10K_sample2.parquet")
    groups = train_df["patient_id"].copy()

    # Select the 50 highest-ranked Cox predictors
    feature_df = pd.read_csv(
        f"cox_importance_{prefix}.csv"
    )

    TOP50_FEATURES = (
        feature_df
        .sort_values("abs_coef", ascending=False)
        .head(50)["variable"]
        .dropna()
        .tolist()
    )

    train_df = train_df[
        TOP50_FEATURES + ["duration_years", "event"]
    ].copy()

    # Run Optuna hyperparameter search
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective_cox_parsimonious, n_trials = 20, gc_after_trial = True)

    # Save all Optuna trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"cox_Pars_HPT_{prefix}.csv", index=False, encoding="utf-8")

#######################################
######### Parsimonious models #########
#######################################

# Train final top-50-feature models on the full training set
PREFIX = [ "met0",
          "met12"]

# Hyperparameters selected from parsimonious-model tuning
model_configs = {
    "met0": {
        "RSF_met0": RandomSurvivalForest(
            n_estimators=68,
            max_depth=18,
            min_samples_split=39,
            min_samples_leaf=38,
            max_features=0.276,
            bootstrap=False,
            max_samples=None,
            n_jobs=3,
            random_state=42,
            verbose=1
        ),
        "GBSA_met0": GradientBoostingSurvivalAnalysis(
            n_estimators=92,
            max_depth=6,
            min_samples_split=95,
            min_samples_leaf=73,
            max_features=0.663,
            subsample=0.451,
            learning_rate=0.192,
            random_state=42,
            verbose=1
         )
     },

    "met12": {
        "RSF_met12": RandomSurvivalForest(
            n_estimators=48,
            max_depth=15,
            min_samples_split=116,
            min_samples_leaf=8,
            max_features=0.124,
            bootstrap=True,
            max_samples=0.538,
            n_jobs=3,
            random_state=42,
            verbose=1
        ),
        "GBSA_met12": GradientBoostingSurvivalAnalysis(
            n_estimators=90,
            max_depth=6,
            min_samples_split=58,
            min_samples_leaf=141,
            max_features=0.761,
            subsample=0.562,
            learning_rate=0.107,
            random_state=42,
            verbose=1
         )
    }
}

# Fit final RSF and GBSA models using their respective top 50 features
for prefix in PREFIX:

    path = rf"train_{prefix}.parquet"
    train_df = pd.read_parquet(path)

    y_train = Surv.from_dataframe("event", "duration_years", train_df)

    models = model_configs[prefix]

    for model_name, model_base in models.items():

        feature_path = f"shap_importance_{model_name}.csv"
        feature_df = pd.read_csv(feature_path)

        features = (
            feature_df
            .sort_values("mean_abs_shap", ascending=False)
            .head(50)["feature"]
            .dropna()
            .tolist()
        )
        X_train = train_df[features].copy()
        model = clone(model_base)
        model.fit(X_train, y_train)

        # Save final parsimonious model
        joblib.dump(model, f"model_{model_name}_Pars.joblib")

MODEL_PREFIX = ["met0", "met12"]
COX_PARAMS = {
    "met0": {
        "penalizer": 0.010,
        "l1_ratio": 0.0
    },
    "met12": {
        "penalizer": 0.048,
        "l1_ratio": 0.0
    }
}

for prefix in MODEL_PREFIX:
    train_df = pd.read_parquet(fr"train_cox_{prefix}.parquet")

    # Select the top 50 Cox predictors
    top50_df = pd.read_csv(
        f"cox_importance_{prefix}.csv"
    )

    top50_features = (
        top50_df
        .sort_values("abs_coef", ascending=False)
        .head(50)["variable"]
        .dropna()
        .tolist()
    )
    
    # Restrict the training data to the selected predictors
    pars_train = train_df[
        top50_features + ["duration_years", "event"]
        ].copy()

    cox_params = COX_PARAMS[prefix]

    cph_pars = CoxPHFitter(
        penalizer=cox_params["penalizer"],
        l1_ratio=cox_params["l1_ratio"]
    )

    cph_pars.fit(
        pars_train,
        duration_col="duration_years",
        event_col="event"
    )

    # Save final parsimonious Cox model
    joblib.dump(
        cph_pars,
        fr"model_cox_pars_{prefix}.joblib"
    )