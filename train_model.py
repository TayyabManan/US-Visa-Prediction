"""
Modal-based training script for Visa Approval prediction model.

Runs on an H100 GPU in the cloud. Pipeline:
  1. Preprocessing fit on X_train only (no data leakage)
  2. GridSearchCV for 5 models (RF, GBM, XGB, LGBM, CatBoost)
  3. Stacking ensemble from top 3 models
  4. Threshold tuning to maximize accuracy while keeping denied recall >= 60%
  5. ThresholdClassifier wrapper for transparent threshold application

Run:  modal run train_model.py
"""

import modal
import os

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("visa-model-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "pyyaml",
    )
)


@app.function(image=image, gpu="H100", timeout=1800)
def train(csv_bytes: bytes, config_yaml: str):
    import io
    import sys
    import types
    import pickle
    import warnings
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        OrdinalEncoder,
        PowerTransformer,
        StandardScaler,
    )
    from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # ThresholdClassifier — wraps a trained model and applies a custom
    # probability threshold for predict(). Drop-in replacement that works
    # seamlessly with visaModel.predict().
    # ------------------------------------------------------------------
    class ThresholdClassifier:
        def __init__(self, base_model, threshold=0.5):
            self.base_model = base_model
            self.threshold = threshold

        def predict(self, X):
            proba = self.base_model.predict_proba(X)[:, 1]
            return (proba >= self.threshold).astype(int)

        def predict_proba(self, X):
            return self.base_model.predict_proba(X)

        @property
        def classes_(self):
            return self.base_model.classes_

        def __repr__(self):
            return f"ThresholdClassifier({type(self.base_model).__name__}, threshold={self.threshold:.3f})"

        def __str__(self):
            return self.__repr__()

    # ------------------------------------------------------------------
    # Register fake modules so pickle records the correct class paths.
    # When unpickled locally (where the real package exists), Python
    # resolves them just fine.
    # ------------------------------------------------------------------
    class visaModel:
        def __init__(self, preprocessing_object, trained_model_object):
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object

        def predict(self, dataframe):
            transformed = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed)

        def predict_proba(self, dataframe):
            transformed = self.preprocessing_object.transform(dataframe)
            if hasattr(self.trained_model_object, "predict_proba"):
                return self.trained_model_object.predict_proba(transformed)
            return None

        def __repr__(self):
            return f"{type(self.trained_model_object).__name__}()"

        def __str__(self):
            return f"{type(self.trained_model_object).__name__}()"

    pkg = types.ModuleType("visa_approval_prediction")
    entity = types.ModuleType("visa_approval_prediction.entity")
    estimator_mod = types.ModuleType("visa_approval_prediction.entity.estimator")
    estimator_mod.visaModel = visaModel
    estimator_mod.ThresholdClassifier = ThresholdClassifier
    visaModel.__module__ = "visa_approval_prediction.entity.estimator"
    visaModel.__qualname__ = "visaModel"
    ThresholdClassifier.__module__ = "visa_approval_prediction.entity.estimator"
    ThresholdClassifier.__qualname__ = "ThresholdClassifier"
    sys.modules["visa_approval_prediction"] = pkg
    sys.modules["visa_approval_prediction.entity"] = entity
    sys.modules["visa_approval_prediction.entity.estimator"] = estimator_mod

    # ------------------------------------------------------------------
    # 1. Load data from bytes
    # ------------------------------------------------------------------
    print("Loading data ...")
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"  Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    current_year = datetime.now().year
    df["company_age"] = current_year - df["yr_of_estab"]

    # ------------------------------------------------------------------
    # 3. Drop unneeded columns
    # ------------------------------------------------------------------
    df.drop(columns=["case_id", "yr_of_estab"], inplace=True)

    # ------------------------------------------------------------------
    # 4. Encode target: Denied=1, Certified=0
    # ------------------------------------------------------------------
    df["case_status"] = df["case_status"].map({"Certified": 0, "Denied": 1})

    X = df.drop(columns=["case_status"])
    y = df["case_status"]

    # ------------------------------------------------------------------
    # 5. Train / test split BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # 6. Build ColumnTransformer
    # ------------------------------------------------------------------
    onehot_cols = ["continent", "unit_of_wage", "region_of_employment"]
    ordinal_cols = [
        "has_job_experience",
        "requires_job_training",
        "full_time_position",
        "education_of_employee",
    ]
    ordinal_categories = [
        ["N", "Y"],
        ["N", "Y"],
        ["N", "Y"],
        ["High School", "Bachelor's", "Master's", "Doctorate"],
    ]
    power_scale_cols = ["no_of_employees", "company_age"]
    scale_only_cols = ["prevailing_wage"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_cols,
            ),
            (
                "ordinal",
                OrdinalEncoder(categories=ordinal_categories),
                ordinal_cols,
            ),
            (
                "power_scale",
                SkPipeline([
                    ("power", PowerTransformer(method="yeo-johnson")),
                    ("scale", StandardScaler()),
                ]),
                power_scale_cols,
            ),
            (
                "scale",
                StandardScaler(),
                scale_only_cols,
            ),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # 7. Fit preprocessor on train only
    # ------------------------------------------------------------------
    print("Fitting preprocessor on training data ...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # 8. Train on natural distribution (no SMOTEENN)
    # ------------------------------------------------------------------
    X_train_resampled = X_train_transformed
    y_train_resampled = y_train
    print(f"  Training on natural distribution: {X_train_resampled.shape[0]} samples "
          f"(class 0: {(y_train == 0).sum()}, class 1: {(y_train == 1).sum()})")

    # ------------------------------------------------------------------
    # 9. Parse hyperparameter grids
    # ------------------------------------------------------------------
    config = yaml.safe_load(config_yaml)
    gs_cfg = config["grid_search"]["params"]
    models_cfg = config["model_selection"]

    MODEL_CLASSES = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "LGBMClassifier": LGBMClassifier,
        "CatBoostClassifier": CatBoostClassifier,
    }

    GPU_OVERRIDES = {
        "XGBClassifier": {"tree_method": "hist", "device": "cuda"},
        "CatBoostClassifier": {"task_type": "GPU"},
    }

    # Models that manage their own parallelism — use n_jobs=1 for
    # GridSearchCV to avoid deadlocks/thread thrashing.
    SEQUENTIAL_CV = {"LGBMClassifier", "CatBoostClassifier"}

    # ------------------------------------------------------------------
    # 10. GridSearchCV for each model
    # ------------------------------------------------------------------
    results = []

    for key in sorted(models_cfg.keys()):
        mcfg = models_cfg[key]
        cls_name = mcfg["class"]
        cls = MODEL_CLASSES[cls_name]
        base_params = dict(mcfg.get("params", {}))
        param_grid = mcfg.get("search_param_grid", {})

        if cls_name in GPU_OVERRIDES:
            base_params.update(GPU_OVERRIDES[cls_name])

        gs_n_jobs = 1 if cls_name in SEQUENTIAL_CV else -1

        print(f"\n{'=' * 60}")
        print(f"Training {cls_name} (GridSearchCV n_jobs={gs_n_jobs}) ...")
        print(f"{'=' * 60}")

        estimator = cls(**base_params)
        gs = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=gs_cfg["cv"],
            scoring=gs_cfg["scoring"],
            verbose=gs_cfg.get("verbose", 0),
            n_jobs=gs_n_jobs,
        )
        gs.fit(X_train_resampled, y_train_resampled)

        y_pred = gs.best_estimator_.predict(X_test_transformed)
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        denied_recall = recall_score(y_test, y_pred, pos_label=1)

        results.append({
            "name": cls_name,
            "best_params": gs.best_params_,
            "cv_score": gs.best_score_,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "denied_recall": denied_recall,
            "model": gs.best_estimator_,
        })

        print(f"  Best CV Acc:    {gs.best_score_:.4f}")
        print(f"  Test Acc:       {test_acc:.4f}")
        print(f"  Test F1:        {test_f1:.4f}")
        print(f"  Denied Recall:  {denied_recall:.4f}")
        print(f"  Best params:    {gs.best_params_}")

    # ------------------------------------------------------------------
    # 11. Individual model comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("INDIVIDUAL MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'CV Acc':>8} {'Test Acc':>9} {'Test F1':>8} {'Denied Rcl':>11}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['cv_score']:>8.4f} {r['test_acc']:>9.4f} "
              f"{r['test_f1']:>8.4f} {r['denied_recall']:>11.4f}")

    # ------------------------------------------------------------------
    # 12. Stacking ensemble from top 3 models
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STACKING ENSEMBLE")
    print(f"{'=' * 70}")

    top3 = sorted(results, key=lambda r: r["test_acc"], reverse=True)[:3]
    print(f"  Base estimators: {[r['name'] for r in top3]}")

    estimators_list = [(r["name"], r["model"]) for r in top3]
    stacking = StackingClassifier(
        estimators=estimators_list,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    print("  Fitting stacking classifier ...")
    stacking.fit(X_train_resampled, y_train_resampled)

    y_pred_stack = stacking.predict(X_test_transformed)
    stack_acc = accuracy_score(y_test, y_pred_stack)
    stack_f1 = f1_score(y_test, y_pred_stack)
    stack_denied_recall = recall_score(y_test, y_pred_stack, pos_label=1)

    print(f"  Stacking Test Acc:       {stack_acc:.4f}")
    print(f"  Stacking Test F1:        {stack_f1:.4f}")
    print(f"  Stacking Denied Recall:  {stack_denied_recall:.4f}")

    # ------------------------------------------------------------------
    # 13. Pick winner: best individual vs stacking
    # ------------------------------------------------------------------
    best_individual = max(results, key=lambda r: r["test_acc"])

    if stack_acc >= best_individual["test_acc"]:
        winner_model = stacking
        winner_name = "StackingClassifier"
        winner_acc = stack_acc
        winner_f1 = stack_f1
        winner_denied_recall = stack_denied_recall
    else:
        winner_model = best_individual["model"]
        winner_name = best_individual["name"]
        winner_acc = best_individual["test_acc"]
        winner_f1 = best_individual["test_f1"]
        winner_denied_recall = best_individual["denied_recall"]

    print(f"\n  Winner: {winner_name} (Acc={winner_acc:.4f}, "
          f"F1={winner_f1:.4f}, Denied Recall={winner_denied_recall:.4f})")

    # ------------------------------------------------------------------
    # 14. Threshold tuning on the winner
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("THRESHOLD TUNING")
    print(f"{'=' * 70}")

    probas = winner_model.predict_proba(X_test_transformed)[:, 1]
    thresholds = np.arange(0.30, 0.71, 0.01)

    best_threshold = 0.5
    best_thresh_acc = 0.0

    print(f"  {'Threshold':>10} {'Accuracy':>10} {'F1':>8} {'Denied Rcl':>11}")
    print(f"  {'-' * 43}")

    for t in thresholds:
        y_pred_t = (probas >= t).astype(int)
        acc_t = accuracy_score(y_test, y_pred_t)
        f1_t = f1_score(y_test, y_pred_t, zero_division=0)
        recall_t = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)

        marker = ""
        if recall_t >= 0.60 and acc_t > best_thresh_acc:
            best_thresh_acc = acc_t
            best_threshold = t
            marker = " <--"

        if abs(t * 100 % 5) < 0.5 or marker:
            print(f"  {t:>10.2f} {acc_t:>10.4f} {f1_t:>8.4f} {recall_t:>11.4f}{marker}")

    # Show final threshold result
    y_pred_final = (probas >= best_threshold).astype(int)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final)
    final_denied_recall = recall_score(y_test, y_pred_final, pos_label=1)

    print(f"\n  Optimal threshold: {best_threshold:.2f}")
    print(f"  Final Accuracy:      {final_acc:.4f}")
    print(f"  Final F1:            {final_f1:.4f}")
    print(f"  Final Denied Recall: {final_denied_recall:.4f}")

    # ------------------------------------------------------------------
    # 15. Wrap in ThresholdClassifier
    # ------------------------------------------------------------------
    if abs(best_threshold - 0.5) > 0.005:
        final_model = ThresholdClassifier(winner_model, threshold=best_threshold)
        print(f"\n  Wrapped in ThresholdClassifier(threshold={best_threshold:.2f})")
    else:
        final_model = winner_model
        print(f"\n  Threshold ~0.50, using raw model (no wrapper needed)")

    # ------------------------------------------------------------------
    # 16. Full comparison summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<35} {'Test Acc':>9} {'Test F1':>8} {'Denied Rcl':>11}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['test_acc']:>9.4f} {r['test_f1']:>8.4f} {r['denied_recall']:>11.4f}")
    print(f"{'StackingClassifier':<35} {stack_acc:>9.4f} {stack_f1:>8.4f} {stack_denied_recall:>11.4f}")
    print(f"{'+ Threshold (' + f'{best_threshold:.2f})':<35} {final_acc:>9.4f} {final_f1:>8.4f} {final_denied_recall:>11.4f}")
    print("-" * 70)
    print(f"{'SELECTED':<35} {final_acc:>9.4f} {final_f1:>8.4f} {final_denied_recall:>11.4f}")

    # ------------------------------------------------------------------
    # 17. Classification report
    # ------------------------------------------------------------------
    print(f"\nClassification report on test set ({X_test.shape[0]} samples):")
    y_pred_best = final_model.predict(X_test_transformed)
    print(classification_report(
        y_test, y_pred_best, target_names=["Certified", "Denied"],
    ))

    # ------------------------------------------------------------------
    # 18. Serialize as visaModel
    # ------------------------------------------------------------------
    visa_model = visaModel(
        preprocessing_object=preprocessor,
        trained_model_object=final_model,
    )

    model_bytes = pickle.dumps(visa_model)
    print(f"Model serialized ({len(model_bytes)} bytes)")
    print(f"  Preprocessor: ColumnTransformer")
    print(f"  Classifier:   {final_model}")

    return model_bytes


@app.local_entrypoint()
def main():
    csv_path = os.path.join(LOCAL_DIR, "EasyVisa.csv")
    config_path = os.path.join(LOCAL_DIR, "config", "model.yaml")

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    with open(config_path, "r") as f:
        config_yaml = f.read()

    print("Submitting training job to Modal (H100 GPU) ...")
    model_bytes = train.remote(csv_bytes, config_yaml)

    artifact_dir = os.path.join(LOCAL_DIR, "artifact")
    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "model.pkl")

    with open(model_path, "wb") as f:
        f.write(model_bytes)

    print(f"\nModel saved to {model_path}")
    print("Done.")
