"""
Model training pipeline.

Trains an XGBoost classifier on the preprocessed IRS dataset to predict
the probability that a nonprofit will have its exempt status revoked.

Key design choices
------------------
- XGBoost with native categorical support (no one-hot encoding needed)
- Stratified 5-fold cross-validation for robust evaluation
- scale_pos_weight to handle severe class imbalance (~3% revoked)
- SHAP values computed post-training for explainability
- Model artefacts serialised to models/ as joblib + JSON metadata
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[2] / "models"

MODEL_PATH    = MODELS_DIR / "risk_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load preprocessed features and labels."""
    X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    y = pd.read_parquet(PROCESSED_DIR / "labels.parquet").squeeze()
    return X, y


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Return an XGBoost classifier configured for this task."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        enable_categorical=True,            # native categorical support
        tree_method="hist",                 # fast histogram method
        eval_metric="aucpr",                # area under PR curve
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


def evaluate(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    5-fold stratified cross-validation evaluation.
    Returns a dict of key metrics.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc  = average_precision_score(y, y_prob)
    report  = classification_report(y, y_pred, output_dict=True)

    print("\n── Cross-validation results ──────────────────────────────")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  PR-AUC         : {pr_auc:.4f}  (main metric — imbalanced data)")
    print(f"  Precision (1)  : {report['1']['precision']:.4f}")
    print(f"  Recall    (1)  : {report['1']['recall']:.4f}")
    print(f"  F1        (1)  : {report['1']['f1-score']:.4f}")
    print("──────────────────────────────────────────────────────────\n")

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_revoked": round(report["1"]["precision"], 4),
        "recall_revoked": round(report["1"]["recall"], 4),
        "f1_revoked": round(report["1"]["f1-score"], 4),
    }


def compute_shap_importance(
    model: XGBClassifier, X: pd.DataFrame, sample_size: int = 5_000
) -> dict[str, float]:
    """
    Compute mean |SHAP| feature importance on a random sample.
    Returns a dict {feature_name: mean_abs_shap}.
    """
    sample = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    importance = dict(
        zip(X.columns, np.abs(shap_values).mean(axis=0))
    )
    return {k: round(float(v), 6) for k, v in sorted(importance.items(), key=lambda x: -x[1])}


def train(sample_frac: float = 1.0) -> XGBClassifier:
    """
    Full training pipeline.

    Parameters
    ----------
    sample_frac : float
        Fraction of data to use (default 1.0 = all). Use < 1.0 for quick dev runs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data()

    if sample_frac < 1.0:
        idx = X.sample(frac=sample_frac, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]
        print(f"Using {sample_frac*100:.0f}% sample: {len(X):,} rows")

    # Class imbalance ratio
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass distribution: {pos:,} revoked ({pos/len(y)*100:.2f}%) | {neg:,} active")
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # Evaluate with cross-validation first
    print("\nRunning 5-fold CV evaluation...")
    model_cv = build_model(scale_pos_weight)
    metrics = evaluate(model_cv, X, y)

    # Train final model on full data
    print("Training final model on full dataset...")
    model = build_model(scale_pos_weight)

    # Split a small eval set for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # SHAP importance
    print("\nComputing SHAP feature importance...")
    shap_importance = compute_shap_importance(model, X)
    print("Top 5 features by mean |SHAP|:")
    for feat, val in list(shap_importance.items())[:5]:
        print(f"  {feat:<35} {val:.6f}")

    # Persist artefacts
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "n_train": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "cv_metrics": metrics,
        "shap_importance": shap_importance,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "xgb_best_iteration": int(model.best_iteration),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to  : {MODEL_PATH}")
    print(f"Metadata saved  : {METADATA_PATH}")
    return model


if __name__ == "__main__":
    train()
