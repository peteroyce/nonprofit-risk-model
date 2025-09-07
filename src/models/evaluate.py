"""
Model evaluation report generator.

Loads the trained model and produces a markdown evaluation report with
detailed metrics, confusion matrix, and threshold analysis. Outputs to
reports/evaluation.md.

Usage
-----
  python -m src.models.evaluate
  python -m src.cli evaluate
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD, MODELS_DIR

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
MODEL_PATH = MODELS_DIR / "risk_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def _load_artifacts():
    """Load model, metadata, and dataset."""
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    y = pd.read_parquet(PROCESSED_DIR / "labels.parquet").squeeze()
    return model, metadata, X, y


def _threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """Evaluate precision/recall/F1 at multiple thresholds."""
    results = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (y_prob >= threshold).astype(int)
        results.append({
            "threshold": threshold,
            "precision": round(float((y_pred[y_pred == 1] == y_true[y_pred == 1]).mean()), 4) if y_pred.sum() > 0 else 0.0,
            "recall": round(float((y_pred[y_true == 1] == 1).mean()), 4) if (y_true == 1).sum() > 0 else 0.0,
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "flagged": int(y_pred.sum()),
            "flagged_pct": round(float(y_pred.mean() * 100), 2),
        })
    return results


def generate_report() -> Path:
    """Generate a full evaluation report in markdown."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model and data...")
    model, metadata, X, y = _load_artifacts()

    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    threshold_results = _threshold_analysis(y.values, y_prob)

    # SHAP importance from metadata
    shap_importance = metadata.get("shap_importance", {})
    cv_metrics = metadata.get("cv_metrics", {})

    # Build markdown report
    lines = [
        "# Model Evaluation Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Dataset",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total samples | {len(X):,} |",
        f"| Revoked (positive) | {int(y.sum()):,} ({y.mean()*100:.2f}%) |",
        f"| Active (negative) | {int((y == 0).sum()):,} ({(1 - y.mean())*100:.2f}%) |",
        f"| Features | {X.shape[1]} |",
        "",
        "## Cross-Validation Metrics (5-fold stratified)",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| ROC-AUC | {roc_auc:.4f} |",
        f"| PR-AUC | {pr_auc:.4f} |",
        f"| Accuracy | {accuracy:.4f} |",
        f"| Precision (revoked) | {report['1']['precision']:.4f} |",
        f"| Recall (revoked) | {report['1']['recall']:.4f} |",
        f"| F1 (revoked) | {report['1']['f1-score']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "```",
        f"                Predicted Active    Predicted Revoked",
        f"Actual Active     {cm[0][0]:>10,}          {cm[0][1]:>10,}",
        f"Actual Revoked    {cm[1][0]:>10,}          {cm[1][1]:>10,}",
        "```",
        "",
        "## Threshold Analysis",
        "",
        "How metrics change at different classification thresholds:",
        "",
        "| Threshold | Precision | Recall | F1 | Flagged | % Flagged |",
        "|---|---|---|---|---|---|",
    ]

    for t in threshold_results:
        lines.append(
            f"| {t['threshold']} | {t['precision']:.4f} | {t['recall']:.4f} | "
            f"{t['f1']:.4f} | {t['flagged']:,} | {t['flagged_pct']}% |"
        )

    lines.extend([
        "",
        "## Feature Importance (SHAP)",
        "",
        "Ranked by mean |SHAP value| — higher means more influence on predictions:",
        "",
        "| Rank | Feature | Mean |SHAP| |",
        "|---|---|---|",
    ])

    for rank, (feat, val) in enumerate(shap_importance.items(), 1):
        lines.append(f"| {rank} | `{feat}` | {val:.6f} |")

    lines.extend([
        "",
        "## Risk Thresholds",
        "",
        f"- **Low risk**: score < {LOW_RISK_THRESHOLD}",
        f"- **Medium risk**: {LOW_RISK_THRESHOLD} <= score < {HIGH_RISK_THRESHOLD}",
        f"- **High risk**: score >= {HIGH_RISK_THRESHOLD}",
        "",
        "## Training Configuration",
        "",
        f"- XGBoost best iteration: {metadata.get('xgb_best_iteration', 'N/A')}",
        f"- scale_pos_weight: {metadata.get('scale_pos_weight', 'N/A')}",
        f"- n_features: {metadata.get('n_features', 'N/A')}",
        "",
    ])

    report_path = REPORTS_DIR / "evaluation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return report_path


if __name__ == "__main__":
    generate_report()


def validate_12(data):
    """Validate: add schema validation"""
    return data is not None
