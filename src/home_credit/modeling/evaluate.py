"""Evaluation metrics, plots, threshold analysis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from home_credit.utils import OUTPUT_DIR, get_logger

log = get_logger(__name__)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 30,
    title: str = "Feature Importance (Gain)",
    save_path: Path | None = None,
) -> None:
    """Plot top N features by importance."""
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_imps = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_imps[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUT_DIR / "feature_importance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Feature importance plot saved to {save_path}")


def threshold_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Analyze precision, recall, F1 at various probability cutoffs."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    for t in thresholds:
        predicted_pos = (y_pred >= t).astype(int)
        tp = ((predicted_pos == 1) & (y_true == 1)).sum()
        fp = ((predicted_pos == 1) & (y_true == 0)).sum()
        fn = ((predicted_pos == 0) & (y_true == 1)).sum()
        tn = ((predicted_pos == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        approval_rate = (predicted_pos == 0).sum() / len(predicted_pos)
        decline_rate = (predicted_pos == 1).sum() / len(predicted_pos)

        rows.append({
            "threshold": round(t, 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "approval_rate": round(approval_rate, 4),
            "decline_rate": round(decline_rate, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "threshold_analysis.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(thresholds, [r["precision"] for r in rows], label="Precision", marker="o", ms=4)
    ax.plot(thresholds, [r["recall"] for r in rows], label="Recall", marker="s", ms=4)
    ax.plot(thresholds, [r["f1"] for r in rows], label="F1", marker="^", ms=4)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, [r["approval_rate"] for r in rows], label="Approval Rate", marker="o", ms=4)
    ax.plot(thresholds, [r["decline_rate"] for r in rows], label="Decline Rate", marker="s", ms=4)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Estimated Approval / Decline Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "threshold_analysis.png", dpi=150)
    plt.close(fig)
    log.info(f"Threshold analysis saved to {save_dir}")

    return df


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = OUTPUT_DIR / "roc_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved to {save_path}")


def run_shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list[str],
    save_dir: Path | None = None,
    max_samples: int = 5000,
) -> None:
    """Run SHAP analysis and save summary plot."""
    import shap

    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    log.info(f"Computing SHAP values for {X_sample.shape[0]} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info(f"SHAP summary plot saved to {save_dir / 'shap_summary.png'}")

    # SHAP feature importance (mean |SHAP|)
    shap_imp = np.abs(shap_values).mean(axis=0)
    plot_feature_importance(
        shap_imp, feature_names, top_n=30,
        title="Feature Importance (Mean |SHAP|)",
        save_path=save_dir / "shap_importance.png",
    )
