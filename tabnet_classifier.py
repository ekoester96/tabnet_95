import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import teradatasql
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

from db_config import db

DATABASE_NAME = ""
VIEW_NAME = ""

ID_COLUMN = ""
DATE_COLUMN = ""
TARGET_COLUMN = ""

EXCLUDE_COLUMNS = [

]

CONTINUOUS_COLUMNS = [
    "",
    "",
]

FLAG_COLUMNS = [

]

COUNT_COLUMNS = [

]

CATEGORICAL_COLUMNS = [

]

# Data loading settings
CHUNK_SIZE = 100000
NUM_THREADS = 4

N_CV_FOLDS = 5

OUTPUT_DIR = "tabnet_unplanned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

@dataclass
class TabNetHyperparameters:
    # --- Architecture ---
    n_d: int = 32
    n_a: int = 32
    n_steps: int = 3
    gamma: float = 1.5
    n_independent: int = 2
    n_shared: int = 2
    mask_type: str = "sparsemax"

    # --- Categorical Embeddings ---
    cat_emb_dim: int = 8

    # --- Regularization ---
    lambda_sparse: float = 1e-3
    clip_value: Optional[float] = 2.0
    momentum: float = 0.2

    # --- Probability Clipping ---
    # Applied at prediction time: p_smoothed = p * (1 - smoothing) + smoothing * 0.05
    label_smoothing: float = 0.05

    # --- Training ---
    learning_rate: float = 2e-2
    max_epochs: int = 20
    patience: int = 5
    batch_size: int = 4096
    virtual_batch_size: int = 512

    # --- Learning Rate Scheduler ---
    scheduler_type: str = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    # --- Class Weights ---
    use_class_weights: bool = True

    # --- Fine-tuning ---
    finetune_lr_factor: float = 0.1  # Fine-tune LR = learning_rate * this

    # --- Reproducibility ---
    seed: int = 42


HP_TABNET = TabNetHyperparameters()


def get_teradata_connection():
    return teradatasql.connect(
        host=db.get_host(),
        user=db.get_username(),
        password=db.get_password(),
        database=DATABASE_NAME,
    )


def get_row_count() -> int:
    query = f"SELECT COUNT(*) FROM {DATABASE_NAME}.{VIEW_NAME}"
    with get_teradata_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            count = cursor.fetchone()[0]
    print(f"Total rows: {count:,}")
    return count


def fetch_chunk(offset: int, chunk_size: int) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {DATABASE_NAME}.{VIEW_NAME}
        QUALIFY ROW_NUMBER() OVER (ORDER BY {ID_COLUMN} ASC)
            BETWEEN {offset + 1} AND {offset + chunk_size}
    """
    with get_teradata_connection() as conn:
        return pd.read_sql(query, conn)


def load_data() -> pd.DataFrame:
    total_rows = get_row_count()
    offsets = list(range(0, total_rows, CHUNK_SIZE))
    print(f"Loading in {len(offsets)} chunks using {NUM_THREADS} threads...")

    chunk_dict = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_offset = {
            executor.submit(fetch_chunk, offset, CHUNK_SIZE): offset
            for offset in offsets
        }
        for future in as_completed(future_to_offset):
            offset = future_to_offset[future]
            chunk_df = future.result()
            chunk_dict[offset] = chunk_df
            completed += 1
            print(f"  Chunk {completed}/{len(offsets)} ({len(chunk_df):,} rows)")

    ordered = [chunk_dict[o] for o in sorted(chunk_dict.keys())]
    df = pd.concat(ordered, ignore_index=True)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return df


def prepare_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns, excluding ID, DATE, TARGET, and EXCLUDE_COLUMNS."""
    exclude = set([ID_COLUMN, DATE_COLUMN, TARGET_COLUMN] + EXCLUDE_COLUMNS)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], List[int], List[int]]:
    """
    Label encode categorical columns for TabNet embeddings.
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    cat_idxs = []
    cat_dims = []

    feature_cols = prepare_feature_columns(df)

    for col in categorical_cols:
        if col not in df.columns:
            print(f"Warning: Categorical column '{col}' not found in DataFrame")
            continue

        df[col] = df[col].fillna("__MISSING__").astype(str)

        if fit:
            le = LabelEncoder()

            unique_vals = df[col].unique().tolist()

            if "__MISSING__" not in unique_vals:
                unique_vals.append("__MISSING__")

            if "__UNKNOWN__" not in unique_vals:
                unique_vals.append("__UNKNOWN__")

            le.fit(unique_vals)
            encoders[col] = le
        else:
            le = encoders[col]

            known_classes = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_classes else "__UNKNOWN__"
            )

        df[col] = le.transform(df[col])

        # Safety: clip to valid embedding range (prevents CUDA assert errors)
        max_valid_idx = len(le.classes_) - 1
        df[col] = df[col].clip(upper=max_valid_idx)

        if col in feature_cols:
            cat_idxs.append(feature_cols.index(col))
            cat_dims.append(len(le.classes_))

    return df, encoders, cat_idxs, cat_dims


def scale_features(
    df: pd.DataFrame,
    continuous_cols: List[str],
    count_cols: List[str],
    continuous_scaler: StandardScaler = None,
    count_scaler: MinMaxScaler = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler, MinMaxScaler]:
    """
    Scale continuous columns with StandardScaler and count columns with MinMaxScaler.
    """
    df = df.copy()

    cont_cols_exist = [c for c in continuous_cols if c in df.columns]
    count_cols_exist = [c for c in count_cols if c in df.columns]

    if cont_cols_exist:
        if fit:
            continuous_scaler = StandardScaler()
            df[cont_cols_exist] = continuous_scaler.fit_transform(
                df[cont_cols_exist].fillna(0).values
            )
        else:
            df[cont_cols_exist] = continuous_scaler.transform(
                df[cont_cols_exist].fillna(0).values
            )

    if count_cols_exist:
        if fit:
            count_scaler = MinMaxScaler()
            df[count_cols_exist] = count_scaler.fit_transform(
                df[count_cols_exist].fillna(0).values
            )
        else:
            df[count_cols_exist] = count_scaler.transform(
                df[count_cols_exist].fillna(0).values
            )

    return df, continuous_scaler, count_scaler


def apply_label_smoothing(y_prob: np.ndarray, smoothing: float) -> np.ndarray:
    """
    Clip predicted probabilities toward 0.5 to reduce overconfidence.

    p_smoothed = p * (1 - smoothing) + smoothing * 0.5

    With smoothing=0.05: a raw 0.99 becomes 0.965, a raw 0.01 becomes 0.035.
    """
    if smoothing <= 0:
        return y_prob
    return y_prob * (1.0 - smoothing) + smoothing * 0.5


def temporal_train_test_split(
    df: pd.DataFrame,
    test_months: int = 2,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last N months for test, everything else for train+val.
    Train+val will be shuffled during K-fold CV; test stays in temporal order.
    """
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    max_date = df[DATE_COLUMN].max()
    cutoff_date = max_date - pd.DateOffset(months=test_months)

    df_train_val = df[df[DATE_COLUMN] < cutoff_date].copy()
    df_test = df[df[DATE_COLUMN] >= cutoff_date].copy()

    # Shuffle train+val (test stays sorted by date)
    df_train_val = df_train_val.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test = df_test.sort_values(DATE_COLUMN).reset_index(drop=True)

    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Train+Val set: {len(df_train_val):,} rows (shuffled, before {cutoff_date.date()})")
    print(f"Test set: {len(df_test):,} rows (temporal, {cutoff_date.date()} to {max_date.date()})")
    print(f"Test spans {df_test[DATE_COLUMN].dt.date.nunique()} unique days")

    train_val_dist = df_train_val[TARGET_COLUMN].value_counts().sort_index()
    test_dist = df_test[TARGET_COLUMN].value_counts().sort_index()
    print(f"Train+Val class distribution: {dict(train_val_dist)}")
    print(f"Test class distribution: {dict(test_dist)}")

    return df_train_val, df_test


def get_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = SEED
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate standard K-fold CV splits (randomized, non-temporal).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        splits.append((train_idx, val_idx))
        print(f"Fold {fold + 1}: Train [{len(train_idx):,}], Val [{len(val_idx):,}]")

    return splits


def get_fold_class_weights(
    fold_idx: int,
    current_y_train: np.ndarray,
    previous_y_train: np.ndarray = None
) -> Union[int, Dict[int, float]]:
    """
    Get class weights for current fold based on lagged distribution strategy.

    Fold 1: Use 50/50 balanced weights
    Fold 2+: Use class distribution from previous fold's training data
    """
    if fold_idx == 0:
        print("  Fold 1: Using 50/50 balanced class weights")
        return 1
    else:
        if previous_y_train is not None:
            class_counts = np.bincount(previous_y_train.astype(int))
            total = len(previous_y_train)

            if len(class_counts) >= 2:
                weight_0 = total / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0
                weight_1 = total / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0

                print(f"  Fold {fold_idx + 1}: Using weights from previous fold - "
                      f"Class 0: {weight_0:.3f}, Class 1: {weight_1:.3f}")
                print(f"    Previous fold distribution: Class 0: {class_counts[0]:,} "
                      f"({class_counts[0]/total*100:.1f}%), "
                      f"Class 1: {class_counts[1]:,} ({class_counts[1]/total*100:.1f}%)")

                return {0: weight_0, 1: weight_1}

        return 1


def build_tabnet(cat_idxs: List[int] = None, cat_dims: List[int] = None) -> TabNetClassifier:
    """Build TabNetClassifier with categorical embeddings."""
    hp = HP_TABNET

    if hp.scheduler_type == "step":
        scheduler_fn = torch.optim.lr_scheduler.StepLR
        scheduler_params = {"step_size": hp.scheduler_step_size, "gamma": hp.scheduler_gamma}
    elif hp.scheduler_type == "cosine":
        scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_params = {"T_max": hp.max_epochs, "eta_min": 1e-6}
    else:
        scheduler_fn = None
        scheduler_params = {}

    return TabNetClassifier(
        n_d=hp.n_d,
        n_a=hp.n_a,
        n_steps=hp.n_steps,
        gamma=hp.gamma,
        n_independent=hp.n_independent,
        n_shared=hp.n_shared,
        mask_type=hp.mask_type,
        lambda_sparse=hp.lambda_sparse,
        clip_value=hp.clip_value,
        momentum=hp.momentum,
        cat_idxs=cat_idxs if cat_idxs else [],
        cat_dims=cat_dims if cat_dims else [],
        cat_emb_dim=hp.cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": hp.learning_rate},
        scheduler_fn=scheduler_fn,
        scheduler_params=scheduler_params,
        seed=hp.seed,
        verbose=1,
        device_name=DEVICE,
    )


def train_tabnet(
    model: TabNetClassifier,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    weights: Union[int, Dict[int, float]] = 1
) -> TabNetClassifier:
    """Train TabNet with built-in early stopping."""
    hp = HP_TABNET

    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        eval_set=[(X_va, y_va)],
        eval_name=["val"],
        eval_metric=["logloss"],
        max_epochs=hp.max_epochs,
        patience=hp.patience,
        batch_size=hp.batch_size,
        virtual_batch_size=hp.virtual_batch_size,
        drop_last=False,
        weights=weights,
    )

    return model


def fine_tune_tabnet(
    model: TabNetClassifier,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    epochs: int = 5,
    batch_size: int = HP_TABNET.batch_size,
    virtual_batch_size: int = HP_TABNET.virtual_batch_size,
) -> TabNetClassifier:
    """
    Fine-tune an already-trained TabNet on newly revealed data.
    Uses warm_start=True to continue from existing weights with a reduced LR.
    
    Note: eval set is the same as training set here since we have no holdout
    from the revealed data. The eval loss is logged but not used for early stopping
    (patience is set beyond max epochs).
    """
    hp = HP_TABNET
    fine_tune_lr = hp.learning_rate * hp.finetune_lr_factor

    # Update optimizer LR before warm-start fit (TabNet reinitializes the
    # optimizer on each fit call, reading from this attribute)
    model.optimizer_params = {"lr": fine_tune_lr}

    model.fit(
        X_train=X_finetune,
        y_train=y_finetune,
        eval_set=[(X_finetune, y_finetune)],
        eval_name=["finetune"],
        eval_metric=["logloss"],
        max_epochs=epochs,
        patience=epochs + 1,  # Don't early stop during fine-tuning
        batch_size=min(batch_size, len(X_finetune)),
        virtual_batch_size=min(virtual_batch_size, len(X_finetune)),
        drop_last=False,
        warm_start=True,
    )

    return model


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fold_name: str = "Fold"
) -> Dict:
    """Compute and print evaluation metrics."""

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = np.nan

    try:
        avg_precision = average_precision_score(y_true, y_prob)
    except ValueError:
        avg_precision = np.nan

    report = classification_report(y_true, y_pred, output_dict=True)
    report_str = classification_report(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{fold_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
    print(f"\nClassification Report:\n{report_str}")
    print(f"\nConfusion Matrix:\n{cm}")

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_daily_metrics(
    daily_results: List[Dict],
    title: str,
    save_path: str
):
    """Plot per-day metrics over the test period with fine-tune markers."""
    dates = [r["date"] for r in daily_results]
    n_samples = [r["n_samples"] for r in daily_results]

    # Filter to days with enough samples for meaningful metrics
    valid = [r for r in daily_results if r["n_samples"] >= 5 and not np.isnan(r["roc_auc"])]

    if not valid:
        print(f"Warning: No days with enough samples for daily metrics plot.")
        return

    valid_dates = [r["date"] for r in valid]
    roc_aucs = [r["roc_auc"] for r in valid]
    avg_precs = [r["avg_precision"] for r in valid]
    accuracies = [r["accuracy"] for r in valid]
    finetune_flags = [r.get("finetuned_before", False) for r in valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Mark fine-tune points
    finetune_dates = [d for d, f in zip(valid_dates, finetune_flags) if f]

    # ROC-AUC
    ax = axes[0]
    ax.plot(valid_dates, roc_aucs, "b-o", markersize=4, label="ROC-AUC")
    ax.axhline(y=np.mean(roc_aucs), color="red", linestyle="--", alpha=0.5,
               label=f"Mean: {np.mean(roc_aucs):.4f}")
    for fd in finetune_dates:
        ax.axvline(x=fd, color="green", alpha=0.3, linestyle=":")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average Precision
    ax = axes[1]
    ax.plot(valid_dates, avg_precs, "g-o", markersize=4, label="Avg Precision")
    ax.axhline(y=np.mean(avg_precs), color="red", linestyle="--", alpha=0.5,
               label=f"Mean: {np.mean(avg_precs):.4f}")
    for fd in finetune_dates:
        ax.axvline(x=fd, color="green", alpha=0.3, linestyle=":")
    ax.set_ylabel("Average Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample count per day
    ax = axes[2]
    ax.bar(dates, n_samples, color="steelblue", alpha=0.7, label="Shipments per Day")
    for fd in finetune_dates:
        ax.axvline(x=fd, color="green", alpha=0.3, linestyle=":",
                   label="Weekly Fine-tune" if fd == finetune_dates[0] else "")
    ax.set_ylabel("Number of Shipments")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved daily metrics plot: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_feature_importance(
    model: TabNetClassifier,
    feature_names: List[str],
    title: str,
    save_path: str,
    top_n: int = 30
):
    """Plot TabNet feature importance from attention masks."""
    if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        importances = model.feature_importances_
    else:
        print("  Warning: feature_importances_ not available, skipping feature importance plot.")
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))

    bars = ax.barh(
        range(len(top_features)),
        top_features["importance"].values,
        color="steelblue",
        edgecolor="navy",
        alpha=0.8
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Attention Weight)")
    ax.set_title(title)

    for i, (bar, val) in enumerate(zip(bars, top_features["importance"].values)):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance: {save_path}")

    return importance_df


def plot_feature_masks(
    model: TabNetClassifier,
    X_sample: np.ndarray,
    feature_names: List[str],
    title: str,
    save_path: str,
    n_samples: int = 100
):
    """Visualize TabNet's step-wise attention masks."""
    try:
        explain_matrix, masks = model.explain(X_sample[:n_samples])

        if masks is None or not hasattr(masks, '__iter__'):
            print(f"Warning: TabNet masks not available in expected format. Plotting aggregate only.")
            masks = []

        valid_masks = []
        if isinstance(masks, (list, tuple)):
            for m in masks:
                if isinstance(m, np.ndarray) and m.ndim >= 1:
                    valid_masks.append(m)
        elif isinstance(masks, dict):
            for key, m in masks.items():
                if isinstance(m, np.ndarray) and m.ndim >= 1:
                    valid_masks.append(m)

        n_steps = len(valid_masks)
        n_plots = n_steps + 1

        if n_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 8))
            if n_plots == 1:
                axes = [axes]

        for step, mask in enumerate(valid_masks):
            ax = axes[step]

            if mask.ndim == 2:
                avg_mask = mask.mean(axis=0)
            else:
                avg_mask = mask

            n_features_to_show = min(20, len(avg_mask))
            sorted_idx = np.argsort(avg_mask)[::-1][:n_features_to_show]

            ax.barh(range(len(sorted_idx)), avg_mask[sorted_idx], color=f"C{step}")
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"Feature {i}"
                               for i in sorted_idx])
            ax.invert_yaxis()
            ax.set_title(f"Step {step + 1}")
            ax.set_xlabel("Attention")

        ax = axes[-1]

        if explain_matrix.ndim == 2:
            avg_explain = explain_matrix.mean(axis=0)
        else:
            avg_explain = explain_matrix

        n_features_to_show = min(20, len(avg_explain))
        sorted_idx = np.argsort(avg_explain)[::-1][:n_features_to_show]

        ax.barh(range(len(sorted_idx)), avg_explain[sorted_idx], color="darkgreen")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"Feature {i}"
                           for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_title("Aggregate Importance")
        ax.set_xlabel("Importance")

        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved feature masks visualization: {save_path}")

    except Exception as e:
        print(f"Warning: Could not plot feature masks - {e}")
        plt.close()

        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            importances = model.feature_importances_

            n_features_to_show = min(20, len(importances))
            sorted_idx = np.argsort(importances)[::-1][:n_features_to_show]

            ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="darkgreen")
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"Feature {i}"
                               for i in sorted_idx])
            ax.invert_yaxis()
            ax.set_title(f"{title} (Fallback - Aggregate Only)")
            ax.set_xlabel("Importance")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved feature importance (fallback): {save_path}")
        except Exception as e2:
            print(f"Warning: Fallback plotting also failed - {e2}")


def plot_individual_row_mask(
    model: TabNetClassifier,
    x_row: np.ndarray,
    feature_names: List[str],
    title: str,
    save_path: str,
    top_n: int = 15,
):
    """
    Plot step-wise TabNet attention masks for a single input row.
    Each step is shown as a separate subplot side-by-side, plus an aggregate.
    """
    try:
        x_2d = x_row.reshape(1, -1)
        explain_matrix, masks = model.explain(x_2d)

        valid_masks = []
        if isinstance(masks, (list, tuple)):
            for m in masks:
                if isinstance(m, np.ndarray) and m.ndim >= 1:
                    valid_masks.append(m.flatten())
        elif isinstance(masks, dict):
            for m in masks.values():
                if isinstance(m, np.ndarray) and m.ndim >= 1:
                    valid_masks.append(m.flatten())

        n_steps = len(valid_masks)
        n_plots = n_steps + 1  # steps + aggregate

        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        colors = plt.cm.tab10.colors

        for step_idx, (mask, ax) in enumerate(zip(valid_masks, axes[:n_steps])):
            n_show = min(top_n, len(mask))
            sorted_idx = np.argsort(mask)[::-1][:n_show]
            values = mask[sorted_idx]
            labels = [
                feature_names[i] if i < len(feature_names) else f"Feature {i}"
                for i in sorted_idx
            ]
            ax.barh(range(n_show), values, color=colors[step_idx % len(colors)], alpha=0.8)
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"Step {step_idx + 1}", fontsize=10)
            ax.set_xlabel("Attention", fontsize=8)
            ax.grid(True, alpha=0.2)

        # Aggregate subplot
        ax = axes[-1]
        agg = explain_matrix.flatten()
        n_show = min(top_n, len(agg))
        sorted_idx = np.argsort(agg)[::-1][:n_show]
        values = agg[sorted_idx]
        labels = [
            feature_names[i] if i < len(feature_names) else f"Feature {i}"
            for i in sorted_idx
        ]
        ax.barh(range(n_show), values, color="darkgreen", alpha=0.8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Aggregate", fontsize=10)
        ax.set_xlabel("Importance", fontsize=8)
        ax.grid(True, alpha=0.2)

        plt.suptitle(title, fontsize=11, y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  Warning: Could not plot individual row mask - {e}")
        plt.close()


def plot_individual_row_masks_batch(
    model: TabNetClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    label: str = "fold_1",
    n_samples: int = 5,
    top_n: int = 15,
    seed: int = SEED,
):
    """
    Randomly sample n_samples rows and save an individual mask plot for each.
    Saved to output_dir/individual_masks/<label>/row_<i>_idx<original_index>.png
    """
    save_dir = os.path.join(output_dir, "individual_masks", label)
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    n_available = len(X)
    n_to_plot = min(n_samples, n_available)
    sampled_indices = rng.choice(n_available, size=n_to_plot, replace=False)

    print(f"  Saving {n_to_plot} individual mask plots to: {save_dir}")

    for i, idx in enumerate(sampled_indices):
        x_row = X[idx]
        true_label = int(y_true[idx])
        prob = float(y_prob[idx])

        title = (
            f"{label} | Sample {i+1} (dataset index {idx}) | "
            f"True: {true_label} | Pred prob: {prob:.3f}"
        )
        save_path = os.path.join(save_dir, f"row_{i+1}_idx{idx}.png")

        plot_individual_row_mask(
            model=model,
            x_row=x_row,
            feature_names=feature_names,
            title=title,
            save_path=save_path,
            top_n=top_n,
        )

    print(f"  Done saving individual masks for: {label}")


def plot_training_history(
    model: TabNetClassifier,
    title: str,
    save_path: str
):
    """Plot training and validation loss over epochs."""
    try:
        history_dict = model.history.history if hasattr(model.history, 'history') else dict(model.history)

        fig, ax = plt.subplots(figsize=(10, 6))

        if "loss" in history_dict:
            epochs = range(1, len(history_dict["loss"]) + 1)
            ax.plot(epochs, history_dict["loss"], "b-", label="Training Loss", linewidth=2)

        if "val_0_logloss" in history_dict:
            ax.plot(epochs, history_dict["val_0_logloss"], "r-", label="Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved training history: {save_path}")

    except Exception as e:
        print(f"Warning: Could not plot training history - {e}")
        plt.close()


def plot_cv_comparison(
    fold_metrics: List[Dict],
    title: str,
    save_path: str
):
    """Plot comparison of metrics across K folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    folds = range(1, len(fold_metrics) + 1)
    roc_aucs = [m["roc_auc"] for m in fold_metrics]
    avg_precs = [m["avg_precision"] for m in fold_metrics]

    # ROC-AUC plot
    ax = axes[0]
    bars = ax.bar(folds, roc_aucs, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(y=np.mean(roc_aucs), color="red", linestyle="--",
               label=f"Mean: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC by Fold")
    ax.legend()
    ax.set_ylim([min(roc_aucs) - 0.05, max(roc_aucs) + 0.05])

    for bar, val in zip(bars, roc_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Average Precision plot
    ax = axes[1]
    bars = ax.bar(folds, avg_precs, color="forestgreen", edgecolor="darkgreen", alpha=0.8)
    ax.axhline(y=np.mean(avg_precs), color="red", linestyle="--",
               label=f"Mean: {np.mean(avg_precs):.4f} ± {np.std(avg_precs):.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Average Precision")
    ax.set_title("Average Precision by Fold")
    ax.legend()
    ax.set_ylim([min(avg_precs) - 0.05, max(avg_precs) + 0.05])

    for bar, val in zip(bars, avg_precs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CV comparison plot: {save_path}")

# MAIN TRAINING PIPELINE

def run_kfold_cv_training(
    df_train_val: pd.DataFrame,
    cat_cols: List[str],
    cont_cols: List[str],
    count_cols: List[str],
    flag_cols: List[str],
) -> Tuple[List[TabNetClassifier], List[Dict], np.ndarray, Dict]:
    """
    Run K-fold cross-validation training with proper preprocessing per fold.

    flag_cols is accepted for documentation but not explicitly transformed.
    Binary 0/1 flags pass through as unscaled float features, which is the
    correct behavior — no scaling or encoding needed.

    Returns:
        models: List of trained models (one per fold)
        fold_metrics: List of evaluation metrics per fold
        oof_probabilities: Out-of-fold probability predictions (one per train+val sample)
        final_artifacts: Preprocessing artifacts refit on full train+val
    """
    smoothing = HP_TABNET.label_smoothing

    # Generate K-fold CV splits (randomized)
    cv_splits = get_kfold_splits(df_train_val, n_folds=N_CV_FOLDS, seed=SEED)

    models = []
    fold_metrics = []
    previous_y_train = None

    # OOF predictions — one probability per train+val sample
    oof_probabilities = np.zeros(len(df_train_val))

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold_idx + 1}/{N_CV_FOLDS} (Randomized K-Fold)")
        print(f"{'#'*60}")

        # Split raw data
        df_train_fold = df_train_val.iloc[train_idx].copy()
        df_val_fold = df_train_val.iloc[val_idx].copy()

        print(f"Training samples: {len(df_train_fold):,}")
        print(f"Validation samples: {len(df_val_fold):,}")

        # FIT preprocessing on training fold ONLY, then transform both

        # 1. Encode categoricals - fit on train, transform both
        df_train_fold, encoders, cat_idxs, cat_dims = encode_categoricals(
            df_train_fold, cat_cols, encoders=None, fit=True
        )
        df_val_fold, _, _, _ = encode_categoricals(
            df_val_fold, cat_cols, encoders=encoders, fit=False
        )

        # 2. Scale features - fit on train, transform both
        df_train_fold, cont_scaler, count_scaler = scale_features(
            df_train_fold, cont_cols, count_cols,
            continuous_scaler=None, count_scaler=None, fit=True
        )
        df_val_fold, _, _ = scale_features(
            df_val_fold, cont_cols, count_cols,
            continuous_scaler=cont_scaler, count_scaler=count_scaler, fit=False
        )

        # Get feature columns
        feature_cols = prepare_feature_columns(df_train_fold)

        # Prepare arrays for training

        X_train = df_train_fold[feature_cols].values.astype(np.float32)
        y_train = df_train_fold[TARGET_COLUMN].values.astype(np.int64)
        X_val = df_val_fold[feature_cols].values.astype(np.float32)
        y_val = df_val_fold[TARGET_COLUMN].values.astype(np.int64)

        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Validation class distribution: {np.bincount(y_val)}")
        print(f"Number of features: {len(feature_cols)}")
        print(f"Categorical indices: {cat_idxs}, dims: {cat_dims}")

        weights = get_fold_class_weights(fold_idx, y_train, previous_y_train)

        # Store current training labels for next fold
        previous_y_train = y_train.copy()

        # Build and train model
        model = build_tabnet(cat_idxs=cat_idxs, cat_dims=cat_dims)
        model = train_tabnet(model, X_train, y_train, X_val, y_val, weights=weights)

        # Get predictions (with label smoothing)
        y_pred = model.predict(X_val)
        y_prob_raw = model.predict_proba(X_val)[:, 1]
        y_prob = apply_label_smoothing(y_prob_raw, smoothing)

        # Store OOF predictions
        oof_probabilities[val_idx] = y_prob

        # Evaluate
        metrics = evaluate_model(y_val, y_pred, y_prob, fold_name=f"Fold {fold_idx + 1}")
        fold_metrics.append(metrics)

        # Save fold artifacts
        plot_confusion_matrix(
            y_val, y_pred,
            title=f"Fold {fold_idx + 1} Confusion Matrix (Randomized)",
            save_path=os.path.join(OUTPUT_DIR, f"fold_{fold_idx + 1}_confusion_matrix.png")
        )

        plot_training_history(
            model,
            title=f"Fold {fold_idx + 1} Training History",
            save_path=os.path.join(OUTPUT_DIR, f"fold_{fold_idx + 1}_training_history.png")
        )

        plot_feature_importance(
            model, feature_cols,
            title=f"Fold {fold_idx + 1} Feature Importance",
            save_path=os.path.join(OUTPUT_DIR, f"fold_{fold_idx + 1}_feature_importance.png")
        )
        
        # Individual row mask plots for this validation fold
        plot_individual_row_masks_batch(
            model=model,
            X=X_val,
            y_true=y_val,
            y_prob=y_prob,
            feature_names=feature_cols,
            output_dir=OUTPUT_DIR,
            label=f"fold_{fold_idx + 1}",
            n_samples=5,
            top_n=15,
            seed=SEED,
        )

        # Save model
        model.save_model(os.path.join(OUTPUT_DIR, f"fold_{fold_idx + 1}_model"))

        models.append(model)

    # OOF evaluation

    y_all = df_train_val[TARGET_COLUMN].values.astype(np.int64)
    oof_preds = (oof_probabilities >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("OUT-OF-FOLD (OOF) EVALUATION")
    print(f"{'='*60}")
    oof_metrics = evaluate_model(y_all, oof_preds, oof_probabilities, fold_name="OOF")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        ID_COLUMN: df_train_val[ID_COLUMN].values,
        "y_true": y_all,
        "oof_prob": oof_probabilities,
        "oof_pred": oof_preds,
    })
    oof_df.to_csv(os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
    print(f"Saved OOF predictions to: {os.path.join(OUTPUT_DIR, 'oof_predictions.csv')}")

    # Refit preprocessing on FULL train+val for test set usage

    print(f"\nRefitting preprocessing on full train+val ({len(df_train_val):,} samples)...")

    df_train_val_processed = df_train_val.copy()

    df_train_val_processed, final_encoders, final_cat_idxs, final_cat_dims = encode_categoricals(
        df_train_val_processed, cat_cols, encoders=None, fit=True
    )
    df_train_val_processed, final_cont_scaler, final_count_scaler = scale_features(
        df_train_val_processed, cont_cols, count_cols,
        continuous_scaler=None, count_scaler=None, fit=True
    )
    final_feature_cols = prepare_feature_columns(df_train_val_processed)

    final_artifacts = {
        "encoders": final_encoders,
        "cont_scaler": final_cont_scaler,
        "count_scaler": final_count_scaler,
        "cat_idxs": final_cat_idxs,
        "cat_dims": final_cat_dims,
        "feature_cols": final_feature_cols,
    }

    # Summary of CV results
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    roc_aucs = [m["roc_auc"] for m in fold_metrics]
    avg_precs = [m["avg_precision"] for m in fold_metrics]

    print(f"ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"Avg Precision: {np.mean(avg_precs):.4f} ± {np.std(avg_precs):.4f}")
    print(f"OOF ROC-AUC: {oof_metrics['roc_auc']:.4f}")
    print(f"OOF Avg Precision: {oof_metrics['avg_precision']:.4f}")

    for i, m in enumerate(fold_metrics):
        print(f"  Fold {i+1}: ROC-AUC={m['roc_auc']:.4f}, AP={m['avg_precision']:.4f}")

    # Plot CV comparison
    plot_cv_comparison(
        fold_metrics,
        title="K-Fold Cross-Validation Results (Randomized)",
        save_path=os.path.join(OUTPUT_DIR, "cv_comparison.png")
    )

    return models, fold_metrics, oof_probabilities, final_artifacts


def evaluate_on_test_set(
    models: List[TabNetClassifier],
    df_test: pd.DataFrame,
    preprocessing_artifacts: Dict,
    fold_metrics: List[Dict],
    finetune_epochs: int = 5,
    finetune_every_n_days: int = 7,
) -> Dict:
    """
    Day-by-day evaluation on temporal holdout test set.

    - Predicts each day's shipments using only the current model
    - After every 7 days, fine-tunes the model on all revealed test data so far
    - Uses best single fold model as the starting point
    - Preprocessing artifacts are fitted on the full train+val set
    """
    smoothing = HP_TABNET.label_smoothing

    encoders = preprocessing_artifacts["encoders"]
    cont_scaler = preprocessing_artifacts["cont_scaler"]
    count_scaler = preprocessing_artifacts["count_scaler"]
    feature_cols = preprocessing_artifacts["feature_cols"]
    cat_idxs = preprocessing_artifacts["cat_idxs"]
    cat_dims = preprocessing_artifacts["cat_dims"]

    print(f"\n{'#'*60}")
    print("# DAY-BY-DAY TEST EVALUATION (Temporal Holdout)")
    print(f"# Fine-tuning every {finetune_every_n_days} days on revealed data")
    print(f"# Label smoothing: {smoothing}")
    print(f"{'#'*60}")

    # Select best single fold model
    best_model_idx = np.argmax([m["roc_auc"] for m in fold_metrics])
    model = models[best_model_idx]
    print(f"Starting from best fold model: Fold {best_model_idx + 1}")

    # Preprocess entire test set (transform only, no fitting)
    df_test_processed = df_test.copy()
    df_test_processed[DATE_COLUMN] = pd.to_datetime(df_test_processed[DATE_COLUMN])

    df_test_processed, _, _, _ = encode_categoricals(
        df_test_processed, CATEGORICAL_COLUMNS, encoders=encoders, fit=False
    )
    df_test_processed, _, _ = scale_features(
        df_test_processed, CONTINUOUS_COLUMNS, COUNT_COLUMNS,
        continuous_scaler=cont_scaler, count_scaler=count_scaler, fit=False
    )

    X_test_all = df_test_processed[feature_cols].values.astype(np.float32)
    y_test_all = df_test_processed[TARGET_COLUMN].values.astype(np.int64)
    test_dates = df_test_processed[DATE_COLUMN].dt.date

    unique_days = sorted(test_dates.unique())
    print(f"Test period: {unique_days[0]} to {unique_days[-1]} ({len(unique_days)} days)")
    print(f"Total test samples: {len(X_test_all):,}")

    # Day-by-day prediction loop
    daily_results = []
    all_preds = np.zeros(len(X_test_all))
    all_probs = np.zeros(len(X_test_all))

    revealed_X = []
    revealed_y = []
    days_since_finetune = 0
    finetune_count = 0
    finetuned_this_round = False

    # Validate categorical indices are within embedding bounds
    for i, (idx, dim) in enumerate(zip(cat_idxs, cat_dims)):
        max_val = int(X_test_all[:, idx].max())
        if max_val >= dim:
            print(f"  ERROR: Feature col '{feature_cols[idx]}' has max index {max_val} "
                  f"but embedding dim is {dim}. Clipping.")
            X_test_all[:, idx] = np.clip(X_test_all[:, idx], 0, dim - 1)

    for day_idx, day in enumerate(unique_days):
        day_mask = (test_dates == day).values
        X_day = X_test_all[day_mask]
        y_day = y_test_all[day_mask]

        # Predict this day's shipments (model has NOT seen these labels)
        y_prob_raw = model.predict_proba(X_day)[:, 1]
        y_prob_day = apply_label_smoothing(y_prob_raw, smoothing)
        y_pred_day = (y_prob_day >= 0.5).astype(int)

        # Store predictions
        all_probs[day_mask] = y_prob_day
        all_preds[day_mask] = y_pred_day

        # Compute daily metrics
        day_result = {
            "date": day,
            "day_idx": day_idx,
            "n_samples": len(y_day),
            "n_positive": int(y_day.sum()),
            "finetuned_before": finetuned_this_round,
        }

        if len(np.unique(y_day)) >= 2 and len(y_day) >= 2:
            day_result["roc_auc"] = roc_auc_score(y_day, y_prob_day)
            day_result["avg_precision"] = average_precision_score(y_day, y_prob_day)
        else:
            day_result["roc_auc"] = np.nan
            day_result["avg_precision"] = np.nan

        day_result["accuracy"] = (y_pred_day == y_day).mean()

        daily_results.append(day_result)

        auc_str = f"AUC={day_result['roc_auc']:.4f}" if not np.isnan(day_result.get("roc_auc", np.nan)) else "AUC=N/A (single class)"
        print(f"  Day {day_idx + 1}/{len(unique_days)} ({day}): "
              f"{len(y_day)} shipments, Acc={day_result['accuracy']:.3f}, {auc_str}")

        # Reveal labels — add to revealed pool
        revealed_X.append(X_day)
        revealed_y.append(y_day)

        days_since_finetune += 1
        finetuned_this_round = False

        # Fine-tune every N days
        if days_since_finetune >= finetune_every_n_days and day_idx < len(unique_days) - 1:
            X_revealed = np.vstack(revealed_X)
            y_revealed = np.concatenate(revealed_y)

            finetune_count += 1
            print(f"\n  >>> Weekly fine-tune #{finetune_count} after day {day} "
                  f"({len(y_revealed):,} revealed samples)")

            model = fine_tune_tabnet(
                model, X_revealed, y_revealed, epochs=finetune_epochs
            )

            days_since_finetune = 0
            finetuned_this_round = True
            print(f"  >>> Fine-tune complete\n")

    # Overall test metrics
    all_preds_int = all_preds.astype(int)

    print(f"\n{'='*60}")
    print("OVERALL TEST RESULTS (Day-by-Day Prediction)")
    print(f"{'='*60}")
    metrics_overall = evaluate_model(
        y_test_all, all_preds_int, all_probs, fold_name="Test Set (Day-by-Day)"
    )

    # Daily metrics summary
    daily_df = pd.DataFrame(daily_results)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_metrics.csv"), index=False)
    print(f"\nSaved daily metrics to: {os.path.join(OUTPUT_DIR, 'daily_metrics.csv')}")

    valid_days = daily_df.dropna(subset=["roc_auc"])
    if len(valid_days) > 0:
        print(f"\nDaily ROC-AUC: mean={valid_days['roc_auc'].mean():.4f}, "
              f"std={valid_days['roc_auc'].std():.4f}, "
              f"min={valid_days['roc_auc'].min():.4f}, "
              f"max={valid_days['roc_auc'].max():.4f}")
        print(f"Daily Avg Precision: mean={valid_days['avg_precision'].mean():.4f}, "
              f"std={valid_days['avg_precision'].std():.4f}")

    print(f"Fine-tune events: {finetune_count}")

    plot_confusion_matrix(
        y_test_all, all_preds_int,
        title="Test Set Confusion Matrix (Day-by-Day, Weekly Fine-tune)",
        save_path=os.path.join(OUTPUT_DIR, "test_confusion_matrix.png")
    )

    plot_daily_metrics(
        daily_results,
        title="Day-by-Day Test Metrics (Fine-tuned Weekly)",
        save_path=os.path.join(OUTPUT_DIR, "daily_metrics_plot.png")
    )

    plot_feature_importance(
        model, feature_cols,
        title="Final Model Feature Importance (After Fine-tuning)",
        save_path=os.path.join(OUTPUT_DIR, "final_feature_importance.png"),
        top_n=40
    )
    
        # Individual row mask plots for test set
    plot_individual_row_masks_batch(
        model=model,
        X=X_test_all,
        y_true=y_test_all,
        y_prob=all_probs,
        feature_names=feature_cols,
        output_dir=OUTPUT_DIR,
        label="test",
        n_samples=5,
        top_n=15,
        seed=SEED,
    )
    

    plot_feature_masks(
        model, X_test_all, feature_cols,
        title="TabNet Attention Masks (Test Set)",
        save_path=os.path.join(OUTPUT_DIR, "final_feature_masks.png"),
        n_samples=min(100, len(X_test_all))
    )

    # Save predictions
    test_predictions_df = pd.DataFrame({
        ID_COLUMN: df_test[ID_COLUMN].values,
        DATE_COLUMN: df_test[DATE_COLUMN].values,
        "y_true": y_test_all,
        "y_pred": all_preds_int,
        "y_prob": all_probs,
    })
    test_predictions_df.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False
    )
    print(f"Saved test predictions to: {os.path.join(OUTPUT_DIR, 'test_predictions.csv')}")

    return {
        "overall": metrics_overall,
        "daily_results": daily_results,
        "daily_df": daily_df,
        "finetune_count": finetune_count,
        "predictions_df": test_predictions_df,
    }


def main():
    """Main training pipeline — Shuffled K-Fold CV + Temporal Day-by-Day Test."""
    print("="*60)
    print("TABNET CLASSIFIER TRAINING PIPELINE")
    print("MODE: SHUFFLED K-FOLD CV + TEMPORAL DAY-BY-DAY TEST")
    print("="*60)
    print("\nTrain+Val: Randomized K-Fold (all data before last 2 months)")
    print("Test: Day-by-day prediction on last 2 months, fine-tuned weekly")
    print("="*60)

    # Load data
    print("\n[1/5] Loading data from Teradata...")
    df = load_data()

    # Temporal split — last 2 months for test
    print("\n[2/5] Performing temporal train/test split (last 2 months = test)...")
    df_train_val, df_test = temporal_train_test_split(df, test_months=2, seed=SEED)

    # Run K-fold CV training on train+val (shuffled)
    print("\n[3/5] Running randomized K-fold cross-validation on train+val...")
    models, fold_metrics, oof_probabilities, final_artifacts = run_kfold_cv_training(
        df_train_val,
        cat_cols=CATEGORICAL_COLUMNS,
        cont_cols=CONTINUOUS_COLUMNS,
        count_cols=COUNT_COLUMNS,
        flag_cols=FLAG_COLUMNS,
    )

    # Save preprocessing artifacts (fitted on full train+val)
    print("\n[4/5] Saving preprocessing artifacts (fitted on full train+val)...")
    joblib.dump(final_artifacts["encoders"], os.path.join(OUTPUT_DIR, "label_encoders.joblib"))
    joblib.dump(final_artifacts["cont_scaler"], os.path.join(OUTPUT_DIR, "continuous_scaler.joblib"))
    joblib.dump(final_artifacts["count_scaler"], os.path.join(OUTPUT_DIR, "count_scaler.joblib"))
    joblib.dump(final_artifacts["feature_cols"], os.path.join(OUTPUT_DIR, "feature_columns.joblib"))
    joblib.dump(
        {"cat_idxs": final_artifacts["cat_idxs"], "cat_dims": final_artifacts["cat_dims"]},
        os.path.join(OUTPUT_DIR, "categorical_info.joblib")
    )
    print("Saved preprocessing artifacts (fitted on full train+val)")

    # Day-by-day evaluation with weekly fine-tuning
    print("\n[5/5] Day-by-day evaluation on temporal test set...")
    test_results = evaluate_on_test_set(
        models, df_test, final_artifacts, fold_metrics,
        finetune_epochs=5, finetune_every_n_days=7
    )

    # Save summary
    summary = {
        "experiment_type": "shuffled_kfold_temporal_test",
        "n_folds": N_CV_FOLDS,
        "train_val_size": len(df_train_val),
        "test_size": len(df_test),
        "test_days": len(test_results["daily_results"]),
        "finetune_events": test_results["finetune_count"],
        "n_features": len(final_artifacts["feature_cols"]),
        "label_smoothing": HP_TABNET.label_smoothing,
        "cv_roc_auc_mean": np.mean([m["roc_auc"] for m in fold_metrics]),
        "cv_roc_auc_std": np.std([m["roc_auc"] for m in fold_metrics]),
        "cv_avg_precision_mean": np.mean([m["avg_precision"] for m in fold_metrics]),
        "cv_avg_precision_std": np.std([m["avg_precision"] for m in fold_metrics]),
        "oof_roc_auc": roc_auc_score(
            df_train_val[TARGET_COLUMN].values, oof_probabilities
        ),
        "test_roc_auc": test_results["overall"]["roc_auc"],
        "test_avg_precision": test_results["overall"]["avg_precision"],
    }

    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "training_summary.csv"), index=False)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"\nKey files:")
    print(f"  - training_summary.csv: Overall metrics summary")
    print(f"  - oof_predictions.csv: Out-of-fold predictions from CV")
    print(f"  - test_predictions.csv: Test set predictions with dates")
    print(f"  - daily_metrics.csv: Per-day breakdown of test metrics")
    print(f"  - daily_metrics_plot.png: Daily metrics over time")
    print(f"  - cv_comparison.png: K-fold metrics comparison")
    print(f"  - final_feature_importance.png: Feature importance chart")
    print(f"  - fold_*_model/: Saved models for each fold")

    return models, fold_metrics, oof_probabilities, test_results


if __name__ == "__main__":
    models, fold_metrics, oof_probabilities, test_results = main()
