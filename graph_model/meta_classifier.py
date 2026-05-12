"""
Fold-wise meta-classifier for graph-model predictions.

This script uses fold-specific graph predictions:
    results/graph_model/fold_0/predictions.csv
    ...
    results/graph_model/fold_9/predictions.csv

For each meta fold:
    train = predictions from the other 9 folds
    test  = predictions from the held-out fold

Outputs:
    results/graph_model/meta/fold_x/meta_model.pkl
    results/graph_model/meta/fold_x/predictions.csv
    results/graph_model/meta/oof_meta_predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_fold_predictions(
    graph_result_dir: str,
    n_folds: int = 10,
) -> dict[int, pd.DataFrame]:
    """
    Load graph-model predictions from each fold.
    """

    graph_result_dir = Path(graph_result_dir)

    fold_tables = {}

    for fold in range(n_folds):

        pred_path = graph_result_dir / f"fold_{fold}" / "predictions.csv"

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing fold prediction file: {pred_path}")

        df = pd.read_csv(pred_path)

        required_cols = {
            "patient_id",
            "true_label",
            "predicted_label",
            "p0",
            "p1",
        }

        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"{pred_path} missing columns: {missing}")

        df = df.copy()
        df["fold"] = fold

        fold_tables[fold] = df

    return fold_tables


def add_uncertainty_features(
    df: pd.DataFrame,
    low_conf_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Construct uncertainty features from binary class probabilities.

    Required input columns:
        p0, p1
    """

    df = df.copy()

    probs = df[["p0", "p1"]].to_numpy(dtype=np.float32)

    p_max = probs.max(axis=1)
    p_min = probs.min(axis=1)

    prob_clip = np.clip(probs, 1e-12, 1.0)

    entropy = -np.sum(
        prob_clip * np.log2(prob_clip),
        axis=1,
    )

    df["max_prob"] = p_max
    df["conf_gap"] = p_max - p_min
    df["prob_entropy"] = entropy
    df["low_confidence"] = (p_max < low_conf_threshold).astype(int)
    df["p1_margin"] = df["p1"] - 0.5
    df["odds_p1"] = df["p1"] / (df["p0"] + 1e-6)

    return df


def get_feature_columns(
    use_uncertainty_features: bool = True,
) -> list[str]:
    """
    Meta-classifier feature columns.
    """

    base_cols = [
        "p0",
        "p1",
    ]

    uncertainty_cols = [
        "max_prob",
        "conf_gap",
        "prob_entropy",
        "low_confidence",
        "p1_margin",
        "odds_p1",
    ]

    if use_uncertainty_features:
        return base_cols + uncertainty_cols

    return base_cols


def build_meta_model(
    model_type: str = "logistic",
    random_state: int = 42,
):
    """
    Build fold-specific meta-classifier.
    """

    if model_type == "logistic":

        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ])

    if model_type == "random_forest":

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("model_type must be 'logistic' or 'random_forest'.")


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Evaluate binary predictions.
    """

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = np.nan

    return metrics


def train_one_meta_fold(
    fold: int,
    fold_tables: dict[int, pd.DataFrame],
    output_dir: str,
    n_folds: int = 10,
    model_type: str = "logistic",
    use_uncertainty_features: bool = True,
    low_conf_threshold: float = 0.6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train one fold-specific meta-classifier.

    train = predictions from all folds except current fold
    test  = predictions from current fold
    """

    train_dfs = [
        fold_tables[i]
        for i in range(n_folds)
        if i != fold
    ]

    test_df = fold_tables[fold].copy()

    train_df = pd.concat(
        train_dfs,
        axis=0,
        ignore_index=True,
    )

    train_df = add_uncertainty_features(
        train_df,
        low_conf_threshold=low_conf_threshold,
    )

    test_df = add_uncertainty_features(
        test_df,
        low_conf_threshold=low_conf_threshold,
    )

    feature_cols = get_feature_columns(
        use_uncertainty_features=use_uncertainty_features,
    )

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["true_label"].to_numpy(dtype=int)

    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["true_label"].to_numpy(dtype=int)

    model = build_meta_model(
        model_type=model_type,
        random_state=random_state,
    )

    model.fit(
        x_train,
        y_train,
    )

    if hasattr(model, "predict_proba"):
        p1 = model.predict_proba(x_test)[:, 1]
    else:
        raise RuntimeError("Meta model must support predict_proba().")

    pred_label = (p1 >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        "patient_id": test_df["patient_id"].to_numpy(),
        "true_label": y_test,
        "graph_p0": test_df["p0"].to_numpy(),
        "graph_p1": test_df["p1"].to_numpy(),
        "meta_p0": 1.0 - p1,
        "meta_p1": p1,
        "predicted_label": pred_label,
        "fold": fold,
    })

    metrics = evaluate_predictions(
        y_true=y_test,
        y_prob=p1,
    )

    fold_dir = Path(output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "low_conf_threshold": low_conf_threshold,
            "model_type": model_type,
        },
        fold_dir / "meta_model.pkl",
    )

    pred_df.to_csv(
        fold_dir / "predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        fold_dir / "metrics.csv",
        index=False,
    )

    print(
        f"Meta fold {fold} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"Macro-F1={metrics['macro_f1']:.4f} | "
        f"AUROC={metrics['auroc']:.4f}"
    )

    return pred_df


def run_meta_cross_validation(args):
    """
    Run fold-wise meta-classifier training.
    """

    fold_tables = load_fold_predictions(
        graph_result_dir=args.graph_result_dir,
        n_folds=args.n_folds,
    )

    all_predictions = []

    for fold in range(args.n_folds):

        print("\n" + "=" * 70)
        print(f"Training meta-classifier fold {fold}")

        pred_df = train_one_meta_fold(
            fold=fold,
            fold_tables=fold_tables,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            model_type=args.model_type,
            use_uncertainty_features=args.use_uncertainty_features,
            low_conf_threshold=args.low_conf_threshold,
            random_state=args.seed,
        )

        all_predictions.append(pred_df)

    oof_df = pd.concat(
        all_predictions,
        axis=0,
        ignore_index=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df.to_csv(
        output_dir / "oof_meta_predictions.csv",
        index=False,
    )

    y_true = oof_df["true_label"].to_numpy(dtype=int)
    y_prob = oof_df["meta_p1"].to_numpy(dtype=float)

    summary = evaluate_predictions(
        y_true=y_true,
        y_prob=y_prob,
    )

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        output_dir / "meta_cv_summary.csv",
        index=False,
    )

    print("\nMeta-classifier cross-validation summary")
    print(summary_df)

    return oof_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph_result_dir",
        type=str,
        default="results/graph_model",
        help="Directory containing fold_0 ... fold_9 graph predictions.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/graph_model/meta",
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
    )

    parser.add_argument(
        "--low_conf_threshold",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--use_uncertainty_features",
        action="store_true",
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    run_meta_cross_validation(args)