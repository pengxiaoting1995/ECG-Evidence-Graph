"""
Train evidence-graph classifier with fixed patient-level cross-validation.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import PatientGraphDataset, collate_graphs
from model import EvidenceGraphTransformer, label_smoothing, soft_cross_entropy


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, smoothing: float):
    model.train()

    losses = []

    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_index = batch["batch"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        logits = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_index,
        )

        if smoothing > 0:
            targets = label_smoothing(y, classes=2, smoothing=smoothing)
            loss = soft_cross_entropy(logits, targets)
        else:
            loss = F.cross_entropy(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    patient_ids = []
    y_true = []
    y_prob = []

    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_index = batch["batch"].to(device)

        logits = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_index,
        )

        probs = torch.softmax(logits, dim=1)[:, 1]

        patient_ids.extend(batch["patient_id"])
        y_true.extend(batch["y"].numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = np.nan

    pred_df = pd.DataFrame({
        "patient_id": patient_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "p0": 1.0 - y_prob,
        "p1": y_prob,
    })

    return metrics, pred_df


def train_one_fold(args, fold: int, device: str):
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PatientGraphDataset(
        graph_root=args.graph_root,
        labels_path=args.labels_path,
        folds_path=args.folds_path,
        fold=fold,
        mode="train",
    )

    test_dataset = PatientGraphDataset(
        graph_root=args.graph_root,
        labels_path=args.labels_path,
        folds_path=args.folds_path,
        fold=fold,
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    model = EvidenceGraphTransformer(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_layers=args.num_layers,
        num_neighbours=args.num_neighbours,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logs = []
    best_f1 = -1.0
    best_pred_df = None

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            smoothing=args.label_smoothing,
        )

        metrics, pred_df = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }

        logs.append(row)

        print(
            f"Fold {fold} | Epoch {epoch:03d} | "
            f"Loss={train_loss:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_pred_df = pred_df.copy()

            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                fold_dir / "checkpoint.pt",
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(fold_dir / "train_log.csv", index=False)

    best_pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    return best_pred_df


def run_cross_validation(args, device: str):
    all_predictions = []

    for fold in range(args.n_folds):
        print("\n" + "=" * 70)
        print(f"Training fold {fold}")

        pred_df = train_one_fold(
            args=args,
            fold=fold,
            device=device,
        )

        pred_df["fold"] = fold
        all_predictions.append(pred_df)

    oof_df = pd.concat(all_predictions, axis=0, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    y_true = oof_df["true_label"].to_numpy()
    y_pred = oof_df["predicted_label"].to_numpy()
    y_prob = oof_df["p1"].to_numpy()

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    try:
        summary["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        summary["auroc"] = np.nan

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)

    print("\nCross-validation summary")
    print(summary_df)

    return oof_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_root", type=str, default="sample/graphs")
    parser.add_argument("--labels_path", type=str, default="sample/labels.txt")
    parser.add_argument("--folds_path", type=str, default="sample/splits/folds.csv")
    parser.add_argument("--output_dir", type=str, default="results/graph_model")

    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbours", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_cross_validation(
        args=args,
        device=device,
    )