"""
Train CCR-Score on beat-level ECG data.
"""

import argparse
from pathlib import Path

import torch

from model import build_model, load_pretrained_extractor
from trainer import pretrain_classifier, train_ccr
from utils import setup_seed, create_logger


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_h5",
        type=str,
        default="../../ecg-evidence-graph/ccr_score/beat_samples.h5",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ccr_score",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--rejection_cost",
        type=float,
        default=0.375,
    )

    parser.add_argument(
        "--pretrained_extractor",
        type=str,
        default=None,
        help="Optional pretrained encoder checkpoint.",
    )

    parser.add_argument(
        "--freeze_extractor",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    setup_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(output_dir)

    logger.info("Building CCR-Score model")

    model = build_model()

    if args.pretrained_extractor is not None:

        logger.info(
            f"Loading pretrained extractor: {args.pretrained_extractor}"
        )

        model = load_pretrained_extractor(
            model=model,
            checkpoint_path=args.pretrained_extractor,
            freeze=args.freeze_extractor,
        )

    logger.info("Starting classifier pretraining")

    model = pretrain_classifier(
        model=model,
        train_h5=args.train_h5,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.pretrain_epochs,
        logger=logger,
    )

    logger.info("Starting CCR training")

    model = train_ccr(
        model=model,
        train_h5=args.train_h5,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        rejection_cost=args.rejection_cost,
        logger=logger,
    )

    logger.info("Training completed")


if __name__ == "__main__":

    main()