"""Backfill detailed Strava calories in resumable batches, then retrain models."""

import argparse
import sys
from pathlib import Path

# Also support: python scripts/backfill_calories.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.update_models import update_model
from src.config import MODEL_DIR
from src.strava_client import StravaClient


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def update_all_models() -> tuple[int, int]:
    metadata_files = sorted(MODEL_DIR.glob("*.txt"))
    updated = sum(update_model(path) for path in metadata_files)
    return updated, len(metadata_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=100,
        help="activity detail requests per checkpointed batch (default: 100)",
    )
    parser.add_argument(
        "--max-batches",
        type=positive_int,
        help="stop after this many batches; rerun later to resume",
    )
    parser.add_argument(
        "--skip-model-update",
        action="store_true",
        help="only backfill calories; do not retrain models when complete",
    )
    args = parser.parse_args()

    client = StravaClient()
    total_checked = total_updated = batches = 0
    while True:
        result = client.backfill_calories(args.batch_size)
        batches += 1
        total_checked += result.checked
        total_updated += result.updated
        print(
            f"Batch {batches}: checked={result.checked}, calories={result.updated}, "
            f"remaining={result.remaining}"
        )

        if result.daily_limit_reached:
            print("Progress is saved. Rerun after Strava resets the daily limit.")
            return 2
        if result.remaining == 0:
            break
        if result.checked == 0:
            print("No progress was possible; stopping without retraining models.")
            return 1
        if args.max_batches is not None and batches >= args.max_batches:
            print("Batch limit reached. Progress is saved; rerun to continue.")
            return 0

    print(f"Calorie backfill complete: checked={total_checked}, calories={total_updated}.")
    if args.skip_model_update:
        return 0

    updated, total = update_all_models()
    print(f"Model update complete: {updated}/{total} models retrained.")
    return 0 if updated == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
