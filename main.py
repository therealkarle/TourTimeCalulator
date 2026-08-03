"""Backward-compatible entry point for the complete workflow.

For independent execution, use the scripts in ``scripts/`` instead.
"""

from scripts.estimate_tour_duration import main as estimate_main
from scripts.sync_strava import main as sync_main
from scripts.train_models import train_sport


def main() -> None:
    """Synchronize data, train available models, and predict a planned tour."""
    print("========================================")
    print("       Strava Tour Predictor")
    print("========================================\n")
    sync_main()

    print("\nTraining a regression model...")
    sport = input("Sport profile (ride/mtb-ride/gravel): ").strip().lower()
    model_name = input("Model name: ").strip()
    train_sport(sport, model_name)
    estimate_main()


if __name__ == "__main__":
    main()
