"""Train preset models with dynamic 365-day lookback window."""

import sys
from datetime import date, timedelta
from pathlib import Path

# Support: python scripts/train_presets.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_models import train_sport


def get_last_365_days() -> tuple[date, date]:
    """Return (start_date, end_date) for the last 365 days."""
    today = date.today()
    return today - timedelta(days=365), today


# Define all 7 preset model configurations with 365d and all variants
PRESETS = {
    "mountainStage_min1500hm_max150km_ride_365d": {
        "sport_type": "ride",
        "min_distance_km": 1.0,
        "max_distance_km": 150.0,
        "min_elevation_m": 1500.0,
        "use_365d": True,
    },
    "mountainStage_min1500hm_max150km_ride_all": {
        "sport_type": "ride",
        "min_distance_km": 1.0,
        "max_distance_km": 150.0,
        "min_elevation_m": 1500.0,
        "use_365d": False,
    },
    "gravel_365d": {
        "sport_type": "gravel",
        "min_distance_km": 3.0,
        "use_365d": True,
    },
    "gravel_all": {
        "sport_type": "gravel",
        "min_distance_km": 3.0,
        "use_365d": False,
    },
    "ride_withpowerData_365d": {
        "sport_type": "ride",
        "min_distance_km": 3.0,
        "power_data": True,
        "use_365d": True,
    },
    "ride_withpowerData_all": {
        "sport_type": "ride",
        "min_distance_km": 3.0,
        "power_data": True,
        "use_365d": False,
    },
    "hike_365d": {
        "sport_type": "hike",
        "min_distance_km": 1.0,
        "use_365d": True,
    },
    "hike_all": {
        "sport_type": "hike",
        "min_distance_km": 1.0,
        "use_365d": False,
    },
    "run_365d": {
        "sport_type": "run",
        "min_distance_km": 1.0,
        "use_365d": True,
    },
    "run_all": {
        "sport_type": "run",
        "min_distance_km": 1.0,
        "use_365d": False,
    },
    "trail_run_365d": {
        "sport_type": "trail_run",
        "min_distance_km": 1.0,
        "use_365d": True,
    },
    "trail_run_all": {
        "sport_type": "trail_run",
        "min_distance_km": 1.0,
        "use_365d": False,
    },
    "ride_min100km_365d": {
        "sport_type": "ride",
        "min_distance_km": 100.0,
        "use_365d": True,
    },
    "ride_min100km_all": {
        "sport_type": "ride",
        "min_distance_km": 100.0,
        "use_365d": False,
    },
}


def train_all_presets(dry_run: bool = False) -> None:
    """Train all preset models."""
    print(f"\nTraining all presets\n")

    if dry_run:
        print("[DRY RUN] Would train the following presets:")
        for preset_name, config in PRESETS.items():
            use_365d = config.get("use_365d", True)
            window = "last 365 days" if use_365d else "all data"
            print(f"  - {preset_name} ({window})")
        return

    results = {}
    for preset_name, config in PRESETS.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {preset_name}")
        print(f"{'=' * 60}")

        # Extract and apply date range based on use_365d flag
        config = config.copy()
        use_365d = config.pop("use_365d", True)
        sport_type = config.pop("sport_type")
        
        if use_365d:
            start_date, end_date = get_last_365_days()
            config["start_date"] = start_date
            config["end_date"] = end_date
            print(f"Window: {start_date} to {end_date}")
        else:
            print("Window: All data (no date filter)")

        # Train the model
        success = train_sport(
            sport_name=sport_type,
            model_name=preset_name,
            **config
        )
        
        results[preset_name] = "✓ Success" if success else "✗ Failed (no data)"
        print(f"Result: {results[preset_name]}")

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for preset_name, result in results.items():
        print(f"{preset_name}: {result}")


def train_single_preset(preset_name: str, dry_run: bool = False) -> None:
    """Train a single preset model."""
    if preset_name not in PRESETS:
        print(f"Error: Preset '{preset_name}' not found.")
        print(f"Available presets: {', '.join(sorted(PRESETS.keys()))}")
        return

    config = PRESETS[preset_name].copy()
    use_365d = config.get("use_365d", True)
    
    print(f"\n{'=' * 60}")
    print(f"Training: {preset_name}")
    if use_365d:
        start_date, end_date = get_last_365_days()
        print(f"Window: {start_date} to {end_date}")
    else:
        print("Window: All data (no date filter)")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would train with:")
        sport_type = config.get("sport_type")
        print(f"  Sport: {sport_type}")
        for key, value in config.items():
            if key not in ["sport_type", "use_365d"]:
                print(f"  {key}: {value}")
        return

    sport_type = config.pop("sport_type")
    config.pop("use_365d")
    
    if use_365d:
        start_date, end_date = get_last_365_days()
        config["start_date"] = start_date
        config["end_date"] = end_date

    success = train_sport(
        sport_name=sport_type,
        model_name=preset_name,
        **config
    )
    
    result = "✓ Success" if success else "✗ Failed (no data)"
    print(f"\nResult: {result}")


def train_interactive(dry_run: bool = False) -> None:
    """Interactive preset selection."""
    start_date, end_date = get_last_365_days()
    print(f"\n365-day window: {start_date} to {end_date}\n")
    
    # Display available presets
    presets_list = sorted(PRESETS.keys())
    for i, preset in enumerate(presets_list, 1):
        print(f"{i}. {preset}")
    
    print(f"{len(presets_list) + 1}. Train all presets")
    print(f"{len(presets_list) + 2}. Exit")
    
    while True:
        try:
            choice = input("\nSelect preset (number): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(presets_list) + 2:
                print("Exiting.")
                return
            elif choice_num == len(presets_list) + 1:
                train_all_presets(dry_run=dry_run)
                return
            elif 1 <= choice_num <= len(presets_list):
                preset_name = presets_list[choice_num - 1]
                train_single_preset(preset_name, dry_run=dry_run)
                return
            else:
                print(f"Please enter a number between 1 and {len(presets_list) + 2}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main() -> None:
    """CLI entry point for training presets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train preset models with dynamic 365-day window"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Specific preset to train (if not specified, interactive selection)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without actually training"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all presets"
    )
    args = parser.parse_args()

    if args.all:
        train_all_presets(dry_run=args.dry_run)
    elif args.preset:
        train_single_preset(args.preset, dry_run=args.dry_run)
    else:
        train_interactive(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
