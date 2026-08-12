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


def train_custom_preset(dry_run: bool = False) -> None:
    """Create and train one custom preset."""
    try:
        name, config = _custom_preset()
    except ValueError as exc:
        print(f"Fehler: {exc}")
        return
    if dry_run:
        print(f"[DRY RUN] Würde '{name}' mit {config} trainieren.")
        return
    sport_type = config.pop("sport_type")
    success = train_sport(sport_name=sport_type, model_name=name, **config)
    print(f"\nResult: {'✓ Success' if success else '✗ Failed (no data)'}")


def _optional_number(prompt: str) -> float | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            value = float(raw)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("Bitte eine nicht-negative Zahl oder leer eingeben.")


def _optional_date(prompt: str) -> date | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError:
            print("Bitte YYYY-MM-DD oder leer eingeben.")


def _optional_bool(prompt: str) -> bool | None:
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return None
        if raw in {"j", "ja", "y", "yes"}:
            return True
        if raw in {"n", "nein", "no"}:
            return False
        print("Bitte j, n oder leer eingeben.")


def _time_to_seconds(prompt: str) -> int | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            parts = [int(part) for part in raw.split(":")]
            if len(parts) == 2:
                hours, minutes = parts
                seconds = 0
            elif len(parts) == 3:
                hours, minutes, seconds = parts
            else:
                raise ValueError
            if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
                raise ValueError
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            print("Bitte HH:MM oder HH:MM:SS eingeben, oder leer lassen.")


def _custom_preset() -> tuple[str, dict]:
    print("\nEigenes Preset erstellen (leer = keine Grenze)")
    sport_type = input("Sportprofil (ride/mtb-ride/gravel/hike/run/trail_run): ").strip().lower()
    if not sport_type:
        raise ValueError("Das Sportprofil darf nicht leer sein.")
    raw_types = input("Aktivitätstypen (optional, Komma getrennt): ").strip()
    activity_types = [item.strip() for item in raw_types.split(",") if item.strip()] or None
    config = {
        "sport_type": sport_type,
        "activity_types": activity_types,
        "min_distance_km": _optional_number("Min. km: "),
        "max_distance_km": _optional_number("Max. km: "),
        "min_elevation_m": _optional_number("Min. hm: "),
        "max_elevation_m": _optional_number("Max. hm: "),
        "min_moving_time_s": _time_to_seconds("Min. Bewegungszeit (HH:MM[:SS]): "),
        "max_moving_time_s": _time_to_seconds("Max. Bewegungszeit (HH:MM[:SS]): "),
        "min_elapsed_time_s": _time_to_seconds("Min. Elapsed-Zeit (HH:MM[:SS]): "),
        "max_elapsed_time_s": _time_to_seconds("Max. Elapsed-Zeit (HH:MM[:SS]): "),
        "power_data": _optional_bool("Powerdaten erforderlich? (j/n/leer): "),
        "heart_rate_data": _optional_bool("Herzfrequenz erforderlich? (j/n/leer): "),
        "start_date": _optional_date("Startdatum (YYYY-MM-DD): "),
        "end_date": _optional_date("Enddatum (YYYY-MM-DD): "),
    }
    for lower, upper in (("min_distance_km", "max_distance_km"), ("min_elevation_m", "max_elevation_m"), ("min_moving_time_s", "max_moving_time_s"), ("min_elapsed_time_s", "max_elapsed_time_s")):
        if config[lower] is not None and config[upper] is not None and config[lower] > config[upper]:
            raise ValueError(f"{lower} darf nicht größer als {upper} sein.")
    if config["start_date"] and config["end_date"] and config["start_date"] > config["end_date"]:
        raise ValueError("Das Startdatum darf nicht nach dem Enddatum liegen.")
    name = input("Name des Presets/Modells: ").strip()
    if not name:
        raise ValueError("Der Name darf nicht leer sein.")
    return name, config


def train_interactive(dry_run: bool = False) -> None:
    presets_list = sorted(PRESETS)
    for index, preset in enumerate(presets_list, 1):
        print(f"{index}. {preset}")
    custom_option = len(presets_list) + 1
    all_option = custom_option + 1
    exit_option = all_option + 1
    print(f"{custom_option}. Eigenes Preset erstellen")
    print(f"{all_option}. Alle Presets trainieren")
    print(f"{exit_option}. Beenden")
    while True:
        try:
            choice = int(input("\nAuswahl: ").strip())
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            continue
        if choice == custom_option:
            train_custom_preset(dry_run)
            return
        if choice == all_option:
            train_all_presets(dry_run)
            return
        if choice == exit_option:
            return
        if 1 <= choice <= len(presets_list):
            train_single_preset(presets_list[choice - 1], dry_run)
            return
        print(f"Bitte eine Zahl zwischen 1 und {exit_option} eingeben.")


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
