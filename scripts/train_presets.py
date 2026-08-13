"""Train preset models with dynamic 365-day lookback window."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PRESET_DIR = BASE_DIR / "presets"

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


def load_saved_presets() -> dict[str, dict]:
    """Load custom preset definitions from the presets directory."""
    saved = {}
    if not PRESET_DIR.exists():
        return saved
    for path in sorted(PRESET_DIR.glob("*.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not config.get("sport_type"):
                print(f"Warning: Skipping invalid preset '{path.name}'.")
                continue
            for key in ("start_date", "end_date"):
                if config.get(key):
                    config[key] = date.fromisoformat(config[key])
            saved[path.stem] = config
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Warning: Could not load preset '{path.name}': {exc}")
    return saved


def save_preset(name: str, config: dict) -> None:
    """Save a preset definition as JSON."""
    preset_id = "_".join(name.split())
    preset_id = "".join(c for c in preset_id if c.isalnum() or c in "_-")
    if not preset_id:
        raise ValueError("Preset name must contain at least one letter or number.")
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    serializable = dict(config)
    for key in ("start_date", "end_date"):
        if isinstance(serializable.get(key), date):
            serializable[key] = serializable[key].isoformat()
    (PRESET_DIR / f"{preset_id}.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )


def refresh_saved_presets() -> None:
    """Add saved presets without replacing built-in presets."""
    for name, config in load_saved_presets().items():
        if name not in PRESETS:
            PRESETS[name] = config


def train_all_presets(dry_run: bool = False) -> None:
    """Train all preset models."""
    refresh_saved_presets()
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
        print(f"Error: {exc}")
        return
    if dry_run:
        print(f"[DRY RUN] Would train '{name}' with {config}.")
        return
    save_preset(name, config)
    if name not in PRESETS:
        PRESETS[name] = config.copy()
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
            print("Please enter a non-negative number or leave blank.")


def _optional_date(prompt: str) -> date | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError:
            print("Please enter YYYY-MM-DD or leave blank.")


def _optional_bool(prompt: str) -> bool | None:
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return None
        if raw in {"j", "ja", "y", "yes"}:
            return True
        if raw in {"n", "nein", "no"}:
            return False
        print("Please enter y, n, or leave blank.")


def _elevation_mode() -> str:
    """Ask user whether to count only uphill or uphill and downhill separately."""
    while True:
        choice = input("Elevation filter mode (up/separate) [up = uphill only, separate = uphill + downhill]: ").strip().lower()
        if not choice or choice == "up":
            return "up"
        if choice == "separate":
            return "separate"
        print("Please enter 'up' or 'separate', or leave blank for 'up'.")


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
            print("Please enter HH:MM or HH:MM:SS, or leave blank.")


def _custom_preset() -> tuple[str, dict]:
    print("\nCreate custom preset (blank = no limit)")
    sport_type = input("Sport profile (ride/mtb-ride/gravel/hike/run/trail_run): ").strip().lower()
    if not sport_type:
        raise ValueError("Sport profile cannot be blank.")
    raw_types = input("Activity types (optional, comma-separated): ").strip()
    activity_types = [item.strip() for item in raw_types.split(",") if item.strip()] or None
    
    config = {
        "sport_type": sport_type,
        "activity_types": activity_types,
        "min_distance_km": _optional_number("Min. distance (km): "),
        "max_distance_km": _optional_number("Max. distance (km): "),
    }
    
    elevation_mode = _elevation_mode()
    config["elevation_mode"] = elevation_mode
    
    if elevation_mode == "separate":
        config["min_elevation_up_m"] = _optional_number("Min. uphill elevation (m): ")
        config["max_elevation_up_m"] = _optional_number("Max. uphill elevation (m): ")
        config["min_elevation_down_m"] = _optional_number("Min. downhill elevation (m): ")
        config["max_elevation_down_m"] = _optional_number("Max. downhill elevation (m): ")
    else:
        config["min_elevation_m"] = _optional_number("Min. elevation (m): ")
        config["max_elevation_m"] = _optional_number("Max. elevation (m): ")
    
    config["min_moving_time_s"] = _time_to_seconds("Min. moving time (HH:MM[:SS]): ")
    config["max_moving_time_s"] = _time_to_seconds("Max. moving time (HH:MM[:SS]): ")
    config["min_elapsed_time_s"] = _time_to_seconds("Min. elapsed time (HH:MM[:SS]): ")
    config["max_elapsed_time_s"] = _time_to_seconds("Max. elapsed time (HH:MM[:SS]): ")
    config["power_data"] = _optional_bool("Require power data? (y/n/blank): ")
    config["heart_rate_data"] = _optional_bool("Require heart-rate data? (y/n/blank): ")
    config["start_date"] = _optional_date("Start date (YYYY-MM-DD): ")
    config["end_date"] = _optional_date("End date (YYYY-MM-DD): ")
    
    if elevation_mode == "separate":
        for lower, upper in (("min_elevation_up_m", "max_elevation_up_m"), ("min_elevation_down_m", "max_elevation_down_m")):
            if config.get(lower) is not None and config.get(upper) is not None and config[lower] > config[upper]:
                raise ValueError(f"{lower} cannot be greater than {upper}.")
    else:
        min_elev = config.get("min_elevation_m")
        max_elev = config.get("max_elevation_m")
        if min_elev is not None and max_elev is not None and min_elev > max_elev:
            raise ValueError("min_elevation_m cannot be greater than max_elevation_m.")
    
    for lower, upper in (("min_distance_km", "max_distance_km"), ("min_moving_time_s", "max_moving_time_s"), ("min_elapsed_time_s", "max_elapsed_time_s")):
        if config[lower] is not None and config[upper] is not None and config[lower] > config[upper]:
            raise ValueError(f"{lower} cannot be greater than {upper}.")
    
    if config["start_date"] and config["end_date"] and config["start_date"] > config["end_date"]:
        raise ValueError("Start date cannot be after end date.")
    name = input("Preset/model name: ").strip()
    if not name:
        raise ValueError("Name cannot be blank.")
    return name, config


def train_interactive(dry_run: bool = False) -> None:
    refresh_saved_presets()
    presets_list = sorted(PRESETS)
    for index, preset in enumerate(presets_list, 1):
        print(f"{index}. {preset}")
    custom_option = len(presets_list) + 1
    all_option = custom_option + 1
    exit_option = all_option + 1
    print(f"{custom_option}. Create custom preset")
    print(f"{all_option}. Train all presets")
    print(f"{exit_option}. Exit")
    while True:
        try:
            choice = int(input("\nSelection: ").strip())
        except ValueError:
            print("Please enter a number.")
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
        print(f"Please enter a number between 1 and {exit_option}.")


def main() -> None:
    """CLI entry point for training presets."""
    import argparse

    refresh_saved_presets()
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
