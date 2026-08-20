"""Estimate duration, energy consumption, and speed for a planned tour."""

import argparse
import sys
from pathlib import Path

# Also support: python scripts/estimate_tour_duration.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictor import list_models, predict_tour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Name or ID of the trained model")
    parser.add_argument("--distance-km", type=float, help="Route distance in kilometres")
    parser.add_argument("--elevation-m", type=float, help="Elevation gain in metres (legacy alias for ascent)")
    parser.add_argument("--elevation-up-m", type=float, help="Elevation ascent in metres")
    parser.add_argument("--elevation-down-m", type=float, help="Elevation descent in metres")
    parser.add_argument(
        "--break-minutes",
        type=float,
        help="Override predicted stopped time with planned break minutes",
    )
    args = parser.parse_args()

    try:
        models = list_models()
        if not models:
            raise FileNotFoundError("No trained models found. Run train_models.py first.")
        model_name = args.model
        if not model_name:
            print("Available models:")
            for index, model in enumerate(models, start=1):
                print(f"  {index}. {model['model_name']} ({model['sport_type']})")
            selected_index = int(input("Select model number: ")) - 1
            model_name = models[selected_index]["model_id"]
        selected_model = next(
            model for model in models
            if model.get("model_id") == model_name or model.get("model_name") == model_name
        )
        distance = args.distance_km
        if distance is None:
            distance = float(input("Route distance in km: "))
        separate = "elevation_down_m" in selected_model.get("features", [])
        if separate:
            elevation = args.elevation_up_m if args.elevation_up_m is not None else args.elevation_m
            if elevation is None:
                elevation = float(input("Elevation ascent in m: "))
            descent = args.elevation_down_m
            if descent is None:
                raw_descent = input("Elevation descent in m (empty = ascent): ").strip()
                descent = float(raw_descent) if raw_descent else elevation
        else:
            elevation = args.elevation_m if args.elevation_m is not None else args.elevation_up_m
            if elevation is None:
                elevation = float(input("Elevation gain in m: "))
            descent = None
        stopped_time_s = args.break_minutes * 60 if args.break_minutes is not None else None
        result = predict_tour(model_name, distance, elevation, descent, stopped_time_s)
    except (ValueError, FileNotFoundError, IndexError, KeyError, StopIteration) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("\n================ RESULT ================")
    print(f"Model:             {result['model_name']}")
    print(f"Sport:             {result['sport_type'].capitalize()}")
    print(f"Distance / Elev:   {result['distance_km']} km | +{result['elevation_up_m']} m | -{result['elevation_down_m']} m")
    print(f"Elapsed Time:      {result['predicted_time']}")
    elapsed_low, elapsed_high = result["predicted_elapsed_time_interval_sec"]
    print(
        f"Likely Range:      {_format_duration(elapsed_low)} – "
        f"{_format_duration(elapsed_high)}"
    )
    moving_hours, moving_remainder = divmod(result['predicted_moving_time_sec'], 3600)
    print(f"Moving Time:       {moving_hours}h {moving_remainder // 60:02d}m")
    stopped_hours, stopped_remainder = divmod(result['predicted_stopped_time_sec'], 3600)
    print(f"Stopped Time:      {stopped_hours}h {stopped_remainder // 60:02d}m")
    if result["predicted_kcal"] is not None:
        kcal_low, kcal_high = result["predicted_kcal_interval"]
        print(
            f"Estimated Energy:  {result['predicted_kcal']} kcal "
            f"({kcal_low}–{kcal_high})"
        )
    else:
        print("Estimated Energy:  unavailable (no kcal-trained model)")
    print(f"Avg Speed:         {result['avg_speed_kmh']} km/h")
    if result["average_power_watts"] is not None:
        print(f"Average Power:     {result['average_power_watts']} W")
    if result["weighted_average_power_watts"] is not None:
        print(f"Weighted Avg Power:{result['weighted_average_power_watts']} W")
    if result["average_hr_bpm"] is not None:
        print(f"Average HR:        {result['average_hr_bpm']} bpm")
    for warning in result["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)


def _format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    return f"{hours}h {remainder // 60:02d}m"


if __name__ == "__main__":
    main()
