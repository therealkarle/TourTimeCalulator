"""Estimate duration, energy consumption, and speed for a planned tour."""

import argparse
import sys

from src.predictor import predict_tour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=["ride", "run"], help="Sport type")
    parser.add_argument("--distance-km", type=float, help="Route distance in kilometres")
    parser.add_argument("--elevation-m", type=float, help="Elevation gain in metres")
    args = parser.parse_args()

    sport = args.sport or input("Sport (ride/run) [ride]: ").strip().lower() or "ride"
    try:
        distance = args.distance_km
        if distance is None:
            distance = float(input("Route distance in km: "))
        elevation = args.elevation_m
        if elevation is None:
            elevation = float(input("Elevation gain in m: "))
        result = predict_tour(sport, distance, elevation)
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("\n================ RESULT ================")
    print(f"Sport:             {result['sport_type'].capitalize()}")
    print(f"Distance / Elev:   {result['distance_km']} km | {result['elevation_m']} m")
    print(f"Estimated Time:    {result['predicted_time']}")
    print(f"Estimated Energy:  {result['predicted_kcal']} kcal")
    print(f"Avg Speed:         {result['avg_speed_kmh']} km/h")


if __name__ == "__main__":
    main()
