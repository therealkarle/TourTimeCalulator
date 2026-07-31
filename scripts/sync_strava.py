"""Download new Strava activities and keep the local cache up to date."""

import sys
from pathlib import Path

# Also support: python scripts/sync_strava.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strava_client import StravaClient


def main() -> None:
    """Synchronize Strava activities into the local SQLite database."""
    print("Synchronizing Strava activities...")
    count = StravaClient().sync_activities()
    print(f"Done. {count} activities were added or updated.")


if __name__ == "__main__":
    main()
