"""Download new Strava activities and keep the local cache up to date."""

from src.strava_client import StravaClient


def main() -> None:
    """Synchronize Strava activities into the local SQLite database."""
    print("Synchronizing Strava activities...")
    count = StravaClient().sync_activities()
    print(f"Done. {count} activities were added or updated.")


if __name__ == "__main__":
    main()
