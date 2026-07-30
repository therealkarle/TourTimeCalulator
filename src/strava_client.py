"""Strava OAuth, activity synchronization, rate limiting, and SQLite caching."""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from src.config import (
    DB_PATH,
    STRAVA_API_URL,
    STRAVA_CLIENT_ID,
    STRAVA_CLIENT_SECRET,
    STRAVA_REFRESH_TOKEN,
)


class StravaClient:
    """Synchronize Strava activities into a local SQLite cache."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.client_id = STRAVA_CLIENT_ID
        self.client_secret = STRAVA_CLIENT_SECRET
        self.refresh_token = STRAVA_REFRESH_TOKEN
        self.access_token: Optional[str] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create the activities table and its useful indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                    start_date INTEGER, distance REAL, moving_time INTEGER,
                    elapsed_time INTEGER, total_elevation_gain REAL,
                    kilojoules REAL, calories REAL, workout_type INTEGER
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activities_start_date "
                "ON activities(start_date)"
            )

    def refresh_access_token(self) -> None:
        """Obtain a fresh OAuth access token from the configured refresh token."""
        if not all((self.client_id, self.client_secret, self.refresh_token)):
            raise RuntimeError("Strava credentials are missing from .env")
        response = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        response.raise_for_status()
        self.access_token = response.json()["access_token"]

    @staticmethod
    def _parse_start_date(activity: dict[str, Any]) -> int:
        """Convert Strava's ISO timestamp to a UTC Unix timestamp."""
        raw = activity.get("start_date") or activity.get("start_date_local")
        if not raw:
            raise ValueError("Activity has no start date")
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())

    @staticmethod
    def _handle_rate_limits(response: requests.Response) -> None:
        """Sleep for 15 minutes when the short-term limit reaches 90 percent."""
        try:
            usage = [int(value) for value in response.headers["X-RateLimit-Usage"].split(",")]
            limits = [int(value) for value in response.headers["X-RateLimit-Limit"].split(",")]
            if usage and limits and limits[0] > 0 and usage[0] >= limits[0] * 0.9:
                print(f"Rate limit threshold reached ({usage[0]}/{limits[0]}). Sleeping 15 minutes...")
                time.sleep(900)
        except (KeyError, TypeError, ValueError):
            return

    def sync_activities(self) -> int:
        """Fetch activities newer than the cache and return the insert count."""
        if not all((self.client_id, self.client_secret, self.refresh_token)):
            print("Strava credentials missing in .env file. Skipping API sync.")
            return 0

        self.refresh_access_token()
        with sqlite3.connect(self.db_path) as conn:
            last_ts = conn.execute("SELECT MAX(start_date) FROM activities").fetchone()[0] or 0

        page, new_records = 1, 0
        while True:
            response = requests.get(
                f"{STRAVA_API_URL}/athlete/activities",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={"after": last_ts, "page": page, "per_page": 200},
                timeout=30,
            )
            if response.status_code == 429:
                print("HTTP 429 rate limit hit. Retrying in 15 minutes...")
                time.sleep(900)
                continue
            response.raise_for_status()
            self._handle_rate_limits(response)
            activities = response.json()
            if not activities:
                break

            with sqlite3.connect(self.db_path) as conn:
                for activity in activities:
                    conn.execute(
                        """INSERT OR REPLACE INTO activities
                        (id, name, type, start_date, distance, moving_time,
                         elapsed_time, total_elevation_gain, kilojoules,
                         calories, workout_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            activity["id"], activity.get("name", ""), activity.get("type", ""),
                            self._parse_start_date(activity), activity.get("distance", 0.0),
                            activity.get("moving_time", 0), activity.get("elapsed_time", 0),
                            activity.get("total_elevation_gain", 0.0), activity.get("kilojoules"),
                            activity.get("calories"), activity.get("workout_type"),
                        ),
                    )
                    new_records += 1
            page += 1

        print(f"Sync finished: {new_records} activities cached.")
        return new_records
