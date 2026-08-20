import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import train_presets, update_models
from scripts.train_presets import get_last_365_days
from src import feature_eng, predictor
from src.feature_eng import load_cleaned_data
from src.model_trainer import MIN_TRAINING_SAMPLES, train_models_for_sport
from src.strava_client import StravaClient, ensure_db_schema


class ElevationCalculationTests(unittest.TestCase):
    def test_stationary_altitude_noise_is_not_descent(self) -> None:
        self.assertEqual(
            StravaClient._calculate_elevation_loss([100.0, 99.0, 100.0, 99.0]),
            0.0,
        )

    def test_gradual_descent_is_accumulated(self) -> None:
        descent = StravaClient._calculate_elevation_loss([100, 99, 98, 97, 96])
        self.assertAlmostEqual(descent, 4.0)

    def test_missing_altitude_remains_unknown(self) -> None:
        self.assertIsNone(StravaClient._calculate_elevation_loss(None))


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        # SQLite file release can lag briefly on Windows/Python; cleanup errors
        # must not turn successful calculation assertions into test failures.
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "activities.sqlite"
        ensure_db_schema(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _insert_activity(
        self,
        activity_id: int,
        *,
        sport_type: str = "Ride",
        calories: float | None = 500.0,
        kilojoules: float | None = 1000.0,
        descent: float | None = 900.0,
        moving_time: int = 1800,
        elapsed_time: int = 3600,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO activities
                (id, type, sport_type, start_date, distance, moving_time,
                 elapsed_time, total_elevation_gain, kilojoules, calories,
                 descent_elevation_m, commute)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity_id,
                    "Ride",
                    sport_type,
                    1_700_000_000 + activity_id,
                    10_000.0,
                    moving_time,
                    elapsed_time,
                    1_000.0,
                    kilojoules,
                    calories,
                    descent,
                    0,
                ),
            )

    def _load(self, **kwargs):
        with patch.object(feature_eng, "DB_PATH", self.db_path):
            return load_cleaned_data(min_distance_km=None, **kwargs)

    def test_calories_are_used_and_kilojoules_are_ignored(self) -> None:
        self._insert_activity(1, calories=500.0, kilojoules=1000.0)
        frame = self._load()
        self.assertEqual(frame.iloc[0]["kcal_clean"], 500.0)

    def test_activity_without_calories_is_excluded(self) -> None:
        self._insert_activity(1, calories=None, kilojoules=1000.0)
        frame = self._load()
        self.assertTrue(frame.empty)

    def test_stop_heavy_activity_is_retained_and_stop_time_is_derived(self) -> None:
        self._insert_activity(1, moving_time=1800, elapsed_time=7200)
        frame = self._load()
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["stopped_time"], 5400)

    def test_gradient_has_correct_percent_units(self) -> None:
        self._insert_activity(1)
        frame = self._load()
        self.assertEqual(frame.iloc[0]["gradient_pct"], 10.0)

    def test_separate_mode_excludes_unknown_descent(self) -> None:
        self._insert_activity(1, descent=None)
        self.assertTrue(self._load(elevation_mode="separate").empty)

    def test_activity_type_filter_uses_sport_type(self) -> None:
        self._insert_activity(1, sport_type="VirtualRide")
        frame = self._load(activity_types=["VirtualRide"])
        self.assertEqual(len(frame), 1)

    def test_all_elevation_ranges_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            load_cleaned_data(min_elevation_down_m=-1)
        with self.assertRaises(ValueError):
            load_cleaned_data(min_elevation_up_m=20, max_elevation_up_m=10)
        with self.assertRaises(ValueError):
            load_cleaned_data(min_elevation_down_m=20, max_elevation_down_m=10)


class PresetWindowTests(unittest.TestCase):
    def test_last_365_days_has_365_inclusive_dates(self) -> None:
        start, end = get_last_365_days()
        self.assertEqual((end - start).days, 364)
        self.assertEqual(end, date.today())

    def test_builtin_preset_accepts_ridge_override(self) -> None:
        with (
            patch.object(train_presets, "train_sport", return_value=True) as train,
            patch("builtins.print"),
        ):
            train_presets.train_single_preset(
                "gravel_all", regression_type="ridge"
            )
        self.assertEqual(train.call_args.kwargs["regression_type"], "ridge")


class ModelIntegrationTests(unittest.TestCase):
    @staticmethod
    def _training_frame(sample_count: int = MIN_TRAINING_SAMPLES + 10) -> pd.DataFrame:
        rows = []
        for index in range(sample_count):
            distance = 10.0 + index
            elevation = 100.0 + index * 8.0
            moving = 900.0 + distance * 120.0 + elevation * 2.0
            stopped = 120.0 + distance * 8.0
            rows.append(
                {
                    "distance_km": distance,
                    "elevation_m": elevation,
                    "moving_time": moving,
                    "stopped_time": stopped,
                    "kcal_clean": 200.0 + distance * 25.0 + elevation * 0.3,
                    "start_date": 1_600_000_000 + index * 86_400,
                }
            )
        return pd.DataFrame(rows)

    def test_training_requires_enough_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trained = train_models_for_sport(
                self._training_frame(MIN_TRAINING_SAMPLES - 1),
                "ride",
                "too-small",
                model_dir=Path(temp_dir),
            )
        self.assertFalse(trained)

    def test_training_and_prediction_preserve_time_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            trained = train_models_for_sport(
                self._training_frame(), "ride", "test-model", model_dir=model_dir
            )
            self.assertTrue(trained)
            metadata = json.loads((model_dir / "test-model.txt").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["regression_type"], "ridge")
            self.assertEqual(metadata["features"], ["distance_km", "elevation_m"])
            self.assertEqual(metadata["energy"], {"unit": "kcal", "source": "calories"})
            self.assertIn("stopped_time", metadata["regressions"])
            self.assertIn("error_p90", metadata["regressions"]["moving_time"])

            with patch.object(predictor, "MODEL_DIR", model_dir):
                shorter = predictor.predict_tour("test-model", 20.0, 180.0)
                longer = predictor.predict_tour("test-model", 30.0, 180.0)
                outside = predictor.predict_tour("test-model", 100.0, 180.0)
                planned_break = predictor.predict_tour(
                    "test-model", 20.0, 180.0, stopped_time_s=3600
                )

                metadata.pop("energy")
                (model_dir / "test-model.txt").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                legacy_energy = predictor.predict_tour("test-model", 20.0, 180.0)

            self.assertGreaterEqual(
                longer["predicted_moving_time_sec"], shorter["predicted_moving_time_sec"]
            )
            self.assertEqual(
                shorter["predicted_elapsed_time_sec"],
                shorter["predicted_moving_time_sec"]
                + shorter["predicted_stopped_time_sec"],
            )
            self.assertEqual(planned_break["predicted_stopped_time_sec"], 3600)
            self.assertIsNone(legacy_energy["predicted_kcal"])
            self.assertIsNone(legacy_energy["predicted_kcal_interval"])
            self.assertTrue(outside["warnings"])
            self.assertGreaterEqual(
                shorter["predicted_elapsed_time_interval_sec"][1],
                shorter["predicted_elapsed_time_sec"],
            )

    def test_linear_can_be_selected_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self.assertTrue(
                train_models_for_sport(
                    self._training_frame(),
                    "ride",
                    "linear-model",
                    model_dir=model_dir,
                    regression_type="linear",
                )
            )
            metadata = json.loads(
                (model_dir / "linear-model.txt").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["regression_type"], "linear")
            self.assertIsNone(metadata["regressions"]["moving_time"]["alpha"])

    def test_saved_separate_model_update_reuses_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "saved.txt"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_name": "Saved",
                        "sport_type": "ride",
                        "features": [
                            "distance_km",
                            "elevation_up_m",
                            "elevation_down_m",
                        ],
                        "elevation_mode": "separate",
                        "sample_count": 30,
                        "filters": {
                            "min_distance_km": 5.0,
                            "elevation_mode": "separate",
                        },
                        "regressions": {},
                    }
                ),
                encoding="utf-8",
            )
            frame = self._training_frame()
            frame["elevation_up_m"] = frame["elevation_m"]
            frame["elevation_down_m"] = frame["elevation_m"]
            with (
                patch.object(update_models, "load_cleaned_data", return_value=frame) as load,
                patch.object(update_models, "train_models_for_sport", return_value=True) as train,
            ):
                self.assertTrue(update_models.update_model(metadata_path))

            self.assertEqual(load.call_args.kwargs["elevation_mode"], "separate")
            self.assertEqual(load.call_args.kwargs["min_distance_km"], 5.0)
            self.assertTrue(train.call_args.kwargs["separate_elevation"])
            self.assertEqual(train.call_args.kwargs["regression_type"], "linear")

    def test_legacy_model_is_migrated_to_two_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            metadata_path = model_dir / "tour.txt"
            source_model = model_dir / "tour.joblib"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_name": "Tour",
                        "model_file": "tour.joblib",
                        "sport_type": "ride",
                    }
                ),
                encoding="utf-8",
            )
            source_model.write_bytes(b"legacy")

            def fake_update(_path, regression_type_override, model_name_override):
                variant_id = update_models._model_id(model_name_override)
                (model_dir / f"{variant_id}.txt").write_text("{}", encoding="utf-8")
                (model_dir / f"{variant_id}.joblib").write_bytes(
                    regression_type_override.encode("ascii")
                )
                return True

            with (
                patch.object(update_models, "MODEL_DIR", model_dir),
                patch.object(update_models, "update_model", side_effect=fake_update),
                patch("builtins.print"),
            ):
                self.assertTrue(update_models.create_model_variants(metadata_path))

            self.assertTrue((model_dir / "tour_linear.txt").exists())
            self.assertTrue((model_dir / "tour_ridge.txt").exists())
            self.assertTrue((model_dir / "legacy" / "tour.txt").exists())
            self.assertTrue((model_dir / "legacy" / "tour.joblib").exists())


if __name__ == "__main__":
    unittest.main()
