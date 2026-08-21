"""Remove dormant calorie regressions from existing model artifacts."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib

# Also support: python scripts/strip_calorie_models.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_DIR


CALORIE_METADATA_KEYS = ("energy", "kcal_intercept", "kcal_coefficients")


def strip_metadata(metadata: dict[str, Any]) -> bool:
    """Remove calorie-only entries from one metadata dictionary."""
    changed = False
    for key in CALORIE_METADATA_KEYS:
        changed = metadata.pop(key, None) is not None or changed
    for container_name in ("regressions", "sample_counts"):
        container = metadata.get(container_name)
        if isinstance(container, dict) and "kcal_clean" in container:
            del container["kcal_clean"]
            changed = True
    return changed


def strip_artifact(artifact: dict[str, Any]) -> bool:
    """Remove serialized calorie estimators from one Joblib dictionary."""
    changed = artifact.pop("kcal_model", None) is not None
    models = artifact.get("models")
    if isinstance(models, dict) and "kcal_clean" in models:
        del models["kcal_clean"]
        changed = True
    return changed


def strip_model_directory(model_dir: Path = MODEL_DIR) -> tuple[int, int]:
    """Strip active and archived models, returning metadata/artifact counts."""
    metadata_changed = 0
    artifacts_changed = 0
    for path in model_dir.rglob("*.txt"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if strip_metadata(metadata):
            path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            metadata_changed += 1

    for path in model_dir.rglob("*.joblib"):
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or not strip_artifact(artifact):
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            joblib.dump(artifact, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        artifacts_changed += 1
    return metadata_changed, artifacts_changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    metadata_count, artifact_count = strip_model_directory(args.model_dir)
    print(
        f"Removed calorie data from {metadata_count} metadata files and "
        f"{artifact_count} model artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
