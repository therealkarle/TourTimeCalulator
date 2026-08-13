# Tour Time Calculator

Tour Time Calculator is a local Python command-line application that uses your historical Strava activities to estimate the duration and performance metrics of a planned cycling tour.

It stores Strava data in a local SQLite cache and trains named regression models for:

- elapsed time and moving time
- energy consumption
- average speed
- average and weighted average power, when available
- average heart rate, when available

The project is designed to keep personal activity data and trained models on your computer. Nothing is uploaded by the application.

## Requirements

- Python 3.11 or newer (the dependency set is tested with Python 3.14)
- A Strava API application if you want to synchronize activities
- Windows, macOS, or Linux

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/<your-github-username>/tour-time-calculator.git
cd tour-time-calculator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, `run.bat` provides an interactive menu for the main commands. It can use `uv` automatically when it is installed.

## Strava setup

1. Open the [Strava Developer Portal](https://www.strava.com/settings/api) and create an application.
2. Set the authorization callback domain to `localhost`.
3. Copy the client ID and client secret.
4. Create a local environment file from the template:

   ```powershell
   Copy-Item .env.example .env
   ```

   On macOS/Linux, use `cp .env.example .env`.

5. Add your client ID and client secret to `.env`:

   ```dotenv
   STRAVA_CLIENT_ID=your_client_id
   STRAVA_CLIENT_SECRET=your_client_secret
   STRAVA_REFRESH_TOKEN=your_refresh_token
   ```

6. Run the OAuth helper to obtain the refresh token:

   ```bash
   python scripts/get_strava_refresh_token.py
   ```

The helper opens Strava in your browser, starts a temporary local callback server, and writes the refresh token to `.env`. If the browser does not open, use `--no-browser` and open the printed URL manually. The default callback URL is `http://localhost:8765/callback`; use `--port` if that port is already in use.

Never commit `.env` or share your Strava credentials.

## Usage

### Complete interactive workflow

```bash
python main.py
```

This synchronizes activities, asks you to choose a sport and model name, trains the model, and estimates a tour.

### Synchronize Strava activities

```bash
python -m scripts.sync_strava
```

Activities are stored in `data/strava_cache.sqlite`. Later syncs request only activities added since the latest cached activity.

### Train a model

Interactively:

```bash
python -m scripts.train_models
```

Or provide the sport and model name directly:

```bash
python -m scripts.train_models --sport ride --name "Road cycling"
python -m scripts.train_models --sport mtb-ride --name "Mountain biking"
python -m scripts.train_models --sport gravel --name "Gravel riding"

# Train a model with separate ascent and descent parameters
python -m scripts.train_models --sport ride --name "Ascent and descent" --separate-elevation
```

Supported sport profiles are `ride`, `mtb-ride`, and `gravel`. The command also supports filters for exact Strava activity types, gear, commuting, power data, distance, elevation, and inclusive UTC date ranges. Run the following for all options:

```bash
python -m scripts.train_models --help
```

For example, train a non-commute model using only distance and elevation:

```bash
python -m scripts.train_models \
  --sport ride \
  --name "Non-commute distance and elevation" \
  --no-commute \
  --distance-elevation-only
```

Models are saved as Joblib files in `models/`. Their human-readable metadata is saved next to them as `.txt` files. These generated files are ignored by Git by default.

### Estimate a tour

Interactively choose from the available models:

```bash
python -m scripts.estimate_tour_duration
```

Or select a model and provide route inputs directly:

```bash
python -m scripts.estimate_tour_duration \
  --model <model-id> \
  --distance-km 85 \
  --elevation-m 920

# For separate-elevation models:
python -m scripts.estimate_tour_duration \
  --model <model-id> \
  --distance-km 85 \
  --elevation-up-m 920 \
  --elevation-down-m 880
```

### Update existing models

Retrain all saved models using the filters stored in their metadata:

```bash
python -m scripts.update_models
```

Use `--no-sync` to update from the existing local cache without contacting Strava.

## Data, privacy, and rate limits

The application stores local data in:

```text
data/strava_cache.sqlite  # cached activity data
models/                    # generated models and metadata
```

The Strava API allows 200 calls per 15 minutes for the short-term limit. The synchronization client monitors API usage and pauses when the configured short-term threshold is reached.

At least five valid activities are required to train a model for a sport. Predictions are estimates based on your historical activities and should not be treated as guarantees.

## Project structure

```text
main.py                         Complete interactive workflow
scripts/sync_strava.py          Download and cache Strava activities
scripts/train_models.py         Train named models
scripts/update_models.py        Retrain saved models
scripts/estimate_tour_duration.py  Estimate a planned tour
scripts/get_strava_refresh_token.py OAuth helper for Strava authentication
scripts/train_presets.py        Manage and train preset model configurations
src/                            Data, model, prediction, and Strava logic
data/                           Local SQLite cache (ignored by Git)
models/                         Generated models (ignored by Git)
```

## Functions

### Main Entry Point

#### `main.py`

- **`main()`** - Orchestrates the complete interactive workflow: synchronizes Strava activities, prompts for sport profile and model name, trains a regression model, and estimates a tour duration.

### Scripts

#### `scripts/sync_strava.py`

- **`main()`** - Downloads and caches Strava activities into a local SQLite database. Supports incremental syncs using the latest cached activity date.

#### `scripts/train_models.py`

- **`train_sport(sport, model_name, ...)`** - Trains and saves regression models for a specific sport using filtered Strava activity data.
- **`main()`** - Interactive CLI for training models with optional filters (sport, gear, commuting, distance, elevation, power data, date ranges).
- **`non_negative_float(value)`** - Argument parser validator for non-negative float values.
- **`iso_date(value)`** - Argument parser validator for ISO 8601 date format.

#### `scripts/update_models.py`

- **`update_model(metadata_path)`** - Retrains a single saved model using the filters stored in its metadata.
- **`main()`** - Retrains all saved models in the `models/` directory with optional `--no-sync` flag to skip Strava API calls.

#### `scripts/estimate_tour_duration.py`

- **`main()`** - Interactive CLI for estimating a planned tour. Selects a model and prompts for distance and elevation inputs, then predicts time and performance metrics.

#### `scripts/get_strava_refresh_token.py`

- **`read_env_value(name)`** - Reads a configuration value from `.env` file.
- **`update_env_value(name, value)`** - Writes a configuration value to `.env` file.
- **`exchange_code(client_id, client_secret, code)`** - Exchanges Strava authorization code for a refresh token via OAuth.
- **`main()`** - OAuth flow helper. Opens Strava authorization in the browser, receives the callback, and stores the refresh token in `.env`.

#### `scripts/train_presets.py`

- **`get_last_365_days()`** - Returns a tuple of (start_date, end_date) for the last 365 days.
- **`load_saved_presets()`** - Loads all preset model configurations from `presets/` directory.
- **`save_preset(name, config)`** - Saves a preset configuration to a JSON file.
- **`refresh_saved_presets()`** - Retrains all saved presets using the last 365 days of activity data.
- **`train_all_presets(dry_run)`** - Trains all presets with optional dry-run mode to preview without saving.
- **`train_single_preset(preset_name, dry_run)`** - Trains a specific preset by name.
- **`train_custom_preset(dry_run)`** - Interactive CLI to create and train a custom preset configuration.
- **`train_interactive(dry_run)`** - Interactive workflow for training presets with real-time feedback.
- **`main()`** - CLI entry point for managing and training preset configurations.

### Source Code (`src/`)

#### `src/strava_client.py`

- **`ensure_db_schema(db_path)`** - Creates or migrates the SQLite activities table and indices.
- **`StravaClient.refresh_access_token()`** - Obtains a fresh OAuth access token using the configured refresh token.
- **`StravaClient._init_db()`** - Initializes the database schema on client instantiation.
- Additional methods for syncing activities, rate limiting, and caching Strava data.

#### `src/feature_eng.py`

- **`available_sport_profiles()`** - Returns a dictionary of sport profiles with at least one matching cached activity.
- **`load_cleaned_data(...)`** - Loads and cleans Strava activities from the cache with optional filters for sport type, distance, elevation, power data, heart rate, moving time, and date ranges. Supports separate elevation mode (uphill/downhill split).
- **`_validate_filter_ranges(...)`** - Validates that filter ranges are logically consistent and positive.
- **`_utc_timestamp(value)`** - Converts a date to UTC Unix timestamp.

#### `src/model_trainer.py`

- **`train_models_for_sport(df, sport_name, model_name, ...)`** - Trains and persists regression models for elapsed time, moving time, energy consumption, average power, weighted average power, and average heart rate. Supports separate elevation parameters and distance/elevation-only models.

#### `src/predictor.py`

- **`list_models(model_dir)`** - Returns a list of all trained models with their metadata from the models directory.
- **`predict_tour(model_name, distance_km, elevation_m, descent_m)`** - Predicts tour metrics (time, energy, power, heart rate) for a given model and route parameters.
- **`_predict_optional(model, features)`** - Predicts optional metrics (power, heart rate) when available in the model.
- **`_rounded_or_none(value)`** - Rounds predictions to integers or returns None for unavailable metrics.

#### `src/config.py`

- Configuration module defining paths, environment variables, and API endpoints.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by the copyright holder.
