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
src/                            Data, model, prediction, and Strava logic
data/                           Local SQLite cache (ignored by Git)
models/                         Generated models (ignored by Git)
```

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by the copyright holder.
