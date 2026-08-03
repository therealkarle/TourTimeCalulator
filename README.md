# Tour Time Calculator

Tour Time Calculator is a Python application that uses historical Strava activities to predict elapsed time, moving time, calorie burn, average speed, average power, weighted average power, and average heart rate. It stores activity data locally in SQLite and trains linear regression models separately for cycling and running activities.

## Strava API Credentials Setup

### Step 1: Create a Strava Application

1. Sign in to the [Strava Developer Portal](https://www.strava.com/settings/api).
2. Create a new application.
3. Use the following values:

   - **Application Name:** `Tour Time Calculator`
   - **Category:** Select the most appropriate category.
   - **Website:** `http://localhost`
   - **Authorization Callback Domain:** `localhost`

4. Save the application.
5. Copy the generated **Client ID** and **Client Secret**.

### Step 2: Obtain and store the refresh token

Set `STRAVA_CLIENT_ID` and `STRAVA_CLIENT_SECRET` in `.env`, then run:

```bash
python scripts/get_strava_refresh_token.py
```

The script opens Strava in your browser, starts a temporary local callback server,
and stores the returned `STRAVA_REFRESH_TOKEN` in `.env`. Approving access is
required once; the application then uses this token to obtain fresh access tokens
automatically. If the browser does not open, use `--no-browser` and open the
printed URL manually.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/tour-time-calculator.git
cd tour-time-calculator
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Windows, the environment can also be created explicitly with:

```powershell
py -3.14 -m venv venv
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file from the provided template:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Update `.env` with your Strava credentials:

```dotenv
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_REFRESH_TOKEN=your_refresh_token
```

Do not commit `.env` to version control.

### Automatically obtain the refresh token

After adding `STRAVA_CLIENT_ID` and `STRAVA_CLIENT_SECRET`, the refresh token
can be obtained automatically via OAuth and saved to `.env`:

```bash
python scripts/get_strava_refresh_token.py
```

The Strava Developer Portal must have `localhost` configured as the
Authorization Callback Domain. The script uses
`http://localhost:8765/callback` by default. If the port is already in use, a
different port can be specified with `--port`.

## Usage

The three steps can be run independently, so models do not need to be retrained
for every estimate.

### 1. Update Strava data

```bash
python -m scripts.sync_strava
```

Activities are stored in `data/strava_cache.sqlite`. On subsequent runs, only
activities added since the latest cache entry are requested.

### 2. Train a named regression model

Interactively:

```bash
python -m scripts.train_models
```

The script prompts for a sport and model name. Alternatively, both values can
be provided directly:

```bash
python -m scripts.train_models --sport ride --name "Ride training"
python -m scripts.train_models --sport mtb-ride --name "MTB training"
# Example training variations for non-commute rides:
# 1. Distance and elevation only
python -m scripts.train_models --sport ride --name "Ride non-commute distance-elevation" --no-commute --distance-elevation-only
# 2. Distance, elevation, and elevation per km
python -m scripts.train_models --sport ride --name "Ride non-commute with elevation-per-km" --no-commute
# 3. Power data, distance, and elevation only
python -m scripts.train_models --sport ride --name "Ride non-commute power distance-elevation" --no-commute --power-data available --distance-elevation-only
# 4. Power data, distance, elevation, and elevation per km
python -m scripts.train_models --sport ride --name "Ride non-commute power with elevation-per-km" --no-commute --power-data available
# Sport-specific models with Strava filters:
python -m scripts.train_models --sport gravel --name "Gravel without commute" --no-commute
python -m scripts.train_models --sport mtb-ride --name "MTB with power" --power-data available
python -m scripts.train_models --sport ride --name "Ride training" --activity-type Ride --equipment g123
# Filter by distance, elevation gain, and inclusive UTC dates:
python -m scripts.train_models --sport ride --name "Summer 2025" \
  --min-distance-km 40 --max-distance-km 150 \
  --min-elevation-m 300 --max-elevation-m 2500 \
  --start-date 2025-04-01 --end-date 2025-09-30
```

Supported sport profiles are `ride`, `rennrad`, `gravelbike`, `mtb`, and `run`.
You can also filter by Strava activity type, `gear_id`, `--commute` or
`--no-commute`, and `--power-data available|missing`. Power data is considered
available when Strava provides `average_watts`, `weighted_average_watts`, or
`device_watts`.

The available English sport profiles are `ride`, `mtb-ride`, and `gravel`. The
training command validates them against the activity types in the local Strava
cache and only offers profiles with matching activities. It also supports
activity type, `gear_id`, commute, and power-data filters.

The trained model is stored as a Joblib file because a random-forest model cannot
be stored as an executable text file. Its description, name, and selection key
are stored in a readable `.txt` file in `models/`.

### 3. Estimate tour duration

Interactively:

```bash
python -m scripts.estimate_tour_duration
```

Or with command-line arguments:

```bash
python -m scripts.estimate_tour_duration --model rennrad_grundlagen --distance-km 85 --elevation-m 920
```

Without `--model`, the script displays all available models for selection.

### Modelle mit neuen Daten aktualisieren

Alle vorhandenen Modelle können mit den in ihren Metadaten gespeicherten
Filtern neu trainiert werden:

```bash
python -m scripts.update_models
```

Das Script synchronisiert zuerst neue Strava-Aktivitäten. Mit
`--no-sync` wird ausschließlich der lokale Cache verwendet.

Bei jedem Training werden die verwendete Aktivitätszahl und alle Koeffizienten
angezeigt. Beim Aktualisieren werden zusätzlich die Änderungen gegenüber dem
vorherigen Modell ausgegeben.

## Complete workflow

Launch the interactive command-line application:

```bash
python main.py
```

The application will:

1. Synchronize new Strava activities.
2. Store them in `data/strava_cache.sqlite`.
3. Train linear regression models for cycling and running data.
4. Save trained models in the `models/` directory.
5. Ask for a sport, route distance, and elevation gain.
6. Display the predicted tour results.

Example console output:

```text
========================================
       Strava Tour Predictor
========================================

Sync finished: 42 activities cached.

Training regression models...
[RIDE] MAE -> Duration: 8.4 min | Energy: 96 kcal
[RUN] MAE -> Duration: 3.1 min | Energy: 54 kcal

Enter sport type (ride/run) [default: ride]: ride
Enter route distance in km: 85
Enter elevation gain in m: 920

================ RESULT ================
Sport:             Ride
Distance / Elev:   85.0 km | 920.0 m
Elapsed Time:      3h 18m
Moving Time:       3h 02m
Estimated Energy:  2140 kcal
Avg Speed:         25.8 km/h
Average Power:     185 W
Weighted Avg Power:203 W
Average HR:        148 bpm
```

If Strava credentials are missing, the application skips synchronization and uses any existing local data. At least five valid activities are required to train models for a sport.

## Rate Limits and Privacy

The application automatically monitors Strava API usage and pauses when the short-term rate-limit threshold is reached. Strava's short-term API limit is 200 calls per 15 minutes.

Activity data is stored locally in the SQLite database:

```text
data/strava_cache.sqlite
```

Trained models are stored locally as `joblib` files:

```text
models/
```

No activity data or trained models are uploaded by this application. Keep your Strava credentials private and never commit `.env` to the repository.
