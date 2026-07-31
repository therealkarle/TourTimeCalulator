# Tour Time Calculator

Tour Time Calculator is a Python application that uses historical Strava activities to predict route duration, calorie burn, and average speed. It stores activity data locally in SQLite and trains linear regression models separately for cycling and running activities.

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

### Step 2: Obtain and store the Refresh Token

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

### Refresh Token automatisch eintragen

Nach dem Eintragen von `STRAVA_CLIENT_ID` und `STRAVA_CLIENT_SECRET` kann der
Refresh Token automatisch per OAuth geholt und in `.env` gespeichert werden:

```bash
python scripts/get_strava_refresh_token.py
```

Für das Script muss im Strava Developer Portal `localhost` als
Authorization Callback Domain eingetragen sein. Standardmäßig verwendet es
`http://localhost:8765/callback`. Falls der Port bereits belegt ist, kann ein
anderer Port über `--port` angegeben werden.

## Usage

Die drei Arbeitsschritte können unabhängig voneinander ausgeführt werden. Dadurch
müssen die Modelle nicht bei jeder Schätzung neu trainiert werden.

### 1. Strava-Daten aktualisieren

```bash
python -m scripts.sync_strava
```

Die Aktivitäten werden in `data/strava_cache.sqlite` gespeichert. Beim nächsten
Aufruf werden nur Aktivitäten seit dem letzten Cache-Eintrag abgefragt.

### 2. Ein benanntes Regressionsmodell trainieren

Interaktiv:

```bash
python -m scripts.train_models
```

Das Skript fragt nach Sportart und Modellnamen. Alternativ können beide Werte
direkt übergeben werden:

```bash
python -m scripts.train_models --sport ride --name "Ride training"
python -m scripts.train_models --sport mtb-ride --name "MTB training"
# Optional: train with only distance and elevation
python -m scripts.train_models --sport ride --name "Nur Distanz und Höhenmeter" --distance-elevation-only
# Sportartspezifisch und mit Strava-Filtern:
python -m scripts.train_models --sport gravel --name "Gravel without commute" --no-commute
python -m scripts.train_models --sport mtb-ride --name "MTB with power" --power-data available
python -m scripts.train_models --sport ride --name "Ride training" --activity-type Ride --equipment g123
```

Unterstützte Sportprofile sind `ride`, `rennrad`, `gravelbike`, `mtb` und `run`.
Zusätzlich können Strava-Aktivitätstypen, `gear_id`, `--commute` bzw.
`--no-commute` und `--power-data available|missing` gefiltert werden. Power-Daten
gelten als verfügbar, wenn Strava `average_watts`, `weighted_average_watts` oder
`device_watts` liefert.

The available English sport profiles are `ride`, `mtb-ride` and `gravel`. The
training command validates them against the
activity types in the local Strava cache and only offers profiles with matching
activities. It also supports activity type, `gear_id`, commute and power-data
filters.

Das trainierte Modell wird technisch als Joblib-Datei gespeichert, weil ein
Random-Forest-Modell nicht direkt als Textdatei ausführbar gespeichert werden
kann. Die zugehörige Beschreibung, der Name und die Auswahlkennung werden als
lesbare `.txt`-Datei in `models/` gespeichert.

### 3. Tourdauer schätzen

Interaktiv:

```bash
python -m scripts.estimate_tour_duration
```

Oder mit Kommandozeilenargumenten:

```bash
python -m scripts.estimate_tour_duration --model rennrad_grundlagen --distance-km 85 --elevation-m 920
```

Ohne `--model` zeigt das Skript alle verfügbaren Modelle zur Auswahl an.

## Gesamter Workflow

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
Estimated Time:    3h 18m
Estimated Energy:  2140 kcal
Avg Speed:         25.8 km/h
```

If Strava credentials are missing, the application skips synchronization and uses any existing local data. At least five valid activities are required to train models for a sport.

## Rate Limits and Privacy

The application automatically monitors Strava API usage and pauses when the short-term rate-limit threshold is reached. Strava’s short-term API limit is 200 calls per 15 minutes.

Activity data is stored locally in the SQLite database:

```text
data/strava_cache.sqlite
```

Trained models are stored locally as `joblib` files:

```text
models/
```

No activity data or trained models are uploaded by this application. Keep your Strava credentials private and never commit `.env` to the repository.
