@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>&1
if not errorlevel 1 (
    set "UV_CACHE_DIR=%~dp0.uv-cache"
    set "PYTHON=uv"
    set "PYTHON_ARGS=run --no-project --with-requirements requirements.txt python"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
) else (
    set "PYTHON=python"
    set "PYTHON_ARGS="
)

if not "%~1"=="" goto :%~1

:menu
cls
echo Tour Time Calculator
echo ====================
echo [1] Start application
echo [2] Estimate tour
echo [3] Synchronize Strava
echo [4] Train models
echo [5] Train presets
echo [6] Update models
echo [7] Set up Strava token
echo [0] Exit
echo.
choice /c 12345670 /n /m "Selection: "

if errorlevel 8 goto :end
if errorlevel 7 goto token
if errorlevel 6 goto update
if errorlevel 5 goto trainpresets
if errorlevel 4 goto train
if errorlevel 3 goto sync
if errorlevel 2 goto estimate
if errorlevel 1 goto app

:app
"%PYTHON%" %PYTHON_ARGS% main.py %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:estimate
"%PYTHON%" %PYTHON_ARGS% -m scripts.estimate_tour_duration %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:sync
"%PYTHON%" %PYTHON_ARGS% -m scripts.sync_strava %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:train
"%PYTHON%" %PYTHON_ARGS% -m scripts.train_models %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:trainpresets
"%PYTHON%" %PYTHON_ARGS% -m scripts.train_presets %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:update
"%PYTHON%" %PYTHON_ARGS% -m scripts.update_models %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:token
"%PYTHON%" %PYTHON_ARGS% scripts\get_strava_refresh_token.py %2 %3 %4 %5 %6 %7 %8 %9
goto :done

:done
echo.
pause
if "%~1"=="" goto :menu
goto :end

:end
endlocal
