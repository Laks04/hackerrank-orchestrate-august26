@echo off
REM ============================================================
REM  Message Notification Router - one-click run
REM  Double-click this file. It generates dataset/output.csv,
REM  runs the unit tests and the evaluation report, and writes
REM  everything to run_output.txt next to this script.
REM ============================================================
setlocal
cd /d "%~dp0"
set "LOG=%~dp0run_output.txt"

> "%LOG%" echo === Message Notification Router : run log ===
>> "%LOG%" echo.

REM --- locate a working Python interpreter -------------------
set "PY="
python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if not defined PY (
  py --version >nul 2>&1
  if not errorlevel 1 set "PY=py"
)
if not defined PY (
  >> "%LOG%" echo ERROR: No Python interpreter found on PATH.
  >> "%LOG%" echo Install Python 3 from https://www.python.org/downloads/
  >> "%LOG%" echo and be sure to tick "Add python.exe to PATH" during setup.
  type "%LOG%"
  echo.
  pause
  exit /b 1
)

>> "%LOG%" echo Interpreter: %PY%
%PY% --version >> "%LOG%" 2>&1
>> "%LOG%" echo.

REM --- step 1: generate predictions ---------------------------
>> "%LOG%" echo === STEP 1: generate dataset/output.csv ===
%PY% main.py --dataset-dir ..\dataset --output-path ..\dataset\output.csv >> "%LOG%" 2>&1
>> "%LOG%" echo [exit code %errorlevel%]
>> "%LOG%" echo.

REM --- step 2: unit tests -------------------------------------
>> "%LOG%" echo === STEP 2: unit tests ===
%PY% -m unittest discover -s tests -v >> "%LOG%" 2>&1
>> "%LOG%" echo [exit code %errorlevel%]
>> "%LOG%" echo.

REM --- step 3: evaluation against labeled samples -------------
>> "%LOG%" echo === STEP 3: evaluation vs sample_messages.csv ===
%PY% evaluation\main.py --dataset-dir ..\dataset >> "%LOG%" 2>&1
>> "%LOG%" echo [exit code %errorlevel%]
>> "%LOG%" echo.

>> "%LOG%" echo === DONE ===
type "%LOG%"
echo.
echo ------------------------------------------------------------
echo Full log saved to: %LOG%
echo Tell Claude it has finished and it will read the log.
echo ------------------------------------------------------------
pause
