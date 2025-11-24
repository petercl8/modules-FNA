@echo off
setlocal enabledelayedexpansion

:: --- Paths ---
set SOURCE_NOTEBOOK=.\FlexCNN_for_Medical_Physics\stitching_notebook.ipynb
set CFG=%USERPROFILE%\.my_sync_settings\drive_path.cfg

:: --- Load stored Google Drive path ---
for /f "usebackq delims=" %%A in ("%CFG%") do set "DRIVE_DEST_PATH=%%A"

:: Remove quotes and trailing spaces
set "DRIVE_DEST_PATH=%DRIVE_DEST_PATH:"=%"
for /f "tokens=* delims= " %%A in ("%DRIVE_DEST_PATH%") do set "DRIVE_DEST_PATH=%%A"

:: --- Validate folder exists ---
if not exist "%DRIVE_DEST_PATH%\" (
    echo ERROR: Google Drive folder "%DRIVE_DEST_PATH%" does not exist!
    pause
    exit /b
)

:: --- Create timestamp ---
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
    set DT=%%d%%b%%c
)
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set TM=%%a%%b%%c
)
set TIMESTAMP=%DT%_%TM%

:: --- Build destination filename with timestamp ---
set NOTEBOOK_NAME=stitching_notebook_%TIMESTAMP%.ipynb

:: --- Copy the notebook ---
echo Copying notebook to Google Drive folder as "%NOTEBOOK_NAME%"...
robocopy "%~dp0FlexCNN_for_Medical_Physics" "%DRIVE_DEST_PATH%" stitching_notebook.ipynb /R:2 /W:2 /NFL /NDL /NJH /NJS /nc /ns /np
:: Rename the copied file to add timestamp
rename "%DRIVE_DEST_PATH%\stitching_notebook.ipynb" "%NOTEBOOK_NAME%"

echo Notebook copied successfully as "%NOTEBOOK_NAME%".
echo.

:: --- Git operations ---
git add -A
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Quick automatic update"
) else (
    echo No changes to commit.
)
git push origin main

echo Update complete!
pause
