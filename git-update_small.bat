@echo off
setlocal enabledelayedexpansion

set SOURCE_NOTEBOOK=.\FlexCNN_for_Medical_Physics\stitching_notebook.ipynb
set CFG=%USERPROFILE%\.my_sync_settings\drive_path.cfg

:: --- Load stored Google Drive path ---
for /f "usebackq delims=" %%A in ("%CFG%") do set "DRIVE_DEST_PATH=%%A"

:: Remove any surrounding quotes
set "DRIVE_DEST_PATH=%DRIVE_DEST_PATH:"=%"

:: Trim trailing spaces
set "DRIVE_DEST_PATH=%DRIVE_DEST_PATH:~0,-0%"  :: this forces batch to normalize string
for /f "tokens=* delims= " %%A in ("%DRIVE_DEST_PATH%") do set "DRIVE_DEST_PATH=%%A"

:: Validate folder exists
if not exist "%DRIVE_DEST_PATH%\" (
    echo ERROR: Google Drive folder "%DRIVE_DEST_PATH%" does not exist!
    echo Please run the machine setup script first or check the folder.
    pause
    exit /b
)

:: Copy the notebook
echo Copying notebook to Google Drive folder...
xcopy "%SOURCE_NOTEBOOK%" "%DRIVE_DEST_PATH%\stitching_notebook.ipynb" /Y /C
if errorlevel 1 (
    echo ERROR: Failed to copy notebook. Check permissions and that the file is not open.
) else (
    echo Notebook copied successfully.
)


:: Git operations
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
