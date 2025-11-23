@echo off
:: ==============================
:: One-Time Machine Setup (SSH + Git + Google Drive Path)
:: ==============================

:: --- GitHub setup ---
:: Prompt for GitHub username
set /p GITHUB_USER="Enter your GitHub username: "

:: Prompt for GitHub email
set /p GITHUB_EMAIL="Enter your GitHub email: "

:: Set global Git identity
git config --global user.name "%GITHUB_USER%"
git config --global user.email "%GITHUB_EMAIL%"

:: Generate SSH key if it doesn't exist
if not exist "%USERPROFILE%\.ssh\id_ed25519" (
    echo Generating new SSH key...
    ssh-keygen -t ed25519 -f "%USERPROFILE%\.ssh\id_ed25519" -N ""
) else (
    echo SSH key already exists.
)

:: Show public key for GitHub
echo =========================
echo Copy the following key to GitHub (Account Settings → SSH and GPG keys):
type "%USERPROFILE%\.ssh\id_ed25519.pub"
echo =========================
pause

:: Test SSH connection to GitHub
echo Testing SSH connection to GitHub...
ssh -T git@github.com
pause

:: ==============================
:: Google Drive folder setup
:: ==============================
set SYNC_CFG_DIR=%USERPROFILE%\.my_sync_settings
set SYNC_CFG_FILE=%SYNC_CFG_DIR%\drive_path.cfg

if not exist "%SYNC_CFG_DIR%" mkdir "%SYNC_CFG_DIR%"

:: Prompt for Google Drive folder until valid
:ASK_DRIVE
echo Enter the full path to your Google Drive folder where the notebook should be copied:
set /p DRIVE_PATH="Google Drive folder path: "

:: Validate folder exists
if not exist "%DRIVE_PATH%" (
    echo ERROR: The folder "%DRIVE_PATH%" does not exist!
    echo Please make sure Google Drive is installed and the folder exists.
    echo.
    goto ASK_DRIVE
)

:: Save validated path
echo %DRIVE_PATH% > "%SYNC_CFG_FILE%"
echo Google Drive path saved to %SYNC_CFG_FILE%.
echo.

echo Machine setup complete! Git, SSH, and notebook sync path are ready.
pause