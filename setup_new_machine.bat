@echo off
:: ==============================
:: One-Time Machine Setup (SSH + Git)
:: ==============================

:: Prompt for GitHub username
set /p GITHUB_USER="Enter your GitHub username: "

:: Prompt for GitHub email
set /p GITHUB_EMAIL="Enter your GitHub email: "

:: 1) Set global Git identity
git config --global user.name "%GITHUB_USER%"
git config --global user.email "%GITHUB_EMAIL%"

:: 2) Generate SSH key if it doesn't exist
if not exist "%USERPROFILE%\.ssh\id_ed25519" (
    echo Generating new SSH key...
    ssh-keygen -t ed25519 -f "%USERPROFILE%\.ssh\id_ed25519" -N ""
) else (
    echo SSH key already exists.
)

:: 3) Show public key for GitHub
echo =========================
echo Copy the following key to GitHub (Account Settings → SSH and GPG keys):
type "%USERPROFILE%\.ssh\id_ed25519.pub"
echo =========================
pause

:: 4) Test SSH connection to GitHub
echo Testing SSH connection to GitHub...
ssh -T git@github.com
pause

echo Machine setup complete! Git and SSH are ready for repositories.
pause