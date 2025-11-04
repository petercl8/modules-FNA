@echo off
:: ===== Start Session =====
:: Fetch latest changes from GitHub
git fetch origin

:: Switch to main branch locally
git checkout main

:: Merge remote changes
git pull origin main

echo Repository is up-to-date.
