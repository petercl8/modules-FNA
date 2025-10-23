@echo off
:: ===== Quick Update =====

:: Stage all changes
git add .

:: Commit with default message
git commit -m "Small edit" 2>nul || echo No changes to commit.

:: Push to origin main (SSH)
git push origin main

echo Small changes pushed to GitHub.
