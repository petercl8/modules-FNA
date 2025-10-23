@echo off
:: ===== Checkpoint Update =====
:: cd C:\path\to\modules-FNA

:: Stage all changes
git add .

:: Prompt for commit message
set /p msg=Enter checkpoint commit message: 
if "%msg%"=="" set msg=Checkpoint update

:: Commit
git commit -m "%msg%" 2>nul || echo No changes to commit.

:: Push to remote
git push origin main

echo Checkpoint changes pushed to GitHub.
pause
