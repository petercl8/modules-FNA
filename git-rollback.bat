@echo off
REM ===============================
REM Git rollback helper
REM ===============================

REM --- Config ---
setlocal enabledelayedexpansion

REM The branch to operate on
set BRANCH=main

REM --- Step 1: Make sure we're on the main branch ---
git checkout %BRANCH%
if errorlevel 1 (
    echo ❌ Failed to checkout %BRANCH%.
    pause
    exit /b
)

REM --- Step 2: Backup current branch ---
set BACKUP_BRANCH=backup_before_rollback_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
echo Creating backup branch !BACKUP_BRANCH! ...
git branch !BACKUP_BRANCH!
if errorlevel 1 (
    echo ❌ Failed to create backup branch.
    pause
    exit /b
)
git push origin !BACKUP_BRANCH! 
echo Backup branch created and pushed.

REM --- Step 3: Show last 20 commits ---
echo.
echo Last 20 commits:
git log --oneline -20
echo.

REM --- Step 4: Ask for commit hash to rollback to ---
set /p TARGET_HASH=Enter the commit hash to rollback to: 

REM --- Step 5: Reset local
eline -