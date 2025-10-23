@echo off
:: ===== Start Git Session =====
echo Starting session...
:: cd C:\path\to\modules-FNA

:: Ensure SSH agent is running
powershell -Command "Start-Service ssh-agent"
powershell -Command "ssh-add $HOME\.ssh\id_ed25519"

:: Prompt for branch to work on
set /p branch=Enter branch to work on (default: main): 
if "%branch%"=="" set branch=main

:: Fetch latest commits from remote
git fetch origin

:: Optional: show commits that are on remote but not in local
echo === Remote commits not in local branch ===
git log HEAD..origin/%branch% --oneline

:: Confirm before merging
set /p mergeconfirm=Merge remote changes into %branch%? (y/n): 
if /i "%mergeconfirm%"=="y" (
    :: Ensure local branch exists, checkout it
    git checkout %branch% 2>nul || git checkout -b %branch% origin/%branch%
    :: Merge remote changes
    git merge origin/%branch%
    echo Remote changes merged.
) else (
    :: Just switch to local branch without merging
    git checkout %branch% 2>nul || git checkout -b %branch% origin/%branch%
    echo Working on local branch without merging remote changes.
)

echo Repository ready on branch %branch%.
