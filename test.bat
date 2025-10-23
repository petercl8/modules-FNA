@echo off
:: ===== Small Update =====

:: Stage all changes (new, modified, deleted)
git add -A

:: Check if there’s anything to commit
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Small edit"
) else (
    echo No changes to commit.
)

:: Push to remote
git push origin main