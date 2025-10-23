@echo off
:: ==============================
:: Automatic Update: Commit & Push
:: ==============================

:: Stage all changes (new, modified, deleted files)
git add -A

:: Commit changes with default message
:: - If there are no changes, skip committing
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Quick automatic update"
) else (
    echo No changes to commit.
)

:: Push commits to the 'main' branch on the remote 'origin'
git push origin main

echo Update complete!
