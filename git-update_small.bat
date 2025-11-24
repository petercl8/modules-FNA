

:: --- Git operations ---
git add -A
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Quick automatic update"
) else (
    echo No changes to commit.
)
git push origin main

echo Update complete!
