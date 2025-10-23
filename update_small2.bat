@echo off
git add -A
git diff --cached --quiet
if %errorlevel%==1 (
    git commit -m "Small edit"
) else (
    echo No changes to commit.
)
git push origin main
