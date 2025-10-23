@echo off
set /p pushfinal=Push any remaining changes before closing? (y/n): 
if /i "%pushfinal%"=="y" (
    git add -A
    set /p msg=Enter commit message: 
    if "%msg%"=="" set msg=Final update
    git diff --cached --quiet
    if %errorlevel%==1 (
        git commit -m "%msg%"
    ) else (
        echo No changes to commit.
    )
    git push origin main
)
echo Session closed.
