@echo off
:: ===== Close Session =====

:: Optional final push
set /p pushfinal=Push any remaining changes before closing? (y/n): 
if /i "%pushfinal%"=="y" (
    git add .
    set /p msg=Enter commit message: 
    if "%msg%"=="" set msg=Final update
    git commit -m "%msg%" 2>nul || echo No changes to commit.
    git push origin main
    echo Final changes pushed.
)

echo Session closed.
