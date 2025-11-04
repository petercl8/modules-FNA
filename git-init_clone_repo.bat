@echo off
:: ==============================
:: Clone Repository
:: ==============================

set /p folder=Enter local folder path to clone into: 
git clone git@github.com:petercl8/Flexible_CNN_Architecture_for_Medical_Physics.git "%folder%"

echo Repository cloned to %folder%.
pause
