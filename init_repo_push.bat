@echo off
:: ==============================
:: Initialize Local Repository and Push
:: ==============================

:: 1) Initialize Git repo (if not already)
git init

:: 2) Add all files and commit
git add -A
git commit -m "Initial commit"

:: 3) Add SSH remote
git remote remove origin 2>nul
git remote add origin git@github.com:petercl8/modules-FNA.git

:: 4) Rename branch to main
git branch -M main

:: 5) Push to GitHub
git push -u origin main

echo Repository initialized and pushed to GitHub.
pause
