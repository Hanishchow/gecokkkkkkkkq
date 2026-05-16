@echo off
:: GEOCK v2 Quick Launcher
:: Double-click this to open terminal in GEOCK directory

:: Try Windows Terminal first, fallback to cmd
if exist "C:\Users\yakka\AppData\Local\Microsoft\WindowsApps\wt.exe" (
    start "" "C:\Users\yakka\AppData\Local\Microsoft\WindowsApps\wt.exe" -d "C:\Users\yakka\wsl_home\home\chow\autoresearch"
) else (
    start cmd /k "cd /d C:\Users\yakka\wsl_home\home\chow\autoresearch && bash"
)
