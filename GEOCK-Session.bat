@echo off
title GEOCK v2 - Binding Affinity Prediction
color 0A

echo.
echo ============================================================
echo    GEOCK v2 - Binding Affinity Prediction Engine
echo ============================================================
echo.
echo Quick Results:
echo   - CASF-2007 R: 0.8766
echo   - CASF-2013 R: 0.8696  
echo   - CV R: 0.8432
echo.
echo Quick Commands:
echo   python geock_engine.py --smiles "CCO"
echo   python casf2007_validation.py
echo.
echo Documentation: C:\Users\yakka\Downloads\Geockk\GEOCK-QuickRef.md
echo.
echo ============================================================
echo.

cd /d C:\Users\yakka\wsl_home\home\chow\autoresearch

bash -c "echo 'Working directory:'; pwd; echo ''; echo 'Type python geock_engine.py --smiles \"CCO\" to test'; echo 'Type exit to close'; exec bash"
