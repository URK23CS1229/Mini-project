@echo off
echo ========================================
echo Installing Dashboard Dependencies
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo This may take a few minutes...
echo.

pip install streamlit pandas plotly joblib numpy scikit-learn xgboost

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Starting dashboard...
echo.

streamlit run dashboard.py --server.port 8501 --server.headless true

pause
