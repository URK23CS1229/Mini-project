@echo off
echo Starting Carbon-Aware Workload Optimizer Dashboard...
echo.
streamlit run dashboard.py --server.port 8501 --server.headless true
pause
