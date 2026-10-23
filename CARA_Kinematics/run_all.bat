@echo off
cd /d "%~dp0"
python workflow.py --stage all --out results\run_1601 --preview
pause
