@echo off
cd /d "%~dp0site"
echo Starting HTTP server on http://localhost:8080 ...
start http://localhost:8080
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -m http.server 8080
) else (
    python -m http.server 8080
)