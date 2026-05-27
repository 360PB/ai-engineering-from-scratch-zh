@echo off
cd /d "%~dp0site"
echo Starting HTTP server on http://localhost:8080 ...
start http://localhost:8080/lesson.html
python -m http.server 8080