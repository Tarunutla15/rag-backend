# Activate virtual environment and run the FastAPI server
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = Split-Path -Parent $scriptPath
& "$rootPath\venv\Scripts\Activate.ps1"
Set-Location $scriptPath
python -m app.main



