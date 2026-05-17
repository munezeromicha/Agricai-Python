# Production start (Windows). From project root:
#   .\scripts\start-production.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Uvicorn = Join-Path $Root ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path $Uvicorn)) {
    Write-Error @"
Missing $Uvicorn. Create the venv and install deps:
  python -m venv .venv
  .\.venv\Scripts\pip install -r requirements.txt
"@
}

$HostAddr = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port = if ($env:PORT) { $env:PORT } else { "8000" }

& $Uvicorn app.main:app --host $HostAddr --port $Port
