# Production start (Windows). From project root:
#   .\scripts\start-production.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$dirs = @()
if ($env:VENV_DIR) { $dirs += $env:VENV_DIR }
$dirs += ".venv", "venv"

$Uvicorn = $null
foreach ($dir in $dirs) {
    $candidate = Join-Path $Root "$dir\Scripts\uvicorn.exe"
    if (Test-Path $candidate) {
        $Uvicorn = $candidate
        break
    }
}

if (-not $Uvicorn) {
    Write-Error @"
No venv uvicorn found under .venv or venv. Create one and install deps:
  python -m venv venv
  .\venv\Scripts\pip install -r requirements.txt
"@
}

$HostAddr = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port = if ($env:PORT) { $env:PORT } else { "8000" }

& $Uvicorn app.main:app --host $HostAddr --port $Port
