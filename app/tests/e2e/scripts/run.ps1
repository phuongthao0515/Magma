# Launch the agent or the frontend (Windows).
#   powershell -ExecutionPolicy Bypass -File run.ps1 agent
#   powershell -ExecutionPolicy Bypass -File run.ps1 frontend
# Paths derive from this script's location, so the absolute repo path doesn't
# matter. Backend defaults to the deployed HF Space; override $env:DOME_SERVER_URL.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("agent", "frontend")]
    [string]$Target
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path   # tests\e2e\scripts -> app
if (-not $env:DOME_SERVER_URL) { $env:DOME_SERVER_URL = "https://wafair-dome.hf.space" }

if ($Target -eq "agent") {
    Set-Location $AppDir
    Write-Host "[run agent] app=$AppDir backend=$env:DOME_SERVER_URL"
    python agent/executor.py --server-url $env:DOME_SERVER_URL
}
else {
    # VITE_API_URL = the URL the BROWSER calls directly (overrides the
    # localhost:8000 default baked into public/config.js).
    # VITE_BACKEND_URL = the Vite dev-proxy target (kept as a fallback).
    $env:VITE_API_URL = $env:DOME_SERVER_URL
    $env:VITE_BACKEND_URL = $env:DOME_SERVER_URL
    Set-Location (Join-Path $AppDir "client")
    Write-Host "[run frontend] client=$AppDir\client VITE_API_URL=$env:VITE_API_URL"
    npx vite --port 4000
}
