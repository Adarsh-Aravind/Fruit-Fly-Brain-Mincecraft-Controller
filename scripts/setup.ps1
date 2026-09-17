# One-time setup: Python venv + PyTorch (CUDA), bot packages, connectome data, server jar, portable Java 21.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
python -m pip install --user uv
if (-not (Test-Path .venv)) { python -m uv venv .venv --python 3.12 }
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
Push-Location bot; npm install; Pop-Location
Push-Location brain
& ..\.venv\Scripts\python.exe -m flybrain.fetch_data
& ..\.venv\Scripts\python.exe -W ignore -m flybrain.connectome --build
& ..\.venv\Scripts\python.exe -W ignore -m flybrain.brainmap
Pop-Location
if (-not (Test-Path server\eula.txt)) {
    Set-Content server\eula.txt "# Set to true only after YOU have read and agreed to https://aka.ms/MinecraftEULA`neula=false"
}
if (-not (Test-Path server\server.jar)) {
    $manifest = Invoke-RestMethod https://piston-meta.mojang.com/mc/game/version_manifest_v2.json
    $version = Invoke-RestMethod ($manifest.versions | Where-Object id -eq "1.21.4").url
    Invoke-WebRequest $version.downloads.server.url -OutFile server\server.jar
}
if (-not (Test-Path tools\jre21)) {
    New-Item -ItemType Directory -Force tools | Out-Null
    Invoke-WebRequest "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jre/hotspot/normal/eclipse" -OutFile tools\jre21.zip
    Expand-Archive tools\jre21.zip tools
    Get-ChildItem tools -Directory -Filter "jdk-21*" | Rename-Item -NewName jre21
    Remove-Item tools\jre21.zip
}
Write-Host "Setup done. Read the Minecraft EULA, set eula=true in server\eula.txt, then run scripts\server.ps1" -ForegroundColor Green
