# Starts the local Minecraft 1.21.4 server (offline mode, localhost only) with the portable Java 21.
$root = Split-Path $PSScriptRoot -Parent
$java = Join-Path $root "tools\jre21\bin\java.exe"
if (-not (Test-Path $java)) { $java = "java" }
Set-Location (Join-Path $root "server")
if (-not (Select-String -Path eula.txt -Pattern "^eula=true" -Quiet)) {
    Write-Host "Read the Minecraft EULA (https://aka.ms/MinecraftEULA) and set eula=true in server\eula.txt first." -ForegroundColor Yellow
    exit 1
}
& $java -Xms2G -Xmx4G -jar server.jar nogui
