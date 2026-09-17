# Runs the fly brain + one Mineflayer body. Extra arguments go to flybrain.run, e.g.
#   .\scripts\play.ps1 --pure-fly
#   .\scripts\play.ps1 --checkpoint runs\default\latest.pt
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$bodies = 1
for ($i = 0; $i -lt $args.Count; $i++) { if ($args[$i] -eq "--bodies") { $bodies = [int]$args[$i + 1] } }
$bot = Start-Process node -ArgumentList "bot.js --count $bodies" -WorkingDirectory (Join-Path $root "bot") -PassThru -NoNewWindow
try {
    Push-Location (Join-Path $root "brain")
    & $py -u -W ignore -m flybrain.run @args
} finally {
    Pop-Location
    if (-not $bot.HasExited) { Stop-Process -Id $bot.Id }
}
