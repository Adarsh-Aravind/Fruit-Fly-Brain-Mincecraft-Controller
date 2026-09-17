# Trains the learned readout with PPO. Extra arguments go to flybrain.train, e.g.
#   .\scripts\train.ps1 --bodies 4 --run default
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$bodies = 4
for ($i = 0; $i -lt $args.Count; $i++) { if ($args[$i] -eq "--bodies") { $bodies = [int]$args[$i + 1] } }
$argv = @($args)
if (-not ($argv -contains "--bodies")) { $argv += @("--bodies", "$bodies") }
$bot = Start-Process node -ArgumentList "bot.js --count $bodies" -WorkingDirectory (Join-Path $root "bot") -PassThru -NoNewWindow
try {
    Push-Location (Join-Path $root "brain")
    & $py -u -W ignore -m flybrain.train @argv
} finally {
    Pop-Location
    if (-not $bot.HasExited) { Stop-Process -Id $bot.Id }
}
