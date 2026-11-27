# Runs the node, restarting it if it exits (as run.sh does elsewhere).
Set-Location $PSScriptRoot
$env:MAGPIE_SUPERVISED = "1"
while ($true) { .\.venv\Scripts\python.exe magpie.py; Write-Host "[run] node exited, restarting in 5 s"; Start-Sleep 5 }
