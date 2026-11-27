# Magpie on Windows: the kernel only (no programs started here), passing work to an Ollama on this machine with Magpie's queue, records and reports.
# Needs Python 3.11 or newer (python.org, "Add python.exe to PATH"). Run from this folder in PowerShell:  .\install.ps1
# Nothing here contacts another server except pip (PyPI) for the three kernel packages.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) { py -3 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Installed. Start it with .\run.ps1 ; to start it at logon: .\.venv\Scripts\python.exe magpie.py autostart on"
Write-Host "Then, in the portal: add this machine as a provider (http://<this machine>:8790, its key from data\state\first_admin.key) and give it the model source 'local' = 127.0.0.1:11434 (Settings > Machines, its assets, Model source)."
