$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe src\lacmus_veri.py *> lacmus_veri.log
"BITTI" | Out-File lacmus_hazir.txt
