$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe src\sar_indir.py *> sar_indir_lacmus.log
"BITTI" | Out-File indir_lacmus_bitti.txt
