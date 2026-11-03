$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe src\tur_kalibrasyon.py runs\deg\val_son.pkl *> tur_kalibrasyon.log
"BITTI" | Out-File kalibrasyon_bitti.txt
