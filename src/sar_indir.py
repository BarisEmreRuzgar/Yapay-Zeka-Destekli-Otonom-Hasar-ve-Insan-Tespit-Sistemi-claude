"""Roboflow Universe'teki arama-kurtarma (SAR) insan veri setlerini YOLOv8 biçiminde indirir → veri/ham/sar/<ad>.

Anahtar .env içindeki ROBOFLOW_API_KEY. Kullanım: python src/sar_indir.py
"""
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from roboflow import Roboflow

KOK = Path(__file__).resolve().parent.parent
SETLER = [  # (çalışma alanı, proje, sürüm) — Universe'te bulunanlar
    ("ilinsk-univerzita-v-iline", "search-and-rescue", 4),
    ("drone-leak", "search-and-rescue-8ywn7", 7),
    ("trash-32e51", "search-rescue", 14),
    ("sandr", "drone-aerial", 1),
    ("new-workspace-qsuag", "heridal-t428z", 2),
]

if __name__ == "__main__":
    rf = Roboflow(api_key=dotenv_values(KOK / ".env")["ROBOFLOW_API_KEY"].strip())
    for ws, proje, surum in SETLER:
        hedef = KOK / "veri" / "ham" / "sar" / f"{ws}__{proje}"
        if hedef.exists() and any(hedef.rglob("*.txt")):
            print("var:", hedef.name); continue
        try:
            rf.workspace(ws).project(proje).version(surum).download("yolov8", location=str(hedef))
            print("indirildi:", hedef.name, flush=True)
        except Exception as e:
            print("HATA", ws, proje, surum, repr(e)[:300], flush=True)
