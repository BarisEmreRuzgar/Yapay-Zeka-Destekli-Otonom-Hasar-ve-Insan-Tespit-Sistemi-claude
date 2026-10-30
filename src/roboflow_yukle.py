"""Eğitilmiş DisasterVision ağırlıklarını Roboflow workspace'ine yükler.

Kullanım (proje kökünden):  .venv\\Scripts\\python src\\roboflow_yukle.py [--sadece-kontrol]
Anahtar .env içindeki ROBOFLOW_API_KEY satırından okunur.
"""
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from roboflow import Roboflow

KOK = Path(__file__).resolve().parent.parent
WORKSPACE = "baris-emre-ruzgar"
PROJE = "disastervision"

anahtar = (dotenv_values(KOK / ".env").get("ROBOFLOW_API_KEY") or "").strip()
if not anahtar:
    sys.exit(".env içinde ROBOFLOW_API_KEY yok")

rf = Roboflow(api_key=anahtar)
ws = rf.workspace(WORKSPACE)
print("workspace erişimi tamam:", ws.url)

if "--sadece-kontrol" in sys.argv:
    sys.exit(0)

ws.deploy_model(
    model_type="yolov8s",
    model_path=str(KOK / "runs" / "genis2"),
    project_ids=[PROJE],
    model_name="disastervision-yolov8s",
    filename="weights/best.pt",
)
