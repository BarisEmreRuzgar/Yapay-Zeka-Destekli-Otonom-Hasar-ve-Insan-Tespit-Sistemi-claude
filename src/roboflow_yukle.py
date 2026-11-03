"""Eğitilmiş DisasterVision ağırlıklarını Roboflow workspace'ine yükler.

Kullanım (proje kökünden):  .venv\\Scripts\\python src\\roboflow_yukle.py [--sadece-kontrol] [--insan]
  --insan: insan uzmanını disastervision-insan projesine yükler (koşu ve ad: --kosu insan_v5 --ad disastervision-insan-v5;
           Roboflow aynı adla yeniden yüklemede 404 veriyor → her sürüme yeni ad).
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

def arg(ad, varsayilan):
    return sys.argv[sys.argv.index(ad) + 1] if ad in sys.argv else varsayilan


insan = "--insan" in sys.argv
ws.deploy_model(
    model_type="yolov8s",
    model_path=str(KOK / "runs" / arg("--kosu", "insan_v5" if insan else "genis2")),
    project_ids=[PROJE + "-insan" if insan else PROJE],
    model_name=arg("--ad", "disastervision-insan-v5" if insan else "disastervision-yolov8s"),
    filename="weights/best.pt",
)
