"""Roboflow'daki DisasterVision workflow'unu yerel örnek görsellerde çalıştırır.

Kullanım (proje kökünden):  .venv\\Scripts\\python src\\roboflow_test.py [gorsel ...]
Çıktılar ornekler/roboflow_cikti/ altına yazılır; aynı görsellerde yerel model
sonucu da yazdırılır ki bulut ile yerel tahminler karşılaştırılabilsin.
"""
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import requests
from dotenv import dotenv_values

KOK = Path(__file__).resolve().parent.parent
URL = "https://serverless.roboflow.com/baris-emre-ruzgar/workflows/disastervision-tespit"
CIKTI = KOK / "ornekler" / "roboflow_cikti"
GUVEN = 0.25

anahtar = dotenv_values(KOK / ".env")["ROBOFLOW_API_KEY"].strip()
gorseller = [Path(p) for p in sys.argv[1:]] or sorted((KOK / "ornekler").glob("*.jpg"))
CIKTI.mkdir(exist_ok=True)

yerel_model = None
try:
    from ultralytics import YOLO
    yerel_model = YOLO(str(KOK / "modeller" / "disastervision.pt"))
except Exception as hata:  # yükleme ortamında ultralytics sürümü farklı olabilir
    print("yerel model atlandı:", hata)

for yol in gorseller:
    veri = base64.b64encode(yol.read_bytes()).decode()
    yanit = requests.post(URL, json={
        "api_key": anahtar,
        "inputs": {"image": {"type": "base64", "value": veri}, "confidence": GUVEN},
    }, timeout=120)
    yanit.raise_for_status()
    sonuc = yanit.json()["outputs"][0]

    (CIKTI / f"{yol.stem}_roboflow.jpg").write_bytes(base64.b64decode(sonuc["gorsel"]["value"]))
    tahminler = sonuc["tespitler"]["predictions"]
    (CIKTI / f"{yol.stem}_roboflow.json").write_text(
        json.dumps(tahminler, ensure_ascii=False, indent=1), encoding="utf-8")

    bulut = Counter(t["class"] for t in tahminler)
    print(f"{yol.name}: roboflow insan={sonuc['insan_sayisi']} sınıflar={dict(bulut)}")
    if yerel_model:
        kutular = yerel_model.predict(str(yol), conf=GUVEN, iou=0.7, imgsz=640, verbose=False)[0].boxes
        yerel = Counter(yerel_model.names[int(c)] for c in kutular.cls)
        print(f"{' ' * len(yol.name)}  yerel    sınıflar={dict(yerel)}")
