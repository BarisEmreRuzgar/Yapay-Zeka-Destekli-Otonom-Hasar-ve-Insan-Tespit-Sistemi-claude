"""Ham veri setlerini indirir: Roboflow'daki afet veri setleri + VisDrone (havadan insan tespiti için).

Roboflow anahtarı ROBOFLOW_API_KEY ortam değişkeninden okunur.
Kullanım: python src/veri_indir.py [--sadece roboflow|visdrone]
"""
import argparse
import os
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham"

# (çalışma alanı, proje, sürüm)
ROBOFLOW_SETLERI = [
    ("akash-gnbdg", "disaster-dnaqz", 2),                                   # insan
    ("damaged-building-detection", "damaged-building-detection-zenbx", 2),  # bina hasarı (vrcqx v8 bunun alt kümesi, alınmaz)
]


def roboflow_indir():
    from roboflow import Roboflow

    anahtar = os.environ.get("ROBOFLOW_API_KEY")
    if not anahtar:
        raise SystemExit("ROBOFLOW_API_KEY ortam değişkeni tanımlı değil.")
    rf = Roboflow(api_key=anahtar)
    for calisma_alani, proje, surum in ROBOFLOW_SETLERI:
        hedef = HAM / proje
        if (hedef / "data.yaml").exists():
            print(f"[atla] {proje} zaten indirilmiş")
            continue
        rf.workspace(calisma_alani).project(proje).version(surum).download("yolov8", location=str(hedef))


def visdrone_indir():
    """VisDrone2019-DET'i indirir ve YOLO biçimine çevirir."""
    from ultralytics.utils.downloads import download

    hedef = HAM / "VisDrone"
    hedef.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    bolumler = ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]
    eksik = [b for b in bolumler if not (hedef / b / "images").exists()]
    if eksik:
        download([url + b + ".zip" for b in eksik], dir=hedef, threads=3)
    for b in bolumler:
        visdrone_yolo_cevir(hedef / b)


def visdrone_yolo_cevir(klasor: Path):
    """VisDrone notasyonlarını YOLO'ya çevirir; yalnızca pedestrian(1) ve people(2) → 0 (person)."""
    from PIL import Image

    (klasor / "labels").mkdir(exist_ok=True)
    for f in (klasor / "annotations").glob("*.txt"):
        g, y = Image.open(klasor / "images" / f.with_suffix(".jpg").name).size
        satirlar = []
        for satir in f.read_text().strip().splitlines():
            p = satir.strip().split(",")
            if len(p) < 6 or p[4] == "0" or p[5] not in ("1", "2"):
                continue
            x, yy, w, h = map(int, p[:4])
            satirlar.append(f"0 {(x + w / 2) / g:.6f} {(yy + h / 2) / y:.6f} {w / g:.6f} {h / y:.6f}")
        (klasor / "labels" / f.name).write_text("\n".join(satirlar) + ("\n" if satirlar else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece", choices=["roboflow", "visdrone"])
    a = ap.parse_args()
    if a.sadece != "visdrone":
        roboflow_indir()
    if a.sadece != "roboflow":
        visdrone_indir()
