"""Bir veya daha fazla modeli aynı test setinde değerlendirir ve sınıf bazlı tablo basar.

Kullanım: python src/degerlendir.py runs/temel/weights/best.pt runs/genis/weights/best.pt
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

KOK = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modeller", nargs="+")
    ap.add_argument("--veri", default="veri/birlesik/data.yaml")
    ap.add_argument("--bolum", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    sonuclar = {}
    for m in a.modeller:
        r = YOLO(m).val(data=str(KOK / a.veri), split=a.bolum, imgsz=a.imgsz, batch=8,
                        plots=False, verbose=False, project=str(KOK / "runs" / "val"), name="v", exist_ok=True)
        adlar = r.names
        sinif = {adlar[int(c)]: {"AP50": float(r.box.ap50[i]), "AP50-95": float(r.box.ap[i])}
                 for i, c in enumerate(r.box.ap_class_index)}
        sonuclar[m] = {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map), "sinif": sinif}

    print(f"\n{'model':45} {'mAP50':>6} {'mAP50-95':>8} | " + " ".join(f"{s[:9]:>9}" for s in adlar.values()))
    for m, s in sonuclar.items():
        print(f"{m[-45:]:45} {s['mAP50']:6.3f} {s['mAP50-95']:8.3f} | "
              + " ".join(f"{s['sinif'].get(ad, {}).get('AP50', float('nan')):9.3f}" for ad in adlar.values()))
    (KOK / "runs" / "degerlendirme.json").write_text(json.dumps(sonuclar, indent=2, ensure_ascii=False))
