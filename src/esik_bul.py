"""Bir modelin doğrulama setlerinde F1'i en yüksek yapan güven eşiğini bulur (varsayılan conf seçimi için).

Kullanım: python src/esik_bul.py modeller/insan_uzman.pt veri/insan.yaml veri/sar_val_insan.yaml [--cihaz cpu]
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

KOK = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("veriler", nargs="+")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--cihaz", default=None)
    a = ap.parse_args()
    model = YOLO(a.model)
    for veri in a.veriler:
        r = model.val(data=str(KOK / veri), split="val", imgsz=a.imgsz, batch=4, device=a.cihaz, plots=False,
                      verbose=False, project=str(KOK / "runs" / "val"), name="esik", exist_ok=True)
        px, f1 = r.box.px, r.box.f1_curve[0]
        p, rc = r.box.p_curve[0], r.box.r_curve[0]
        i = int(np.argmax(f1))
        satir = " ".join(f"{c:.2f}:F1={f1[np.searchsorted(px, c)]:.3f}" for c in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3))
        print(f"{veri}: en iyi conf={px[i]:.2f} F1={f1[i]:.3f} P={p[i]:.3f} R={rc[i]:.3f} | {satir}", flush=True)
