"""Bina sınıflandırıcısını drone panellerinde değerlendirir (veri/tek_drone_eval_rn = RescueNet val, _cr = CRASAR valid).

  python src/tek_drone_deg.py runs/tek1/weights/best.pt [veri/tek1 ...]   (ek kök verilirse onlar ölçülür)
"""
import sys

from ultralytics import YOLO

if __name__ == "__main__":
    m = YOLO(sys.argv[1])
    for n in (sys.argv[2:] or ["veri/tek_drone_eval_rn", "veri/tek_drone_eval_cr"]):
        r = m.val(data=n, split="val", imgsz=256, batch=64, plots=False, verbose=False,
                  workers=0, project="runs/val", name="c", exist_ok=True)
        cm = r.confusion_matrix.matrix  # satır: tahmin, sütun: gerçek
        sinif = {m.names[i]: round(float(cm[i, i] / max(1, cm[:, i].sum())), 3) for i in range(len(m.names))}
        print(n, "top1", round(r.top1, 3), "sınıf duyarlılığı", sinif, flush=True)
