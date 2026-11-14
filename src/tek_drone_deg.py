from ultralytics import YOLO
import sys
m = YOLO(sys.argv[1])
for n in ("rn", "cr"):
    r = m.val(data=f"veri/tek_drone_eval_{n}", split="val", imgsz=256, batch=64, plots=False, verbose=False,
              project="runs/val", name="c", exist_ok=True)
    cm = r.confusion_matrix.matrix
    sinif = {m.names[i]: round(float(cm[i, i] / max(1, cm[:, i].sum())), 3) for i in range(len(m.names))}
    print(n, "top1", round(r.top1, 3), sinif, flush=True)