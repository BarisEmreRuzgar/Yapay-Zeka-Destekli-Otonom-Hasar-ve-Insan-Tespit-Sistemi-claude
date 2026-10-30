"""YOLOv8 eğitimi (6 GB GPU için ayarlı).

Kullanım:
  python src/egit.py --ad temel --veri veri/temel/data.yaml
  python src/egit.py --ad genis --veri veri/birlesik/data.yaml --model yolov8s.pt --epoch 60
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

KOK = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad", required=True)
    ap.add_argument("--veri", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epoch", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2, help="veri yükleyici işçi sayısı (RAM'e göre)")
    ap.add_argument("--devam", action="store_true", help="yarıda kalan eğitimi sürdür")
    a = ap.parse_args()

    if a.devam:
        model = YOLO(KOK / "runs" / a.ad / "weights" / "last.pt")
        model.train(resume=True, workers=a.workers)
    else:
        model = YOLO(a.model)
        model.train(
            data=str(KOK / a.veri), epochs=a.epoch, imgsz=a.imgsz, batch=a.batch,
            patience=20, workers=a.workers, cache=False, cos_lr=True, close_mosaic=10,
            project=str(KOK / "runs"), name=a.ad, exist_ok=True, seed=0,
            # havadan görüntü: yukarıdan bakışta dikey çevirme anlamlı
            flipud=0.5, degrees=10.0,
        )
