"""Öncesi/sonrası panel görselleriyle hasar sınıflandırıcısı (YOLOv8s-cls) eğitir.

Paneller konuma bağlı (sol=öncesi, sağ=sonrası) olduğu için yatay çevirme ve rastgele
kırpma kapalı; dikey çevirme her paneli kendi içinde değil tüm ızgarayı çevirir, o da kapalı.

Kullanım: python src/cift_egit.py [--epoch 30] [--batch 64] [--model yolov8m-cls.pt --veri cift2 --imgsz 256]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

KOK = Path(__file__).resolve().parent.parent

if __name__ == "__main__":  # Windows'ta veri yükleyici süreçleri modülü yeniden içe aktarır
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ad", default="cift")
    ap.add_argument("--model", default="yolov8s-cls.pt")
    ap.add_argument("--veri", default="cift", help="veri/ altındaki klasör")
    ap.add_argument("--imgsz", type=int, default=224, help="cift_veri --panel değerinin iki katı olmalı")
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()

    YOLO(a.model).train(
        data=str(KOK / "veri" / a.veri), imgsz=a.imgsz, epochs=a.epoch, batch=a.batch, workers=2,
        project=str(KOK / "runs"), name=a.ad, exist_ok=True, patience=8, dropout=a.dropout,
        fliplr=0.0, flipud=0.0, scale=0.0, translate=0.0, auto_augment=None, erasing=0.0,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
    )
