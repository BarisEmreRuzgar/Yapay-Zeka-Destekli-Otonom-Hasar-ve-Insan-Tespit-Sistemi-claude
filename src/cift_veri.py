"""xBD'den öncesi/sonrası bina kırpıntılarıyla hasar sınıflandırma veri seti üretir.

Her bina (t2_mask'te bir hasar seviyesinin bağlı bileşeni) için afet öncesi ve sonrası
görüntüden aynı bölge bağlamıyla kırpılır ve tek bir 224 px görsele dizilir:
    üst sol: öncesi   üst sağ: sonrası
    alt sol: |sonrası - öncesi| farkı (güçlendirilmiş)   alt sağ: bina maskesi
Böylece standart YOLOv8-cls mimarisi iki zamanı birlikte görür.

Kullanım: python src/cift_veri.py [--sinif-basina 12000] [--test]
"""
import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
SINIFLAR = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}
DISLA = ("hurricane-maria",)  # xbd_cevir.py ile aynı gerekçe
MIN_ALAN = 24
PANEL = 112


def bina_kutulari(maske):
    for deger, ad in SINIFLAR.items():
        ikili = (maske == deger).astype(np.uint8)
        if not ikili.any():
            continue
        _, etiket, ist, _ = cv2.connectedComponentsWithStats(ikili, connectivity=4)
        for i, (x, y, w, h, alan) in enumerate(ist[1:], start=1):
            if alan >= MIN_ALAN:
                yield ad, x, y, w, h, (etiket == i)


def cift_gorsel(once, sonra, x, y, w, h, bina=None):
    """Kutuyu bağlamla genişletip 2x2 panelli 224 px görsel döndürür (np.uint8, RGB)."""
    H, W = once.shape[:2]
    kenar = max(w, h) * 1.6 + 16
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(max(0, cx - kenar / 2)), int(max(0, cy - kenar / 2))
    x2, y2 = int(min(W, cx + kenar / 2)), int(min(H, cy + kenar / 2))
    a = cv2.resize(once[y1:y2, x1:x2], (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    b = cv2.resize(sonra[y1:y2, x1:x2], (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    fark = np.clip(np.abs(b.astype(np.int16) - a.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
    if bina is None:  # çıkarımda maske yok: kutunun kendisi
        bina = np.zeros((H, W), bool)
        bina[int(y):int(y + h), int(x):int(x + w)] = True
    m = cv2.resize(bina[y1:y2, x1:x2].astype(np.uint8) * 255, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    return np.vstack([np.hstack([a, b]), np.hstack([fark, np.dstack([m] * 3)])])


def oku(satir, alan):
    g = np.array(Image.open(io.BytesIO(satir[alan]["bytes"])).convert("RGB"))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinif-basina", type=int, default=12000, help="eğitimde sınıf başına üst sınır")
    ap.add_argument("--valid-oran", type=float, default=0.1)
    ap.add_argument("--test", action="store_true", help="bağımsız xBD testinden (sınırsız)")
    a = ap.parse_args()

    ham = KOK / "veri" / "ham" / ("xbd_test" if a.test else "xbd")
    cikti = KOK / "veri" / ("cift_test" if a.test else "cift")
    sayac = {b: {s: 0 for s in SINIFLAR.values()} for b in ("train", "val", "test")}
    rng = random.Random(0)
    # Hasarsız binalar çok baskın; sınır dolmadan tükenmesin diye her binayı olasılıkla alıyoruz.
    alma = {"no-damage": 0.12, "minor-damage": 1.0, "major-damage": 1.0, "destroyed": 1.0}

    for dosya in sorted(ham.glob("*.parquet")):
        for grup in pq.ParquetFile(dosya).iter_batches(
                batch_size=8, columns=["t1_image", "t2_image", "t2_mask", "image_name"]):
            for satir in grup.to_pylist():
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA):
                    continue
                bolum = "test" if a.test else ("val" if random.Random(ad).random() < a.valid_oran else "train")
                maske = np.array(Image.open(io.BytesIO(satir["t2_mask"]["bytes"])))
                if maske.ndim == 3:
                    maske = maske[..., 0]
                once, sonra = oku(satir, "t1_image"), oku(satir, "t2_image")
                for k, (sinif, x, y, w, h, bina) in enumerate(bina_kutulari(maske)):
                    if not a.test:
                        sinir = a.sinif_basina if bolum == "train" else a.sinif_basina // 8
                        if sayac[bolum][sinif] >= sinir or rng.random() > alma[sinif]:
                            continue
                    hedef = cikti / bolum / sinif
                    hedef.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(cift_gorsel(once, sonra, x, y, w, h, bina)).save(
                        hedef / f"{ad}_{k}.jpg", quality=92)
                    sayac[bolum][sinif] += 1
        print(dosya.name, {b: s for b, s in sayac.items() if any(s.values())}, flush=True)


if __name__ == "__main__":
    main()
