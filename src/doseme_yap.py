"""Büyük kareli YOLO veri setini kare döşemelere böler (değerlendirme/eğitim için).

Kutular: merkezi döşemede olan kutu kırpılarak alınır; döşemede %40'tan azı kalan kutu atlanır.
Kullanım: python src/doseme_yap.py KAYNAK(images/,labels/) HEDEF [--boyut 960] [--ortusme 0.15]
"""
import argparse
import random
from pathlib import Path

import cv2


def baslangiclar(uzunluk, boyut, adim):
    if uzunluk <= boyut:
        return [0]
    s = list(range(0, uzunluk - boyut, adim)) + [uzunluk - boyut]
    return sorted(set(s))


def dosele(kaynak, hedef, boyut=960, ortusme=0.15, bos_oran=1.0):
    """bos_oran < 1: kutusuz döşemelerin yalnız bu oranı tutulur (eğitimde negatif fazlalığını azaltmak için)."""
    rnd = random.Random(0)
    (hedef / "images").mkdir(parents=True, exist_ok=True)
    (hedef / "labels").mkdir(parents=True, exist_ok=True)
    adim = int(boyut * (1 - ortusme))
    n = k = 0
    for im_yol in sorted((kaynak / "images").iterdir()):
        et = kaynak / "labels" / (im_yol.stem + ".txt")
        im = cv2.imread(str(im_yol))
        if im is None:
            continue
        H, W = im.shape[:2]
        kutular = []
        if et.exists():
            for s in et.read_text().split("\n"):
                p = s.split()
                if len(p) != 5:
                    continue
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
                kutular.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
        for y0 in baslangiclar(H, boyut, adim):
            for x0 in baslangiclar(W, boyut, adim):
                tw, th = min(boyut, W), min(boyut, H)
                satir = []
                for c, x1, y1, x2, y2 in kutular:
                    a1, b1, a2, b2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
                    if a2 <= a1 or b2 <= b1:
                        continue
                    if (a2 - a1) * (b2 - b1) < 0.4 * (x2 - x1) * (y2 - y1):
                        continue
                    satir.append(f"{c} {((a1 + a2) / 2 - x0) / tw:.6f} {((b1 + b2) / 2 - y0) / th:.6f} "
                                 f"{(a2 - a1) / tw:.6f} {(b2 - b1) / th:.6f}")
                if not satir and rnd.random() >= bos_oran:
                    continue
                ad = f"{im_yol.stem}_{x0}_{y0}"
                cv2.imwrite(str(hedef / "images" / f"{ad}.jpg"), im[y0:y0 + th, x0:x0 + tw], [cv2.IMWRITE_JPEG_QUALITY, 92])
                (hedef / "labels" / f"{ad}.txt").write_text("\n".join(satir))
                n += 1; k += len(satir)
    print(f"{hedef}: {n} döşeme, {k} kutu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak", type=Path)
    ap.add_argument("hedef", type=Path)
    ap.add_argument("--boyut", type=int, default=960)
    ap.add_argument("--ortusme", type=float, default=0.15)
    a = ap.parse_args()
    dosele(a.kaynak, a.hedef, a.boyut, a.ortusme)
