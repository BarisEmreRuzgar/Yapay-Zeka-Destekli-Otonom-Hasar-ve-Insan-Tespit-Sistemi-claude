"""YOLO biçimli drone hasar döşemelerinden (RescueNet, CRASAR) bina bazlı 'tek' (yalnız sonrası) sınıflandırıcı panelleri.

Panel düzeni cift_veri.tek_gorsel ile aynı (geniş bağlam | yakın plan | kenar haritası | kutu maskesi), böylece
xBD tek1 verisiyle birlikte aynı sınıflandırıcıya verilebilir. Eksik etiketli kaynaklarda (CRASAR) bile her panel
yalnız etiketli bir binayı içerdiği için sınıflandırma eksik etiketten etkilenmez.

  python src/drone_tek_veri.py --kaynak veri/rescuenet_yolo/train --cikti veri/tek_drone/train
  python src/drone_tek_veri.py --kaynak veri/crasar_yolo/valid --cikti veri/tek_drone/val --oynat 0
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from cift_veri import oynat, tek_gorsel

KOK = Path(__file__).resolve().parent.parent
ADLAR = {1: "destroyed", 2: "major-damage", 3: "minor-damage", 4: "no-damage"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaynak", nargs="+", required=True, help="images/ ve labels/ içeren klasör(ler)")
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--panel", type=int, default=128)
    ap.add_argument("--oynat", type=float, default=0.15, help="kutu kaydırma oranı (eğitimde konum hatası taklidi)")
    ap.add_argument("--min-kenar", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(0)
    cikti = KOK / a.cikti
    sayac = {v: 0 for v in ADLAR.values()}
    for kaynak in a.kaynak:
        k = KOK / kaynak
        for e in sorted((k / "labels").glob("*.txt")):
            satirlar = [s.split() for s in e.read_text().splitlines() if s.strip()]
            satirlar = [s for s in satirlar if int(s[0]) in ADLAR]
            if not satirlar:
                continue
            g = cv2.imread(str(k / "images" / (e.stem + ".jpg")))
            if g is None:
                continue
            g = g[..., ::-1]
            H, W = g.shape[:2]
            for i, (c, x, y, w, h) in enumerate(satirlar):
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                if max(w, h) < a.min_kenar:
                    continue
                x0, y0 = x - w / 2, y - h / 2
                if a.oynat:
                    x0, y0, w, h = oynat(x0, y0, w, h, rng, a.oynat)
                p = tek_gorsel(g, x0, y0, w, h, panel=a.panel)
                ad = ADLAR[int(c)]
                (cikti / ad).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cikti / ad / f"{e.stem}_{i}.jpg"), p[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                sayac[ad] += 1
    print(sayac)
