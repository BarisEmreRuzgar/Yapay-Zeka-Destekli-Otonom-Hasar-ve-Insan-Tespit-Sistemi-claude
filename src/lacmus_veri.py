"""Lacmus / LADD (yaban arazisi arama-kurtarma, Roboflow Universe 2-8hh1e/lacmus v4, CC BY 4.0) → veri/lacmus_yolo.

* Roboflow kareleri 2048x2048'e GERMİŞ (aslı 4:3) → 2048x1536'ya geri açılır (SARD'daki gibi).
* Her kaynak görüntüden 3 artırılmış kopya var (pozlama ±%4, tuz-biber; kutular aynı) → aynı numara + aynı etiket
  içeriği = aynı kaynak, yalnız biri alınır. Aynı numara ~4 farklı kaynakta geçiyor (hepsi kullanılır).
* Roboflow bölmesine güvenilmez (ardışık kareler); numara bloklarıyla ayrılır, kenarlarda tampon atılır.
* 960 px döşemeler (insanlı döşemelerin hepsi + insansızların %10'u).

Kullanım: python src/lacmus_veri.py
"""
import random
import shutil
from pathlib import Path

from PIL import Image

from sar_veri import DOSEME, baslangiclar

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "2-8hh1e__lacmus"
HEDEF = KOK / "veri" / "lacmus_yolo"
BLOKLAR = {"valid": [(300, 400), (900, 1000)], "test": [(600, 700), (1200, 1300)]}
TAMPON = 10
BOYUT = (2048, 1536)
BOS_ORAN = 0.1


def bolumu(no):
    for b, ar in BLOKLAR.items():
        if any(a <= no < z for a, z in ar):
            return b
    if any(a - TAMPON <= no < z + TAMPON for ar in BLOKLAR.values() for a, z in ar):
        return None
    return "train"


def kutular(etiket, w, h):
    for satir in etiket.read_text().splitlines():
        p = satir.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(x) for x in p[1:])
            yield (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


if __name__ == "__main__":
    rng = random.Random(0)
    shutil.rmtree(HEDEF, ignore_errors=True)
    for b in ("train", "valid", "test"):
        for alt in ("images", "labels"):
            (HEDEF / b / alt).mkdir(parents=True)
    gorulen = set()
    sayac = {b: [0, 0, 0] for b in ("train", "valid", "test")}
    for yol in sorted(KAYNAK.glob("*/images/*.jpg")):
        no = int(yol.name.split("_")[0])
        b = bolumu(no)
        if b is None:
            continue
        etiket = yol.parent.parent / "labels" / (yol.stem + ".txt")
        anahtar = (no, etiket.read_text().strip())
        if anahtar in gorulen:  # aynı kaynağın artırılmış kopyası
            continue
        gorulen.add(anahtar)
        im = Image.open(yol).convert("RGB").resize(BOYUT, Image.BILINEAR)
        w, h = BOYUT
        kk = list(kutular(etiket, w, h))
        sayac[b][0] += 1
        for x0 in baslangiclar(w):
            for y0 in baslangiclar(h):
                satirlar = []
                for a, bb, c, d in kk:
                    ka, kb, kc, kd = max(a, x0), max(bb, y0), min(c, x0 + DOSEME), min(d, y0 + DOSEME)
                    if kc <= ka or kd <= kb or (kc - ka) * (kd - kb) < 0.5 * (c - a) * (d - bb):
                        continue
                    satirlar.append(f"0 {((ka + kc) / 2 - x0) / DOSEME:.6f} {((kb + kd) / 2 - y0) / DOSEME:.6f} "
                                    f"{(kc - ka) / DOSEME:.6f} {(kd - kb) / DOSEME:.6f}")
                if not satirlar and rng.random() > BOS_ORAN:
                    continue
                ad = f"lacmus_{yol.stem.split('_jpg.rf.')[0]}_{yol.stem[-6:]}_{x0}_{y0}"
                im.crop((x0, y0, x0 + DOSEME, y0 + DOSEME)).save(HEDEF / b / "images" / f"{ad}.jpg", quality=92)
                (HEDEF / b / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))
                sayac[b][1] += 1; sayac[b][2] += len(satirlar)
    for b, (k, n, i) in sayac.items():
        print(f"{b}: {k} kaynak görüntü → {n} döşeme, {i} insan")
