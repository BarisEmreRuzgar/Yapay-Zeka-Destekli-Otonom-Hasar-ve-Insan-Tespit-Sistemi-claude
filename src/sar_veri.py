"""Arama-kurtarma (SAR) drone videolarından insan döşemeleri üretir → veri/sar_yolo/{train,valid}.

Kaynak: Roboflow Universe drone-leak/search-and-rescue-8ywn7 (CC BY 4.0, 3840x2160 video kareleri).
Roboflow'un kendi bölmesi kareleri rastgele dağıttığı için ardışık kareler train/valid arasında
neredeyse aynı; bu yüzden ayrım **video bazında** yapılır (VAL_VIDEO hiç eğitimde yok).
Kareler VisDrone döşemeleriyle aynı ölçekte 960 px'lik döşemelere bölünür.

Kullanım: python src/sar_veri.py
"""
import random
import re
import shutil
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "drone-leak__search-and-rescue-8ywn7"
HEDEF = KOK / "veri" / "sar_yolo"
VAL_VIDEO = {"45DEG_150FT_MP4", "BIV_RIVER_100FT_MP4", "NEGATIVE-_100FT_MP4"}
DOSEME = 960
BOS_ORAN = {"train": 0.1, "valid": 0.25}  # insansız döşemelerden tutulacak pay
NEGATIF_SINIR = 150  # uzun NEGATIVE_1/2 videolarından alınacak en çok kare (eğitimde)


def baslangiclar(uzunluk):
    n = -(-uzunluk // DOSEME)
    return [round(i * (uzunluk - DOSEME) / max(n - 1, 1)) for i in range(n)]


def kutular(etiket, w, h):
    for satir in etiket.read_text().splitlines():
        p = satir.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(x) for x in p[1:])
            yield (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


if __name__ == "__main__":
    rng = random.Random(0)
    shutil.rmtree(HEDEF, ignore_errors=True)
    kareler = sorted(KAYNAK.glob("*/images/*.jpg"))
    sayac = {}
    negatif = {}
    for yol in kareler:
        video = re.sub(r"-\d+_jpg\.rf\..*", "", yol.name)
        bolum = "valid" if video in VAL_VIDEO else "train"
        if bolum == "train" and video.startswith("NEGATIVE_"):
            negatif[video] = negatif.get(video, 0) + 1
            if negatif[video] > NEGATIF_SINIR:
                continue
        im = Image.open(yol)
        w, h = im.size
        kk = list(kutular(yol.parent.parent / "labels" / (yol.stem + ".txt"), w, h))
        for x0 in baslangiclar(w):
            for y0 in baslangiclar(h):
                satirlar = []
                for a, b, c, d in kk:
                    ka, kb, kc, kd = max(a, x0), max(b, y0), min(c, x0 + DOSEME), min(d, y0 + DOSEME)
                    if kc <= ka or kd <= kb or (kc - ka) * (kd - kb) < 0.5 * (c - a) * (d - b):
                        continue
                    satirlar.append(f"0 {((ka + kc) / 2 - x0) / DOSEME:.6f} {((kb + kd) / 2 - y0) / DOSEME:.6f} "
                                    f"{(kc - ka) / DOSEME:.6f} {(kd - kb) / DOSEME:.6f}")
                if not satirlar and rng.random() > BOS_ORAN[bolum]:
                    continue
                ad = f"{yol.stem.split('.rf.')[0]}_{x0}_{y0}"
                (HEDEF / bolum / "images").mkdir(parents=True, exist_ok=True)
                (HEDEF / bolum / "labels").mkdir(parents=True, exist_ok=True)
                im.crop((x0, y0, x0 + DOSEME, y0 + DOSEME)).save(HEDEF / bolum / "images" / f"{ad}.jpg", quality=92)
                (HEDEF / bolum / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))
                s = sayac.setdefault(bolum, [0, 0])
                s[0] += 1; s[1] += len(satirlar)
    print({b: f"{n} döşeme, {k} insan" for b, (n, k) in sayac.items()})
