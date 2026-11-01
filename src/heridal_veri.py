"""HERIDAL (Hırvatistan dağlık arazi arama-kurtarma, 4000x3000) → veri/heridal_yolo/test: **harici test**.

Hiçbir model bu verilerle eğitilmez; insan uzmanının hiç görmediği bir bölge/kamera/arazide genellemesini ölçer.
Kaynak: Roboflow Universe new-workspace-qsuag/heridal-t428z v2 (CC BY 4.0). Kareler 960 px döşemelere
bölünür (insanlı döşemelerin hepsi + insansızların %10'u).

Kullanım: python src/heridal_veri.py
"""
import random
import shutil
from pathlib import Path

from PIL import Image

from sar_veri import DOSEME, baslangiclar, kutular

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "new-workspace-qsuag__heridal-t428z"
HEDEF = KOK / "veri" / "heridal_yolo" / "test"
BOS_ORAN = 0.1

if __name__ == "__main__":
    rng = random.Random(0)
    shutil.rmtree(HEDEF.parent, ignore_errors=True)
    for alt in ("images", "labels"):
        (HEDEF / alt).mkdir(parents=True)
    n = k = 0
    for yol in sorted(KAYNAK.glob("*/images/*.jpg")):
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
                if not satirlar and rng.random() > BOS_ORAN:
                    continue
                ad = f"{yol.stem.split('.rf.')[0]}_{x0}_{y0}"
                im.crop((x0, y0, x0 + DOSEME, y0 + DOSEME)).save(HEDEF / "images" / f"{ad}.jpg", quality=92)
                (HEDEF / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))
                n += 1; k += len(satirlar)
    print(f"{n} döşeme, {k} insan")
