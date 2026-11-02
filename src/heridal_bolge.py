"""HERIDAL'i **bölgeye göre** eğitim / test olarak ayırır (dosya adındaki 3 harfli bölge kodu).

Önceden HERIDAL tamamen harici testti (uzman v2: 0.628). Dağ/orman alanını eğitime katmak için bölgelerin bir kısmı
eğitime alınır; kalan bölgeler hiç görülmeyen **harici test** olarak kalır (aynı bölgenin kareleri benzer, karışırsa sızıntı).
  eğitim: ZRI, VRD, JAS, BRA, BRS, SB   (829 kare / 1.866 insan)
  test  : GOR, MED, BRK, TRS, BLA, CAP, RAK (717 kare / 1.026 insan)
Çıktı: veri/heridal_bolge/{train,test} (960 px döşemeler; insanlı döşemelerin hepsi + insansızların %10'u).

Kullanım: python src/heridal_bolge.py
"""
import random
import shutil
from pathlib import Path

from PIL import Image

from sar_veri import DOSEME, baslangiclar, kutular

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "new-workspace-qsuag__heridal-t428z"
HEDEF = KOK / "veri" / "heridal_bolge"
EGITIM = {"ZRI", "VRD", "JAS", "BRA", "BRS", "SB"}
BOS_ORAN = 0.1

if __name__ == "__main__":
    rng = random.Random(0)
    shutil.rmtree(HEDEF, ignore_errors=True)
    for b in ("train", "test"):
        for alt in ("images", "labels"):
            (HEDEF / b / alt).mkdir(parents=True)
    sayac = {"train": [0, 0], "test": [0, 0]}
    for yol in sorted(KAYNAK.glob("*/images/*.jpg")):
        bolge = yol.name.split("_")[1]
        b = "train" if bolge in EGITIM else "test"
        im = Image.open(yol)
        w, h = im.size
        kk = list(kutular(yol.parent.parent / "labels" / (yol.stem + ".txt"), w, h))
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
                ad = f"heridal_{yol.stem.split('.rf.')[0]}_{x0}_{y0}"
                im.crop((x0, y0, x0 + DOSEME, y0 + DOSEME)).save(HEDEF / b / "images" / f"{ad}.jpg", quality=92)
                (HEDEF / b / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))
                sayac[b][0] += 1; sayac[b][1] += len(satirlar)
    for b, (n, k) in sayac.items():
        print(f"{b}: {n} döşeme, {k} insan")
