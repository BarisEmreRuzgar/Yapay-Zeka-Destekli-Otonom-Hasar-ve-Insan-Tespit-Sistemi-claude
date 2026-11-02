"""Sudaki insanlar (sel / su baskını senaryosu) → veri/su_yolo/<bölüm>: 960 px döşemeler, yalnız `swimmer` → insan.

Kaynak: Roboflow Universe sigmoidal-ivbj6/search-and-rescue-y6x1c v1 (kamu malı), 2728x1816 deniz/göl kareleri.
Roboflow her kareyi 4 artırılmış kopyayla vermiş; her kareden yalnız bir kopya alınır.
Roboflow bölmesi rastgele kare bazlı: test karelerinin %84'ünün ±1 komşu karesi eğitimde (ardışık video kareleri → sızıntı).
Bu yüzden üç bölüm birleştirilip **kare numarası bloklarına göre** yeniden ayrılır; blok kenarlarında 30 kare tampon atılır.

Kullanım: python src/su_veri.py [test valid train]
"""
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

from sar_veri import DOSEME, baslangiclar

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "sigmoidal-ivbj6__search-and-rescue-y6x1c"
SWIMMER = 4
BOS_ORAN = 0.1
# kare numarası aralıkları (numaralar 2–16731); kalan her şey eğitim
BLOKLAR = {"valid": [(3000, 4000), (11000, 12000)], "test": [(7000, 8000), (14000, 15000)]}
TAMPON = 30


def bolumu(no):
    for b, araliklar in BLOKLAR.items():
        if any(a <= no < z for a, z in araliklar):
            return b
    if any(a - TAMPON <= no < z + TAMPON for ar in BLOKLAR.values() for a, z in ar):
        return None  # tampon
    return "train"


def tum_kareler():
    tekil = {}
    for yol in sorted(KAYNAK.glob("*/images/*.jpg")):
        tekil.setdefault(int(yol.name.split("_jpg.rf.")[0]), yol)
    return tekil


def kutular(etiket, w, h):
    for satir in etiket.read_text().splitlines():
        p = satir.split()
        if len(p) == 5 and int(p[0]) == SWIMMER:
            cx, cy, bw, bh = (float(x) for x in p[1:])
            yield (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


def bolum_hazirla(bolum, rng):
    hedef = KOK / "veri" / "su_yolo" / bolum
    shutil.rmtree(hedef, ignore_errors=True)
    for alt in ("images", "labels"):
        (hedef / alt).mkdir(parents=True)
    tekil = {no: y for no, y in tum_kareler().items() if bolumu(no) == bolum}
    n = k = 0
    for kok_ad, yol in tekil.items():
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
                ad = f"su_{kok_ad}_{x0}_{y0}"
                im.crop((x0, y0, x0 + DOSEME, y0 + DOSEME)).save(hedef / "images" / f"{ad}.jpg", quality=92)
                (hedef / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))
                n += 1; k += len(satirlar)
    print(f"{bolum}: {len(tekil)} kare → {n} döşeme, {k} insan", flush=True)


if __name__ == "__main__":
    rng = random.Random(0)
    for b in sys.argv[1:] or ["test"]:
        bolum_hazirla(b, rng)
