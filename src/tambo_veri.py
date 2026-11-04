"""Universe tiny-person-public/tiny-person-detection-stwdp v6 (CC BY 4.0) → veri/tambo_yolo.

Adına rağmen TinyPerson değil: 8 nehir/kıyı drone videosunun ("Tambo") kareleri, sudaki ve kıyıdaki çok küçük insanlar.
* Roboflow 640x640'a germiş (aslı 16:9 varsayıldı) → 1138x640'a geri açılır.
* Her kareden 3 artırılmış kopya var (yatay çevirme, ±10° döndürme, parlaklık); döndürme siyah köşe bırakır →
  siyah pikseli en az olan kopya (en az döndürülmüş) alınır.
* Roboflow bölmesine güvenilmez (ardışık kareler) → **video bazında** ayrılır:
  valid: New-Tambo-5, 8-17-23-Tambo-2   test: New-Tambo-2   eğitim: kalan 5 video (her 3. kare).

Kullanım: python src/tambo_veri.py
"""
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "tiny-person-public__tiny-person-detection-stwdp"
HEDEF = KOK / "veri" / "tambo_yolo"
BOLUM = {"New-Tambo-5": "valid", "8-17-23-Tambo-2": "valid", "New-Tambo-2": "test"}
BOYUT = (1138, 640)
ADIM = 3  # eğitimde her 3. kare (ardışık kareler neredeyse aynı)


def siyah_oran(yol):
    a = np.asarray(Image.open(yol).convert("L").resize((64, 64)))
    return float((a < 8).mean())


if __name__ == "__main__":
    shutil.rmtree(HEDEF, ignore_errors=True)
    for b in ("train", "valid", "test"):
        for alt in ("images", "labels"):
            (HEDEF / b / alt).mkdir(parents=True)
    kareler = defaultdict(list)
    for yol in KAYNAK.glob("*/images/*.jpg"):
        m = re.match(r"(.+?)-mp4_(\d+)_jpg\.rf\.", yol.name)
        if m:
            kareler[(m.group(1), int(m.group(2)))].append(yol)
    sayac = defaultdict(lambda: [0, 0])
    for (video, no), kopyalar in sorted(kareler.items()):
        b = BOLUM.get(video, "train")
        if b == "train" and no % ADIM:
            continue
        yol = min(kopyalar, key=siyah_oran)
        etiket = yol.parent.parent / "labels" / (yol.stem + ".txt")
        satirlar = [s for s in etiket.read_text().splitlines() if len(s.split()) == 5]
        ad = f"tambo_{video}_{no:05d}"
        Image.open(yol).convert("RGB").resize(BOYUT, Image.BILINEAR).save(HEDEF / b / "images" / f"{ad}.jpg", quality=95)
        (HEDEF / b / "labels" / f"{ad}.txt").write_text("\n".join(satirlar))  # normalize koordinatlar germeden etkilenmez
        sayac[b][0] += 1; sayac[b][1] += len(satirlar)
    for b, (n, k) in sorted(sayac.items()):
        print(f"{b}: {n} kare, {k} insan")
