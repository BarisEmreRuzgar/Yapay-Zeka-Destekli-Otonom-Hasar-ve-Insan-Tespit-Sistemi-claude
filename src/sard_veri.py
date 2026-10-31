"""SARD (arama-kurtarma: moloz/kaya/tarlada kısmen gizli insanlar) → veri/sard_yolo/{train,valid}.

Kaynak: Roboflow Universe sandr/drone-aerial v1 (CC BY 4.0). Kareler tek bir uzun videodan (gss1..gss2104)
ve Roboflow 640x640'a **gererek** küçültmüş; burada 16:9'a geri açılır (1138x640, etiketler normalize
olduğu için değişmez). Roboflow'un rastgele bölmesi ardışık kareleri train/valid'e dağıttığından ayrım
kare numarasına göre: gss >= VAL_BAS doğrulama, aradaki TAMPON kare atılır.

Kullanım: python src/sard_veri.py
"""
import re
import shutil
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "veri" / "ham" / "sar" / "sandr__drone-aerial"
HEDEF = KOK / "veri" / "sard_yolo"
VAL_BAS, TAMPON = 1800, 50
BOYUT = (1138, 640)

if __name__ == "__main__":
    shutil.rmtree(HEDEF, ignore_errors=True)
    goruldu = set()
    sayac = {}
    for yol in sorted(KAYNAK.glob("*/images/*.jpg")):
        kaynak = re.sub(r"_(png|jpg)\.rf\..*", "", yol.stem)
        m = re.fullmatch(r"gss(\d+)", kaynak)
        if m is None:
            bolum = "train"
        else:
            n = int(m.group(1))
            if VAL_BAS - TAMPON <= n < VAL_BAS:
                continue
            bolum = "valid" if n >= VAL_BAS else "train"
        if bolum == "valid":  # doğrulamada her kareden tek kopya (eğitim kopyaları Roboflow artırması)
            if kaynak in goruldu:
                continue
            goruldu.add(kaynak)
        etiket = yol.parent.parent / "labels" / (yol.stem + ".txt")
        satirlar = [s for s in etiket.read_text().splitlines() if len(s.split()) == 5]
        for alt in ("images", "labels"):
            (HEDEF / bolum / alt).mkdir(parents=True, exist_ok=True)
        Image.open(yol).convert("RGB").resize(BOYUT, Image.BICUBIC).save(HEDEF / bolum / "images" / yol.name, quality=95)
        (HEDEF / bolum / "labels" / (yol.stem + ".txt")).write_text("\n".join(satirlar))
        s = sayac.setdefault(bolum, [0, 0])
        s[0] += 1; s[1] += len(satirlar)
    print({b: f"{n} görüntü, {k} insan" for b, (n, k) in sayac.items()})
