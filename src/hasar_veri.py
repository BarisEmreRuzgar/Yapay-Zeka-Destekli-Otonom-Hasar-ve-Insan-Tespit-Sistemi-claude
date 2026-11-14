"""Yalnız hasar sınıflarını öğrenen 'hasar uzmanı' için veri seti.

birlesik/train içinden VisDrone (vd_) ve Roboflow insan (rf_insan) döşemeleri çıkarılır, kalan etiketlerden
insan (sınıf 0) silinir. Sınıf numaraları ana modelle aynı kalır (0 kullanılmaz) → tahmin.py uyumlu.
Doğrulama: xbd_yolo/valid (544 döşeme; hasar mAP50 ana model @640 = 0.410).
Görüntüler sabit bağlantıyla (hard link) eklenir, disk harcanmaz.

  python src/hasar_veri.py --cikti veri/hasar_v1
"""
import argparse
import os
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
ADLAR = {0: "person", 1: "destroyed", 2: "major-damage", 3: "minor-damage", 4: "no-damage"}


def bagla(kaynak: Path, hedef: Path):
    if hedef.exists():
        return
    try:
        os.link(kaynak, hedef)
    except OSError:
        import shutil
        shutil.copy2(kaynak, hedef)


def kopyala(goruntuler: Path, etiketler: Path, cikti: Path, bolum: str, sec):
    (cikti / bolum / "images").mkdir(parents=True, exist_ok=True)
    (cikti / bolum / "labels").mkdir(parents=True, exist_ok=True)
    n = kutu = 0
    for g in sorted(goruntuler.iterdir()):
        if not sec(g.name):
            continue
        e = etiketler / (g.stem + ".txt")
        satirlar = []
        if e.exists():
            satirlar = [s for s in e.read_text().splitlines() if s.strip() and not s.startswith("0 ")]
        bagla(g, cikti / bolum / "images" / g.name)
        (cikti / bolum / "labels" / (g.stem + ".txt")).write_text("\n".join(satirlar) + ("\n" if satirlar else ""))
        n += 1
        kutu += len(satirlar)
    print(f"{bolum}: {n} görüntü, {kutu} hasar kutusu")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cikti", default="veri/hasar_v1")
    a = ap.parse_args()
    cikti = KOK / a.cikti
    b = KOK / "veri" / "birlesik" / "train"
    kopyala(b / "images", b / "labels", cikti, "train",
            lambda ad: not ad.startswith("vd_") and not ad.startswith("rf_insan"))
    v = KOK / "veri" / "xbd_yolo" / "valid"
    kopyala(v / "images", v / "labels", cikti, "valid", lambda ad: True)
    (cikti / "data.yaml").write_text(
        f"path: {cikti.as_posix()}\ntrain: train/images\nval: valid/images\ntest: valid/images\n"
        f"names: {ADLAR}\n", encoding="utf-8")
