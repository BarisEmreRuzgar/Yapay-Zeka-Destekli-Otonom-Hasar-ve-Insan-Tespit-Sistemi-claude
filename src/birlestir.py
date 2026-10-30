"""Tüm kaynakları tek bir 5 sınıflı YOLO veri setinde birleştirir → veri/birlesik/.

Sınıf kimlikleri her kaynağın data.yaml'ındaki ADLARA göre eşlenir (körlemesine +1 kaydırma yok).
Test bölümü yalnızca orijinal Roboflow test setlerinden oluşur; böylece eski modelle
aynı test üzerinde karşılaştırma yapılabilir.

Kullanım: python src/birlestir.py [--visdrone-train 3000]
  Orijinal düzen (karşılaştırma için): --hedef temel --no-xbd --visdrone-train 0 --visdrone-valid 0
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import yaml

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham"
HEDEF = KOK / "veri" / "birlesik"

SINIFLAR = ["person", "destroyed", "major-damage", "minor-damage", "no-damage"]
# Not: damaged-buildings-vrcqx (v8) bu setin alt kümesi (aynı döşemeler, aynı bölümler);
# orijinal defter ikisini birden ekleyip aynı görüntüleri iki kez eğitiyordu.
ROBOFLOW = {"disaster-dnaqz": "rf_insan", "damaged-building-detection-zenbx": "rf_bina"}
VISDRONE = {"train": "train", "valid": "valid"}  # src/visdrone_dosele.py çıktısı


def bagla(kaynak: Path, hedef: Path):
    """Aynı diskte kopya yerine sabit bağlantı (yer kaplamaz)."""
    try:
        os.link(kaynak, hedef)
    except OSError:
        shutil.copy2(kaynak, hedef)


def ekle(img_dir: Path, lbl_dir: Path, bolum: str, onek: str, esleme: dict, secim=None):
    sayac = [0] * len(SINIFLAR)
    resimler = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if secim is not None:
        resimler = secim(resimler)
    for r in resimler:
        lbl = lbl_dir / (r.stem + ".txt")
        satirlar = []
        if lbl.exists():
            for s in lbl.read_text().splitlines():
                p = s.split()
                if len(p) >= 5 and int(p[0]) in esleme:
                    yeni = esleme[int(p[0])]
                    satirlar.append(" ".join([str(yeni)] + p[1:5]))
                    sayac[yeni] += 1
        bagla(r, HEDEF / bolum / "images" / f"{onek}_{r.name}")
        (HEDEF / bolum / "labels" / f"{onek}_{r.stem}.txt").write_text(
            "\n".join(satirlar) + ("\n" if satirlar else ""))
    return len(resimler), sayac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visdrone-train", type=int, default=3000, help="VisDrone train'den alınacak görüntü sayısı")
    ap.add_argument("--visdrone-valid", type=int, default=300)
    ap.add_argument("--xbd", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hedef", default="birlesik", help="veri/ altındaki çıktı klasörü")
    a = ap.parse_args()
    global HEDEF
    HEDEF = KOK / "veri" / a.hedef

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    for b in ("train", "valid", "test"):
        (HEDEF / b / "images").mkdir(parents=True)
        (HEDEF / b / "labels").mkdir(parents=True)

    rapor = []
    for klasor, onek in ROBOFLOW.items():
        adlar = yaml.safe_load((HAM / klasor / "data.yaml").read_text())["names"]
        if isinstance(adlar, dict):
            adlar = [adlar[i] for i in sorted(adlar)]
        esleme = {i: SINIFLAR.index(ad) for i, ad in enumerate(adlar)}
        for b in ("train", "valid", "test"):
            if (HAM / klasor / b / "images").exists():
                rapor.append((onek, b, *ekle(HAM / klasor / b / "images", HAM / klasor / b / "labels", b, onek, esleme)))

    # VisDrone: yalnızca insan içeren görüntülerden rastgele örneklem
    for b, klasor in VISDRONE.items():
        kok = KOK / "veri" / "visdrone_yolo" / klasor
        if not (kok / "labels").exists():
            print("[uyarı] VisDrone yok:", kok)
            continue
        adet = a.visdrone_train if b == "train" else a.visdrone_valid
        if adet == 0:
            continue

        def sec(resimler, adet=adet, kok=kok):
            dolu = [r for r in resimler if (kok / "labels" / (r.stem + ".txt")).stat().st_size > 0]
            random.Random(0).shuffle(dolu)
            return dolu[:adet]

        rapor.append(("visdrone", b, *ekle(kok / "images", kok / "labels", b, "vd", {0: 0}, sec)))

    if a.xbd:
        for b in ("train", "valid"):
            kok = KOK / "veri" / "xbd_yolo" / b
            if (kok / "images").exists():
                rapor.append(("xbd", b, *ekle(kok / "images", kok / "labels", b, "x", {i: i for i in range(5)})))

    (HEDEF / "data.yaml").write_text(yaml.safe_dump({
        "path": str(HEDEF), "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": dict(enumerate(SINIFLAR))}, sort_keys=False, allow_unicode=True))

    print(f"{'kaynak':10} {'bölüm':6} {'görüntü':>7} " + " ".join(f"{s[:9]:>9}" for s in SINIFLAR))
    for kaynak, b, n, say in rapor:
        print(f"{kaynak:10} {b:6} {n:7d} " + " ".join(f"{x:9d}" for x in say))


if __name__ == "__main__":
    main()
