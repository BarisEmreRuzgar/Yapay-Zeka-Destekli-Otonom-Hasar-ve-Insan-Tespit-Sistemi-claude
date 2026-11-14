"""RescueNet (Hurricane Michael, UAV, 4000x3000) segmentasyon maskelerinden YOLO hasar kutuları.

Maske sınıfları: 2 Building_No_Damage, 3 Minor, 4 Major, 5 Total_Destruction (diğerleri yok sayılır).
Her hasar seviyesinin bağlı bileşenleri kutuya çevrilir; bitişik binaların tek kutuda birleşmemesi için bileşenler
hafif aşındırılıp (erosion) bulunur, kutu sonra aşındırma payı kadar genişletilir. Sınıf numaraları ana modelle aynı:
1 destroyed, 2 major-damage, 3 minor-damage, 4 no-damage (0 insan, kullanılmaz).

Görüntüler uzun kenar --boyut px olacak şekilde küçültülür (bütün kare; binalar drone'da büyük) ve
--doseme verilirse o boyutta örtüşen döşemelere bölünür.

  python src/rescuenet_veri.py --zip veri/rescuenet_ham/segmentation-validationset.zip --bolum valid
"""
import argparse
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
ESLEME = {5: 1, 4: 2, 3: 3, 2: 4}  # RescueNet → bizim sınıf
ADLAR = {0: "person", 1: "destroyed", 2: "major-damage", 3: "minor-damage", 4: "no-damage"}


def kutular(maske, olcek, min_alan, asindir):
    """maske: orijinal çözünürlükte sınıf haritası → [(sınıf, x1, y1, x2, y2)] küçültülmüş koordinatta."""
    cikti = []
    cekirdek = np.ones((3, 3), np.uint8)
    for rn, biz in ESLEME.items():
        ikili = (maske == rn).astype(np.uint8)
        if not ikili.any():
            continue
        if asindir:
            ikili = cv2.erode(ikili, cekirdek, iterations=asindir)
        n, _, ist, _ = cv2.connectedComponentsWithStats(ikili, connectivity=4)
        for x, y, w, h, alan in ist[1:]:
            if alan < min_alan:
                continue
            x1, y1, x2, y2 = x - asindir, y - asindir, x + w + asindir, y + h + asindir
            cikti.append((biz, x1 * olcek, y1 * olcek, x2 * olcek, y2 * olcek))
    return cikti


def eslestir(z: zipfile.ZipFile):
    """zip içinde görüntü ↔ etiket eşleşmesi (etiket adı '<kök>_lab.png' ya da label klasöründe aynı kök)."""
    goruntu, etiket = {}, {}
    for ad in z.namelist():
        p = Path(ad)
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        kok = p.stem.removesuffix("_lab")
        if p.stem.endswith("_lab") or "label" in ad.lower():
            etiket[kok] = ad
        else:
            goruntu[kok] = ad
    return [(k, goruntu[k], etiket[k]) for k in sorted(goruntu) if k in etiket]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--bolum", required=True, help="train / valid / test")
    ap.add_argument("--cikti", default="veri/rescuenet_yolo")
    ap.add_argument("--boyut", type=int, default=1280, help="küçültülmüş uzun kenar")
    ap.add_argument("--min-alan", type=int, default=400, help="orijinal çözünürlükte en küçük bina alanı (px)")
    ap.add_argument("--asindir", type=int, default=3)
    ap.add_argument("--adim", type=int, default=1, help="her N. görüntüyü al")
    a = ap.parse_args()
    cikti = KOK / a.cikti
    (cikti / a.bolum / "images").mkdir(parents=True, exist_ok=True)
    (cikti / a.bolum / "labels").mkdir(parents=True, exist_ok=True)
    sayac = {k: 0 for k in ESLEME.values()}
    with zipfile.ZipFile(KOK / a.zip) as z:
        ciftler = eslestir(z)[::a.adim]
        print(len(ciftler), "görüntü/etiket çifti", flush=True)
        for i, (kok, g_ad, e_ad) in enumerate(ciftler):
            g = Image.open(io.BytesIO(z.read(g_ad))).convert("RGB")
            m = np.array(Image.open(io.BytesIO(z.read(e_ad))))
            if m.ndim == 3:
                m = m[..., 0]
            olcek = a.boyut / max(g.size)
            W, H = round(g.width * olcek), round(g.height * olcek)
            satirlar = []
            # maske görüntüyle aynı boyutta olmayabilir → maske pikselinden küçültülmüş görüntüye ölçek
            for c, x1, y1, x2, y2 in kutular(m, W / m.shape[1], a.min_alan, a.asindir):
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                satirlar.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                sayac[c] += 1
            g.resize((W, H), Image.LANCZOS).save(cikti / a.bolum / "images" / f"rn_{kok}.jpg", quality=92)
            (cikti / a.bolum / "labels" / f"rn_{kok}.txt").write_text("\n".join(satirlar) + ("\n" if satirlar else ""))
            if i % 100 == 0:
                print(i, {ADLAR[k]: v for k, v in sayac.items()}, flush=True)
    print("bitti", {ADLAR[k]: v for k, v in sayac.items()})
    yaml = cikti / f"{a.bolum}.yaml"
    yaml.write_text(f"path: {cikti.as_posix()}\ntrain: {a.bolum}/images\nval: {a.bolum}/images\n"
                    f"test: {a.bolum}/images\nnames: {ADLAR}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
