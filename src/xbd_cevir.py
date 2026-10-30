"""xBD (xView2) parquet dosyalarını YOLO döşemelerine çevirir.

Afet sonrası görüntü (t2_image) ve hasar maskesi (t2_mask: 0 arka plan, 1 no-damage,
2 minor-damage, 3 major-damage, 4 destroyed) kullanılır. Her hasar seviyesinin bağlı
bileşenleri bina kutusu olur. 1024 px görüntüler 512 px döşemelere bölünür (uydu
görüntüsünde binalar küçük). Yalnızca hasarsız binası olan döşemelerin bir kısmı
atılarak sınıf dengesizliği azaltılır.

Kullanım: python src/xbd_cevir.py [--hasarsiz-oran 0.25] [--sinir N]
          python src/xbd_cevir.py --test   (bağımsız test seti)
"""
import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham" / "xbd"
CIKTI = KOK / "veri" / "xbd_yolo"

# maske değeri → birleşik sınıf kimliği (veri/birlesik.yaml ile aynı)
SINIF = {1: 4, 2: 3, 3: 2, 4: 1}
DOSEME = 512
# Roboflow hasar seti xBD hurricane-maria döşemelerinden türetilmiş; o testle sızıntı olmasın diye alınmaz.
DISLA = ("hurricane-maria",)
MIN_ALAN = 24  # piksel; daha küçük bileşenler gürültü


def kutular(maske: np.ndarray):
    for deger, sinif in SINIF.items():
        ikili = (maske == deger).astype(np.uint8)
        if not ikili.any():
            continue
        n, _, ist, _ = cv2.connectedComponentsWithStats(ikili, connectivity=4)
        for x, y, w, h, alan in ist[1:]:
            if alan >= MIN_ALAN:
                yield sinif, x, y, w, h


def doseme_yaz(goruntu, maske, ad, bolum, hasarsiz_oran, rng):
    sayac = np.zeros(5, int)
    H, W = maske.shape
    for oy in range(0, H, DOSEME):
        for ox in range(0, W, DOSEME):
            m = maske[oy:oy + DOSEME, ox:ox + DOSEME]
            satirlar, siniflar = [], []
            for s, x, y, w, h in kutular(m):
                satirlar.append(f"{s} {(x + w / 2) / DOSEME:.6f} {(y + h / 2) / DOSEME:.6f} "
                                f"{w / DOSEME:.6f} {h / DOSEME:.6f}")
                siniflar.append(s)
            if not satirlar:
                continue
            if all(s == 4 for s in siniflar) and rng.random() > hasarsiz_oran:
                continue
            isim = f"xbd_{ad}_{oy}_{ox}"
            goruntu.crop((ox, oy, ox + DOSEME, oy + DOSEME)).save(
                CIKTI / bolum / "images" / f"{isim}.jpg", quality=92)
            (CIKTI / bolum / "labels" / f"{isim}.txt").write_text("\n".join(satirlar) + "\n")
            for s in siniflar:
                sayac[s] += 1
    return sayac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasarsiz-oran", type=float, default=0.25,
                    help="yalnız hasarsız bina içeren döşemelerden tutulacak oran")
    ap.add_argument("--valid-oran", type=float, default=0.1)
    ap.add_argument("--sinir", type=int, default=0, help="test için en fazla bu kadar görüntü")
    ap.add_argument("--test", action="store_true",
                    help="veri/ham/xbd_test → veri/xbd_test (eğitimde hiç kullanılmayan bağımsız test; döşeme atılmaz)")
    a = ap.parse_args()
    global HAM, CIKTI
    if a.test:
        HAM, CIKTI = KOK / "veri" / "ham" / "xbd_test", KOK / "veri" / "xbd_test"
        a.hasarsiz_oran = 1.0
    rng = random.Random(0)
    for b in (("test",) if a.test else ("train", "valid")):
        (CIKTI / b / "images").mkdir(parents=True, exist_ok=True)
        (CIKTI / b / "labels").mkdir(parents=True, exist_ok=True)

    toplam, adet = np.zeros(5, int), 0
    for dosya in sorted(HAM.glob("*.parquet")):
        tablo = pq.ParquetFile(dosya)
        for grup in tablo.iter_batches(batch_size=16, columns=["t2_image", "t2_mask", "image_name"]):
            for satir in grup.to_pylist():
                goruntu = Image.open(io.BytesIO(satir["t2_image"]["bytes"])).convert("RGB")
                maske = np.array(Image.open(io.BytesIO(satir["t2_mask"]["bytes"])))
                if maske.ndim == 3:
                    maske = maske[..., 0]
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA):
                    continue
                # Afet olayı bazında değil görüntü bazında bölmek sızıntı riski taşır; sabit tohumla
                # görüntü adına göre ayırıyoruz (aynı görüntünün döşemeleri aynı bölümde kalır).
                if a.test:
                    bolum = "test"
                else:
                    bolum = "valid" if random.Random(ad).random() < a.valid_oran else "train"
                toplam += doseme_yaz(goruntu, maske, ad, bolum, a.hasarsiz_oran, rng)
                adet += 1
                if adet % 200 == 0:
                    print(adet, "görüntü;", dict(zip(["person", "destroyed", "major", "minor", "no"], toplam)))
                if a.sinir and adet >= a.sinir:
                    print("sınır:", dict(zip(["person", "destroyed", "major", "minor", "no"], toplam)))
                    return
    if a.test:
        (KOK / "veri" / "xbd_test.yaml").write_text(
            f"path: {CIKTI.as_posix()}\ntrain: test/images\nval: test/images\ntest: test/images\n"
            "names: {0: person, 1: destroyed, 2: major-damage, 3: minor-damage, 4: no-damage}\n")
    print("bitti:", adet, "görüntü;", dict(zip(["person", "destroyed", "major", "minor", "no"], toplam)))


if __name__ == "__main__":
    main()
