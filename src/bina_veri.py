"""xBD afet ÖNCESİ görüntülerinden tek sınıflı bina konum veri seti (YOLO) üretir.

İki aşamalı sistemin konum aşaması için: sonradan yıkılan binalar öncesi görüntüde hâlâ
sağlam durur. Kutular, xbd_cevir.py ile aynı şekilde t2_mask bağlı bileşenlerinden çıkar
(hasar seviyesi fark etmeksizin), böylece değerlendirmedeki gerçek kutularla tutarlıdır.

Kullanım: python src/bina_veri.py
"""
import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from xbd_cevir import DISLA, DOSEME, kutular

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham" / "xbd"
CIKTI = KOK / "veri" / "bina"


def main():
    for b in ("train", "valid"):
        (CIKTI / b / "images").mkdir(parents=True, exist_ok=True)
        (CIKTI / b / "labels").mkdir(parents=True, exist_ok=True)
    adet = 0
    for dosya in sorted(HAM.glob("*.parquet")):
        for grup in pq.ParquetFile(dosya).iter_batches(batch_size=8, columns=["t1_image", "t2_mask", "image_name"]):
            for satir in grup.to_pylist():
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA):
                    continue
                bolum = "valid" if random.Random(ad).random() < 0.1 else "train"  # xbd_cevir ile aynı ayrım
                goruntu = Image.open(io.BytesIO(satir["t1_image"]["bytes"])).convert("RGB")
                maske = np.array(Image.open(io.BytesIO(satir["t2_mask"]["bytes"])))
                if maske.ndim == 3:
                    maske = maske[..., 0]
                for oy in range(0, maske.shape[0], DOSEME):
                    for ox in range(0, maske.shape[1], DOSEME):
                        satirlar = [f"0 {(x + w / 2) / DOSEME:.6f} {(y + h / 2) / DOSEME:.6f} "
                                    f"{w / DOSEME:.6f} {h / DOSEME:.6f}"
                                    for _, x, y, w, h in kutular(maske[oy:oy + DOSEME, ox:ox + DOSEME])]
                        if not satirlar:
                            continue
                        isim = f"xbd_{ad}_{oy}_{ox}"
                        goruntu.crop((ox, oy, ox + DOSEME, oy + DOSEME)).save(
                            CIKTI / bolum / "images" / f"{isim}.jpg", quality=92)
                        (CIKTI / bolum / "labels" / f"{isim}.txt").write_text("\n".join(satirlar) + "\n")
                adet += 1
        print(dosya.name, adet, "görüntü", flush=True)
    (CIKTI / "data.yaml").write_text(
        f"path: {CIKTI.as_posix()}\ntrain: train/images\nval: valid/images\nnames: {{0: building}}\n")


if __name__ == "__main__":
    main()
