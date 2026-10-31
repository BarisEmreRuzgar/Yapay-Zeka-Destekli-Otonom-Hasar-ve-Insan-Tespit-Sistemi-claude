"""İki aşamalı hasar tespiti: binayı afet ÖNCESİ görüntüde bul, hasarı öncesi/sonrası çiftiyle sınıfla.

Değerlendirme modu bağımsız xBD testinde (veri/ham/xbd_test, 512 px döşemeler) şu kurguları
aynı AP50 hesabıyla karşılaştırır:
  tek      : mevcut dedektör sonrası görüntüde (bugünkü sistem)
  iki      : dedektör öncesi görüntüde konum + çift sınıflandırıcı hasar
  iki+gt   : gerçek kutular + çift sınıflandırıcı (sınıflandırıcının tavanı)

Kullanım: python src/iki_asama.py --siniflandirici runs/cift/weights/best.pt [--konumlayici runs/bina/weights/best.pt]
          python src/iki_asama.py --siniflandirici ... --once once.jpg --sonra sonra.jpg   (tek çift, çizimli)
"""
import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from ultralytics import YOLO

from cift_veri import DISLA, cift_gorsel

KOK = Path(__file__).resolve().parent.parent
HASAR = ["destroyed", "major-damage", "minor-damage", "no-damage"]
DOSEME = 512


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    k = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return k / (aa[:, None] + bb[None] - k + 1e-9)


def ap50(tahminler, gercekler):
    """tahminler: [(görsel, skor, kutu)], gercekler: {görsel: kutular}. COCO tarzı 101 nokta AP."""
    n_gt = sum(len(v) for v in gercekler.values())
    if n_gt == 0:
        return float("nan")
    kullanildi = {k: np.zeros(len(v), bool) for k, v in gercekler.items()}
    dp = []
    for g, s, kutu in sorted(tahminler, key=lambda t: -t[1]):
        gt = gercekler.get(g)
        if gt is None or not len(gt):
            dp.append(0); continue
        o = iou(kutu[None], gt)[0]; j = o.argmax()
        if o[j] >= 0.5 and not kullanildi[g][j]:
            kullanildi[g][j] = True; dp.append(1)
        else:
            dp.append(0)
    dp = np.array(dp)
    tp = np.cumsum(dp); fp = np.cumsum(1 - dp)
    duyarlilik = tp / n_gt; kesinlik = tp / np.maximum(tp + fp, 1e-9)
    kesinlik = np.maximum.accumulate(np.r_[kesinlik, 0][::-1])[::-1]
    duyarlilik = np.r_[duyarlilik, duyarlilik[-1] if len(duyarlilik) else 0]
    noktalar = np.linspace(0, 1, 101)
    idx = np.searchsorted(duyarlilik, noktalar, side="left")
    return float(np.mean([kesinlik[i] if i < len(kesinlik) else 0 for i in idx]))


class Siniflandirici:
    def __init__(self, yol):
        self.m = YOLO(yol)
        self.ad = self.m.names

    def __call__(self, once, sonra, kutular):
        """kutular: Nx4 xyxy (tam görüntü koordinatı) → Nx4 olasılık (HASAR sırasıyla)."""
        if not len(kutular):
            return np.zeros((0, 4))
        gorseller = [cift_gorsel(once, sonra, x1, y1, x2 - x1, y2 - y1)[..., ::-1] for x1, y1, x2, y2 in kutular]
        cikis = []
        for i in range(0, len(gorseller), 64):
            for r in self.m.predict(gorseller[i:i + 64], imgsz=224, verbose=False):
                p = r.probs.data.cpu().numpy()
                cikis.append([p[[k for k, v in self.ad.items() if v == h][0]] for h in HASAR])
        return np.array(cikis)


def bina_kutulari(dedektor, goruntu, conf):
    # tek sınıflı konum modeli (bina) ya da 5 sınıflı ana model (insan sınıfı hariç) olabilir
    siniflar = None if len(dedektor.names) == 1 else [1, 2, 3, 4]
    r = dedektor.predict(goruntu[..., ::-1], conf=conf, iou=0.7, imgsz=640, verbose=False, classes=siniflar,
                         agnostic_nms=True)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def degerlendir(a):
    dedektor = YOLO(str(KOK / "modeller" / "disastervision.pt"))
    konumlayici = YOLO(a.konumlayici) if a.konumlayici else dedektor
    sinif = Siniflandirici(a.siniflandirici)
    etiketler = KOK / "veri" / "xbd_test" / "test" / "labels"
    gercek = {h: {} for h in HASAR}
    kurgular = ("tek", "iki", "iki+gt")
    tahmin = {k: {h: [] for h in HASAR} for k in kurgular}
    dogru = defaultdict(int); toplam = defaultdict(int)

    for dosya in sorted((KOK / "veri" / "ham" / "xbd_test").glob("*.parquet")):
        for grup in pq.ParquetFile(dosya).iter_batches(batch_size=8, columns=["t1_image", "t2_image", "image_name"]):
            for satir in grup.to_pylist():
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA):
                    continue
                once = np.array(Image.open(io.BytesIO(satir["t1_image"]["bytes"])).convert("RGB"))
                sonra = np.array(Image.open(io.BytesIO(satir["t2_image"]["bytes"])).convert("RGB"))
                for oy in range(0, once.shape[0], DOSEME):
                    for ox in range(0, once.shape[1], DOSEME):
                        g = f"xbd_{ad}_{oy}_{ox}"
                        lf = etiketler / f"{g}.txt"
                        if not lf.exists():
                            continue
                        L = np.loadtxt(lf, ndmin=2)
                        gt = np.c_[L[:, 1] - L[:, 3] / 2, L[:, 2] - L[:, 4] / 2,
                                   L[:, 1] + L[:, 3] / 2, L[:, 2] + L[:, 4] / 2] * DOSEME
                        gc = L[:, 0].astype(int)
                        for h in HASAR:
                            gercek[h][g] = gt[gc == HASAR.index(h) + 1]
                        kaydir = np.array([ox, oy, ox, oy])
                        o_d, s_d = once[oy:oy + DOSEME, ox:ox + DOSEME], sonra[oy:oy + DOSEME, ox:ox + DOSEME]

                        # tek: bugünkü sistem, sonrası görüntüde 4 hasar sınıfı
                        r = dedektor.predict(s_d[..., ::-1], conf=0.001, iou=0.7, imgsz=640, verbose=False,
                                             classes=[1, 2, 3, 4])[0]
                        for k, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                           r.boxes.cls.cpu().numpy().astype(int)):
                            tahmin["tek"][HASAR[c - 1]].append((g, float(s), k))

                        # iki: öncesinde konum (sınıf-bağımsız), çiftle hasar
                        kutu, skor, _ = bina_kutulari(konumlayici, o_d, 0.001)
                        p = sinif(once, sonra, kutu + kaydir)
                        for k, s, pi in zip(kutu, skor, p):
                            for j, h in enumerate(HASAR):
                                tahmin["iki"][h].append((g, float(s * pi[j]), k))

                        # iki+gt: sınıflandırıcının tavanı
                        p = sinif(once, sonra, gt + kaydir)
                        for k, c, pi in zip(gt, gc, p):
                            for j, h in enumerate(HASAR):
                                tahmin["iki+gt"][h].append((g, float(pi[j]), k))
                            toplam[c] += 1; dogru[c] += int(pi.argmax() == c - 1)

    sonuc = {}
    for k in kurgular:
        ap = {h: ap50(tahmin[k][h], gercek[h]) for h in HASAR}
        sonuc[k] = {"mAP50": float(np.mean(list(ap.values()))), "AP50": ap}
        print(f"{k:7s} mAP50={sonuc[k]['mAP50']:.3f}  " + "  ".join(f"{h}={v:.3f}" for h, v in ap.items()))
    sonuc["gt_kutu_dogruluk"] = {HASAR[c - 1]: dogru[c] / toplam[c] for c in sorted(toplam)}
    print("gerçek kutularda sınıf doğruluğu:", {h: round(v, 3) for h, v in sonuc["gt_kutu_dogruluk"].items()})
    (KOK / "runs" / "deg_iki_asama.json").write_text(json.dumps(sonuc, indent=1, ensure_ascii=False))


def tek_cift(a):
    dedektor = YOLO(a.konumlayici or str(KOK / "modeller" / "disastervision.pt"))
    sinif = Siniflandirici(a.siniflandirici)
    once = np.array(Image.open(a.once).convert("RGB")); sonra = np.array(Image.open(a.sonra).convert("RGB"))
    kutu, skor, _ = bina_kutulari(dedektor, once, a.conf)
    p = sinif(once, sonra, kutu)
    renk = {"destroyed": (230, 30, 30), "major-damage": (240, 120, 20), "minor-damage": (240, 200, 30),
            "no-damage": (40, 200, 60)}
    cizim = sonra.copy()
    for (x1, y1, x2, y2), s, pi in zip(kutu.astype(int), skor, p):
        h = HASAR[pi.argmax()]
        cv2.rectangle(cizim, (x1, y1), (x2, y2), renk[h], 2)
        cv2.putText(cizim, f"{h} {pi.max():.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, renk[h], 1)
    yol = Path(a.sonra).with_name(Path(a.sonra).stem + "_iki_asama.jpg")
    Image.fromarray(cizim).save(yol)
    print(yol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--siniflandirici", default=str(KOK / "runs" / "cift" / "weights" / "best.pt"))
    ap.add_argument("--konumlayici", help="öncesi görüntüde bina bulan model (yoksa ana model)")
    ap.add_argument("--once"); ap.add_argument("--sonra")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    tek_cift(a) if a.once else degerlendir(a)
