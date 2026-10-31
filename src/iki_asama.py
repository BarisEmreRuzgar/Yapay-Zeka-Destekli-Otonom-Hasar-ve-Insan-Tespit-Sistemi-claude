"""İki aşamalı hasar tespiti: binayı afet ÖNCESİ görüntüde bul, hasarı öncesi/sonrası çiftiyle sınıfla.

Değerlendirme modu bağımsız xBD testinde (veri/ham/xbd_test, 512 px döşemeler) şu kurguları
aynı AP50 hesabıyla karşılaştırır:
  tek      : mevcut dedektör sonrası görüntüde (bugünkü sistem)
  iki      : dedektör öncesi görüntüde konum + çift sınıflandırıcı hasar
  iki+gt   : gerçek kutular + çift sınıflandırıcı (sınıflandırıcının tavanı)

Kullanım: python src/iki_asama.py [--siniflandirici modeller/hasar_cift.pt] [--konumlayici modeller/bina_konum.pt | ana]
          python src/iki_asama.py --siniflandirici ... --once once.jpg --sonra sonra.jpg   (tek çift, çizimli)
"""
import argparse
import io
import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from ultralytics import YOLO

from cift_veri import DISLA, cift_gorsel
from xbd_cevir import kutular

KOK = Path(__file__).resolve().parent.parent
HASAR = ["destroyed", "major-damage", "minor-damage", "no-damage"]
TEK_AGIRLIK = 0.6  # birleşimde tek-model skorlarının ağırlığı (val'de 0.3/0.6/1.0 arasından)
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
        self.imgsz = self.m.ckpt["train_args"]["imgsz"]  # panel = imgsz / 2

    def __call__(self, once, sonra, kutular):
        """kutular: Nx4 xyxy (tam görüntü koordinatı) → Nx4 olasılık (HASAR sırasıyla)."""
        if not len(kutular):
            return np.zeros((0, 4))
        gorseller = [cift_gorsel(once, sonra, x1, y1, x2 - x1, y2 - y1, panel=self.imgsz // 2)[..., ::-1]
                     for x1, y1, x2, y2 in kutular]
        cikis = []
        for i in range(0, len(gorseller), 64):
            for r in self.m.predict(gorseller[i:i + 64], imgsz=self.imgsz, verbose=False):
                p = r.probs.data.cpu().numpy()
                cikis.append([p[[k for k, v in self.ad.items() if v == h][0]] for h in HASAR])
        return np.array(cikis)


def bina_kutulari(dedektor, goruntu, conf, imgsz=640):
    # tek sınıflı konum modeli (bina) ya da 5 sınıflı ana model (insan sınıfı hariç) olabilir
    siniflar = None if len(dedektor.names) == 1 else [1, 2, 3, 4]
    r = dedektor.predict(goruntu[..., ::-1], conf=conf, iou=0.7, imgsz=imgsz, verbose=False, classes=siniflar,
                         agnostic_nms=True)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def ciftler(bolum, sinir=0):
    """(ad, öncesi, sonrası, maske) üretir. test: bağımsız xBD testi; val: eğitim dosyalarının
    görüntü adına göre ayrılmış %10'u (cift_veri, bina_veri, xbd_cevir ile aynı ayrım; hiçbir model görmedi)."""
    ham = KOK / "veri" / "ham" / ("xbd_test" if bolum == "test" else "xbd")
    adet = 0
    for dosya in sorted(ham.glob("*.parquet")):
        for grup in pq.ParquetFile(dosya).iter_batches(
                batch_size=8, columns=["t1_image", "t2_image", "t2_mask", "image_name"]):
            for satir in grup.to_pylist():
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA) or (bolum == "val" and random.Random(ad).random() >= 0.1):
                    continue
                maske = np.array(Image.open(io.BytesIO(satir["t2_mask"]["bytes"])))
                yield (ad, np.array(Image.open(io.BytesIO(satir["t1_image"]["bytes"])).convert("RGB")),
                       np.array(Image.open(io.BytesIO(satir["t2_image"]["bytes"])).convert("RGB")),
                       maske[..., 0] if maske.ndim == 3 else maske)
                adet += 1
                if sinir and adet >= sinir:
                    return


def nms(kutular, skorlar, esik=0.5):
    sira, tut = np.argsort(-skorlar), []
    while len(sira):
        i, sira = sira[0], sira[1:]
        tut.append(i)
        if len(sira):
            sira = sira[iou(kutular[i:i + 1], kutular[sira])[0] < esik]
    return np.array(tut, int)


def birlesik(d, j, tek_agirlik=TEK_AGIRLIK):
    """j. hasar sınıfı için iki aşama kutularına sonrası görüntüdeki tek-model kutularını ekleyip NMS uygular.
    Tek model hafif hasarı çatı dokusundan görebildiği, iki aşama ise yok olmuş binayı bulabildiği için tamamlayıcılar
    (val: mAP50 0.489 → 0.511; ağırlık val'de seçildi)."""
    m = d["tek_sinif"] == j + 1
    K = np.r_[d["kon_kutu"].reshape(-1, 4), d["tek_kutu"][m].reshape(-1, 4)]
    S = np.r_[d["kon_skor"] * (d["kon_olas"][:, j] if len(d["kon_kutu"]) else 0), d["tek_skor"][m] * tek_agirlik]
    t = nms(K, S) if len(K) else np.zeros(0, int)
    return K[t], S[t]


def puanla(ham, kurgular):
    """Kaydedilmiş ham tahminlerden AP50 tablosu. Skorlama değişikliklerini modeli yeniden çalıştırmadan denemek için ayrı."""
    gercek = {h: {g: d["gt"][d["gc"] == j + 1] for g, d in ham.items()} for j, h in enumerate(HASAR)}
    sonuc = {}
    for k in kurgular:
        tahmin = {h: [] for h in HASAR}
        for g, d in ham.items():
            for j, h in enumerate(HASAR):
                if k == "tek":
                    m = d["tek_sinif"] == j + 1
                    K, S = d["tek_kutu"][m], d["tek_skor"][m]
                elif k == "iki":
                    K, S = d["kon_kutu"], d["kon_skor"] * (d["kon_olas"][:, j] if len(d["kon_kutu"]) else 0)
                elif k == "iki+tek":
                    K, S = birlesik(d, j)
                else:  # iki+gt
                    K, S = d["gt"], d["gt_olas"][:, j]
                tahmin[h] += [(g, float(si), ki) for ki, si in zip(K, S)]
        ap = {h: ap50(tahmin[h], gercek[h]) for h in HASAR}
        sonuc[k] = {"mAP50": float(np.mean(list(ap.values()))), "AP50": ap}
        print(f"{k:7s} mAP50={sonuc[k]['mAP50']:.3f}  " + "  ".join(f"{h}={v:.3f}" for h, v in ap.items()))
    if {"iki", "iki+tek"} & set(kurgular):  # konum aşamasının kendi başarısı: sınıf yok sayılarak
        tum = {g: d["gt"] for g, d in ham.items()}
        ap = ap50([(g, float(s), k) for g, d in ham.items() for k, s in zip(d["kon_kutu"], d["kon_skor"])], tum)
        sonuc["konum_AP50"] = ap
        print(f"konum (sınıf-bağımsız) AP50={ap:.3f}")
    if "iki+gt" in kurgular:
        dogru = {h: np.mean([p.argmax() == j for d in ham.values() for p, c in zip(d["gt_olas"], d["gc"]) if c == j + 1])
                 for j, h in enumerate(HASAR)}
        sonuc["gt_kutu_dogruluk"] = {h: float(v) for h, v in dogru.items()}
        print("gerçek kutularda sınıf doğruluğu:", {h: round(v, 3) for h, v in dogru.items()})
    return sonuc


def degerlendir(a):
    kurgular = a.kurgu.split(",")
    dedektor = YOLO(str(KOK / "modeller" / "disastervision.pt"))
    konumlayici = dedektor if a.konumlayici == "ana" else YOLO(a.konumlayici)
    sinif = Siniflandirici(a.siniflandirici)
    ham = {}
    for ad, once, sonra, maske in ciftler(a.bolum, a.sinir):
        for oy in range(0, maske.shape[0], DOSEME):
            for ox in range(0, maske.shape[1], DOSEME):
                kutular_ = list(kutular(maske[oy:oy + DOSEME, ox:ox + DOSEME]))
                if not kutular_:
                    continue
                g = f"xbd_{ad}_{oy}_{ox}"
                d = ham[g] = {"gc": np.array([k[0] for k in kutular_]),
                              "gt": np.array([[x, y, x + w, y + h] for _, x, y, w, h in kutular_], float)}
                kaydir = np.array([ox, oy, ox, oy])
                o_d, s_d = once[oy:oy + DOSEME, ox:ox + DOSEME], sonra[oy:oy + DOSEME, ox:ox + DOSEME]
                if {"tek", "iki+tek"} & set(kurgular):  # bugünkü sistem: sonrası görüntüde 4 hasar sınıfı
                    r = dedektor.predict(s_d[..., ::-1], conf=0.001, iou=0.7, imgsz=640, verbose=False,
                                         classes=[1, 2, 3, 4])[0]
                    d["tek_kutu"], d["tek_skor"] = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                    d["tek_sinif"] = r.boxes.cls.cpu().numpy().astype(int)
                if {"iki", "iki+tek"} & set(kurgular):  # öncesinde konum (sınıf-bağımsız), çiftle hasar
                    d["kon_kutu"], d["kon_skor"], _ = bina_kutulari(konumlayici, o_d, 0.001, a.konum_imgsz)
                    d["kon_olas"] = sinif(once, sonra, d["kon_kutu"] + kaydir)
                if "iki+gt" in kurgular:  # sınıflandırıcının tavanı
                    d["gt_olas"] = sinif(once, sonra, d["gt"] + kaydir)
    print(a.bolum, len(ham), "döşeme")
    sonuc = puanla(ham, kurgular)
    if a.kaydet:
        with open(a.kaydet, "wb") as f:
            pickle.dump(ham, f)
    (KOK / "runs" / f"deg_iki_asama_{a.bolum}.json").write_text(json.dumps(sonuc, indent=1, ensure_ascii=False))


def tek_cift(a):
    ana = YOLO(str(KOK / "modeller" / "disastervision.pt"))
    konumlayici = ana if a.konumlayici == "ana" else YOLO(a.konumlayici)
    sinif = Siniflandirici(a.siniflandirici)
    once = np.array(Image.open(a.once).convert("RGB")); sonra = np.array(Image.open(a.sonra).convert("RGB"))
    d = {}
    d["kon_kutu"], d["kon_skor"], _ = bina_kutulari(konumlayici, once, 0.01, a.konum_imgsz)
    d["kon_olas"] = sinif(once, sonra, d["kon_kutu"])
    r = ana.predict(sonra[..., ::-1], conf=0.01, iou=0.7, imgsz=640, verbose=False, classes=[1, 2, 3, 4])[0]
    d["tek_kutu"], d["tek_skor"] = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    d["tek_sinif"] = r.boxes.cls.cpu().numpy().astype(int)
    # sınıf bazında birleşik tahminler → çizim için sınıf-bağımsız NMS ve eşik
    K, S, C = [], [], []
    for j in range(len(HASAR)):
        k, s = birlesik(d, j)
        K.append(k.reshape(-1, 4)); S.append(s); C += [j] * len(s)
    K, S, C = np.concatenate(K), np.concatenate(S), np.array(C, int)
    t = nms(K, S) if len(K) else np.zeros(0, int)
    t = t[S[t] >= a.conf]
    renk = {"destroyed": (230, 30, 30), "major-damage": (240, 120, 20), "minor-damage": (240, 200, 30),
            "no-damage": (40, 200, 60)}
    cizim = sonra.copy()
    for (x1, y1, x2, y2), s, c in zip(K[t].astype(int), S[t], C[t]):
        h = HASAR[c]
        cv2.rectangle(cizim, (x1, y1), (x2, y2), renk[h], 2)
        cv2.putText(cizim, f"{h} {s:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, renk[h], 1)
    yol = Path(a.sonra).with_name(Path(a.sonra).stem + "_iki_asama.jpg")
    Image.fromarray(cizim).save(yol)
    print(yol, {h: int((C[t] == j).sum()) for j, h in enumerate(HASAR)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--siniflandirici", default=str(KOK / "modeller" / "hasar_cift.pt"))
    ap.add_argument("--konumlayici", default=str(KOK / "modeller" / "bina_konum.pt"),
                    help="öncesi görüntüde bina bulan model; \"ana\" verilirse disastervision.pt")
    ap.add_argument("--once"); ap.add_argument("--sonra")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--bolum", choices=["test", "val"], default="test")
    ap.add_argument("--kurgu", default="tek,iki,iki+tek,iki+gt", help="virgülle: tek, iki, iki+tek, iki+gt")
    ap.add_argument("--konum-imgsz", type=int, default=640)
    ap.add_argument("--sinir", type=int, default=0, help="en fazla bu kadar görüntü (hızlı deneme)")
    ap.add_argument("--kaydet", help="ham tahminleri pickle olarak kaydet (puanlama denemeleri için)")
    a = ap.parse_args()
    tek_cift(a) if a.once else degerlendir(a)
