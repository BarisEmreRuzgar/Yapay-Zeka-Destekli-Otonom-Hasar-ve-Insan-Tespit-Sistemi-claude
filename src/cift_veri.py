"""xBD'den öncesi/sonrası bina kırpıntılarıyla hasar sınıflandırma veri seti üretir.

Her bina (t2_mask'te bir hasar seviyesinin bağlı bileşeni) için afet öncesi ve sonrası
görüntüden aynı bölge bağlamıyla kırpılır ve tek bir 224 px görsele dizilir:
    üst sol: öncesi   üst sağ: sonrası
    alt sol: |sonrası - öncesi| farkı (güçlendirilmiş)   alt sağ: bina maskesi
Böylece standart YOLOv8-cls mimarisi iki zamanı birlikte görür.

Kullanım: python src/cift_veri.py [--sinif-basina 12000] [--test] [--maske kutu|gercek] [--panel 128 --cikti cift2]
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
SINIFLAR = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}
DISLA = ("hurricane-maria",)  # xbd_cevir.py ile aynı gerekçe
MIN_ALAN = 24
PANEL = 112


def bina_kutulari(maske):
    for deger, ad in SINIFLAR.items():
        ikili = (maske == deger).astype(np.uint8)
        if not ikili.any():
            continue
        _, etiket, ist, _ = cv2.connectedComponentsWithStats(ikili, connectivity=4)
        for i, (x, y, w, h, alan) in enumerate(ist[1:], start=1):
            if alan >= MIN_ALAN:
                yield ad, x, y, w, h, (etiket == i)


def oynat(x, y, w, h, rng, oran=0.15):
    """Kutuyu konum modelinin kusurlarını taklit edecek şekilde rastgele kaydırıp boyutlandırır."""
    dw, dh = w * rng.uniform(-oran, oran), h * rng.uniform(-oran, oran)
    return x + w * rng.uniform(-oran, oran) - dw / 2, y + h * rng.uniform(-oran, oran) - dh / 2, w + dw, h + dh


def paneller_cevir(g, islem):
    """2x2 panelli görselde her paneli kendi içinde çevirir/döndürür; panel düzeni (öncesi sol üstte vb.) korunur.
    islem: 0 kimlik, 1 yatay, 2 dikey, 3 180°, 4 90°, 5 270°, 6 devrik, 7 ters devrik (kare simetri grubu)."""
    if islem == 0:
        return g
    p = g.shape[0] // 2
    dondur = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1], lambda a: a[::-1, ::-1],
              lambda a: np.rot90(a), lambda a: np.rot90(a, 3), lambda a: a.transpose(1, 0, 2),
              lambda a: np.rot90(a, 2).transpose(1, 0, 2)][islem]
    paneller = [dondur(g[i:i + p, j:j + p]) for i in (0, p) for j in (0, p)]
    return np.ascontiguousarray(np.vstack([np.hstack(paneller[:2]), np.hstack(paneller[2:])]))


def cift_gorsel(once, sonra, x, y, w, h, bina=None, panel=PANEL):
    """Kutuyu bağlamla genişletip 2x2 panelli (2*panel px) görsel döndürür (np.uint8, RGB).
    bina verilmezse maske paneli kutunun kendisidir (çıkarımda maske yok)."""
    H, W = once.shape[:2]
    kenar = max(w, h) * 1.6 + 16
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(max(0, cx - kenar / 2)), int(max(0, cy - kenar / 2))
    x2, y2 = int(min(W, cx + kenar / 2)), int(min(H, cy + kenar / 2))
    a = cv2.resize(once[y1:y2, x1:x2], (panel, panel), interpolation=cv2.INTER_AREA)
    b = cv2.resize(sonra[y1:y2, x1:x2], (panel, panel), interpolation=cv2.INTER_AREA)
    fark = np.clip(np.abs(b.astype(np.int16) - a.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
    if bina is None:
        bina = np.zeros((H, W), bool)
        bina[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = True
    m = cv2.resize(bina[y1:y2, x1:x2].astype(np.uint8) * 255, (panel, panel), interpolation=cv2.INTER_NEAREST)
    return np.vstack([np.hstack([a, b]), np.hstack([fark, np.dstack([m] * 3)])])


def _kirp(goruntu, cx, cy, kenar, panel):
    """Merkez (cx, cy) çevresinde kenar×kenar bölgeyi kırpıp panel boyutuna getirir; dışarı taşan kısım siyah."""
    H, W = goruntu.shape[:2]
    x1, y1 = int(round(cx - kenar / 2)), int(round(cy - kenar / 2))
    x2, y2 = x1 + max(1, int(round(kenar))), y1 + max(1, int(round(kenar)))
    tuval = np.zeros((y2 - y1, x2 - x1) + goruntu.shape[2:], goruntu.dtype)
    ax1, ay1, ax2, ay2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if ax2 > ax1 and ay2 > ay1:
        tuval[ay1 - y1:ay2 - y1, ax1 - x1:ax2 - x1] = goruntu[ay1:ay2, ax1:ax2]
    return cv2.resize(tuval, (panel, panel), interpolation=cv2.INTER_AREA if kenar > panel else cv2.INTER_LINEAR)


def tek_gorsel(sonra, x, y, w, h, bina=None, panel=PANEL):
    """Yalnız afet SONRASI görüntüden 2x2 panel (öncesi görüntü yokken, drone senaryosu):
        üst sol: geniş bağlam (enkaz, çevre)   üst sağ: yakın plan (bina)
        alt sol: yakın planın kenar haritası    alt sağ: bina maskesi (yakın plan ölçeğinde)"""
    H, W = sonra.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    yakin = max(w, h) * 1.3 + 8
    genis = max(w, h) * 3.0 + 32
    a = _kirp(sonra, cx, cy, genis, panel)
    b = _kirp(sonra, cx, cy, yakin, panel)
    gri = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kenar = cv2.magnitude(cv2.Sobel(gri, cv2.CV_32F, 1, 0), cv2.Sobel(gri, cv2.CV_32F, 0, 1))
    kenar = np.clip(kenar * (255.0 / (np.percentile(kenar, 99) + 1e-6)), 0, 255).astype(np.uint8)
    if bina is None:
        bina = np.zeros((H, W), bool)
        bina[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = True
    m = _kirp(bina.astype(np.uint8) * 255, cx, cy, yakin, panel)
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return np.vstack([np.hstack([a, b]), np.hstack([np.dstack([kenar] * 3), np.dstack([m] * 3)])])


def oku(satir, alan):
    g = np.array(Image.open(io.BytesIO(satir[alan]["bytes"])).convert("RGB"))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinif-basina", type=int, default=12000, help="eğitimde sınıf başına üst sınır")
    ap.add_argument("--valid-oran", type=float, default=0.1)
    ap.add_argument("--test", action="store_true", help="bağımsız xBD testinden (sınırsız)")
    ap.add_argument("--maske", choices=["kutu", "gercek"], default="kutu",
                    help="kutu: oynatılmış dikdörtgen (çıkarımla aynı); gercek: bina şekli (ilk sürüm)")
    ap.add_argument("--panel", type=int, default=PANEL)
    ap.add_argument("--hasarsiz-alma", type=float, default=0.12)
    ap.add_argument("--cikti", default="cift")
    ap.add_argument("--cevir", action="store_true", help="eğitim örneklerini panel içinde rastgele çevir/döndür")
    ap.add_argument("--tek", action="store_true", help="yalnız sonrası görüntüden paneller (tek_gorsel)")
    a = ap.parse_args()

    ham = KOK / "veri" / "ham" / ("xbd_test" if a.test else "xbd")
    cikti = KOK / "veri" / (a.cikti + "_test" if a.test else a.cikti)
    sayac = {b: {s: 0 for s in SINIFLAR.values()} for b in ("train", "val", "test")}
    rng = random.Random(0)
    # Hasarsız binalar çok baskın; sınır dolmadan tükenmesin diye her binayı olasılıkla alıyoruz.
    alma = {"no-damage": a.hasarsiz_alma, "minor-damage": 1.0, "major-damage": 1.0, "destroyed": 1.0}

    for dosya in sorted(ham.glob("*.parquet")):
        for grup in pq.ParquetFile(dosya).iter_batches(
                batch_size=8, columns=["t1_image", "t2_image", "t2_mask", "image_name"]):
            for satir in grup.to_pylist():
                ad = Path(satir["image_name"]).stem
                if ad.startswith(DISLA):
                    continue
                bolum = "test" if a.test else ("val" if random.Random(ad).random() < a.valid_oran else "train")
                maske = np.array(Image.open(io.BytesIO(satir["t2_mask"]["bytes"])))
                if maske.ndim == 3:
                    maske = maske[..., 0]
                once, sonra = oku(satir, "t1_image"), oku(satir, "t2_image")
                for k, (sinif, x, y, w, h, bina) in enumerate(bina_kutulari(maske)):
                    if not a.test:
                        sinir = a.sinif_basina if bolum == "train" else a.sinif_basina // 8
                        if sayac[bolum][sinif] >= sinir or rng.random() > alma[sinif]:
                            continue
                    hedef = cikti / bolum / sinif
                    hedef.mkdir(parents=True, exist_ok=True)
                    if a.maske == "kutu":
                        x, y, w, h = oynat(x, y, w, h, rng) if not a.test else (x, y, w, h)
                        bina = None
                    g = (tek_gorsel(sonra, x, y, w, h, bina, a.panel) if a.tek
                         else cift_gorsel(once, sonra, x, y, w, h, bina, a.panel))
                    if a.cevir and bolum == "train":
                        g = paneller_cevir(g, rng.randrange(8))
                    Image.fromarray(g).save(
                        hedef / f"{ad}_{k}.jpg", quality=92)
                    sayac[bolum][sinif] += 1
        print(dosya.name, {b: s for b, s in sayac.items() if any(s.values())}, flush=True)


if __name__ == "__main__":
    main()
