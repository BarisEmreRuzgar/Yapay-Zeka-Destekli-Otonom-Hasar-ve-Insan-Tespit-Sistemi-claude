"""CRASAR-U-DROIDs (Hugging Face, CC-BY-4.0) drone ortomozaiklerinden YOLO hasar döşemeleri.

Her ortomozaik (geo.tif, 0.3–15 GB) sırayla indirilir, işlenir ve silinir (disk dar):
  * Bina çokgenleri (Microsoft ayak izleri, Joint Damage Scale etiketli) piksel koordinatında gelir; yazarların
    hizalama vektörleri (building_alignment_adjustments: [başlangıç, bitiş]) en yakın 3 vektörün ters uzaklık
    ağırlıklı ortalamasıyla kaydırma olarak uygulanır.
  * Görüntü, bina uzun kenarı medyanı --hedef-bina px olacak şekilde küçültülür (RescueNet @1280 ile benzer ölçek).
  * 'un-classified' binalar griyle örtülür (etiketsiz bina = arka plan öğrenilmesin).
  * --doseme px döşemeler; geçerli (nodata olmayan) alan < %50 ya da etiketli bina yoksa atlanır.
Sınıflar ana modelle aynı: 1 destroyed, 2 major-damage, 3 minor-damage, 4 no-damage.

  python src/crasar_veri.py --bolum train --maks-gb 4.6
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

KOK = Path(__file__).resolve().parent.parent
DEPO = "CRASAR/CRASAR-U-DROIDs"
HAM = KOK / "veri" / "crasar_ham"
SINIF = {"destroyed": 1, "major damage": 2, "minor damage": 3, "no damage": 4}


def kaydirmalar(cokgenler, vektorler):
    if not vektorler:
        return [np.zeros(2)] * len(cokgenler)
    bas = np.array([v[0] for v in vektorler], float)
    fark = np.array([np.subtract(v[1], v[0]) for v in vektorler], float)
    cikti = []
    for c in cokgenler:
        m = c.mean(0)
        d = np.linalg.norm(bas - m, axis=1)
        sira = np.argsort(d)[:3]
        w = 1.0 / (d[sira] + 1.0)
        cikti.append((fark[sira] * w[:, None]).sum(0) / w.sum())
    return cikti


def isle(tif_yol: Path, etiketler, vektorler, cikti: Path, bolum, hedef_bina, doseme, sayac):
    import rasterio
    from rasterio.enums import Resampling

    cokgenler = [np.array([[p["x"], p["y"]] for p in e["pixels"]], float) for e in etiketler]
    for c, k in zip(cokgenler, kaydirmalar(cokgenler, vektorler)):
        c += k
    kenar = [max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) for c in cokgenler]
    s = min(1.0, hedef_bina / max(1.0, float(np.median(kenar))))
    with rasterio.open(tif_yol) as ds:
        W, H = round(ds.width * s), round(ds.height * s)
        bantlar = ds.read(out_shape=(ds.count, H, W), resampling=Resampling.average)
    rgb = np.ascontiguousarray(np.transpose(bantlar[:3], (1, 2, 0)))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(1, np.percentile(rgb, 99.5)) * 255, 0, 255).astype(np.uint8)
    gecerli = (bantlar[3] > 0) if bantlar.shape[0] >= 4 else (rgb.max(2) > 0)
    del bantlar
    kutular = []
    for e, c in zip(etiketler, cokgenler):
        c = c * s
        if e["label"] not in SINIF:  # un-classified → gri örtü
            cv2.fillPoly(rgb, [np.round(c).astype(np.int32)], (114, 114, 114))
            continue
        kutular.append((SINIF[e["label"]], *c.min(0), *c.max(0)))
    kutular = np.array(kutular, float).reshape(-1, 5)
    ad = tif_yol.name.replace(".geo.tif", "").replace(" ", "_")
    n = 0
    for oy in range(0, max(1, H - doseme // 4), doseme):
        for ox in range(0, max(1, W - doseme // 4), doseme):
            g = gecerli[oy:oy + doseme, ox:ox + doseme]
            if g.size == 0 or g.mean() < 0.5:
                continue
            satirlar, ortu = [], []
            for c, x1, y1, x2, y2 in kutular:
                a1, b1 = max(x1, ox), max(y1, oy)
                a2, b2 = min(x2, ox + doseme), min(y2, oy + doseme)
                if a2 <= a1 or b2 <= b1:
                    continue
                if (a2 - a1) * (b2 - b1) < 0.4 * (x2 - x1) * (y2 - y1):
                    ortu.append((a1 - ox, b1 - oy, a2 - ox, b2 - oy))  # çoğu döşeme dışında: etiketsiz kalmasın, ört
                    continue
                tw, th = min(doseme, W - ox), min(doseme, H - oy)
                satirlar.append(f"{int(c)} {((a1 + a2) / 2 - ox) / tw:.6f} {((b1 + b2) / 2 - oy) / th:.6f} "
                                f"{(a2 - a1) / tw:.6f} {(b2 - b1) / th:.6f}")
                sayac[int(c)] += 1
            if not satirlar:
                continue
            parca = rgb[oy:oy + doseme, ox:ox + doseme].copy()
            parca[~g] = 0
            for x1, y1, x2, y2 in ortu:
                parca[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 114
            isim = f"cr_{ad}_{oy}_{ox}"
            cv2.imwrite(str(cikti / bolum / "images" / f"{isim}.jpg"), parca[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            (cikti / bolum / "labels" / f"{isim}.txt").write_text("\n".join(satirlar) + "\n")
            n += 1
    return n, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="train", help="HF klasörü: train / valid / test")
    ap.add_argument("--cikti", default="veri/crasar_yolo")
    ap.add_argument("--maks-gb", type=float, default=4.6)
    ap.add_argument("--hedef-bina", type=float, default=160)
    ap.add_argument("--doseme", type=int, default=1024)
    a = ap.parse_args()
    cikti = KOK / a.cikti
    for alt in ("images", "labels"):
        (cikti / a.bolum / alt).mkdir(parents=True, exist_ok=True)
    biten = cikti / f"{a.bolum}_biten.txt"
    yapildi = set(biten.read_text().split("\n")) if biten.exists() else set()
    dosyalar = sorted((f for f in HfApi().dataset_info(DEPO, files_metadata=True).siblings
                       if f.rfilename.startswith(f"{a.bolum}/imagery/UAS/") and f.rfilename.endswith(".tif")
                       and f.size <= a.maks_gb * 1e9), key=lambda f: f.size)
    print(len(dosyalar), "ortomozaik,", round(sum(f.size for f in dosyalar) / 1e9, 1), "GB", flush=True)
    sayac = {1: 0, 2: 0, 3: 0, 4: 0}
    for f in dosyalar:
        ad = f.rfilename.split("/")[-1]
        if ad in yapildi:
            continue
        try:
            e_yol = hf_hub_download(DEPO, f"{a.bolum}/annotations/UAS/building_damage_assessment/{ad}.json",
                                    repo_type="dataset", local_dir=HAM)
            etiketler = json.load(open(e_yol))
        except Exception as e:
            print(ad, "etiket yok:", e, flush=True)
            continue
        if not any(x["label"] in SINIF for x in etiketler):
            print(ad, "etiketli bina yok", flush=True)
            continue
        try:
            v_yol = hf_hub_download(DEPO, f"{a.bolum}/annotations/UAS/building_alignment_adjustments/{ad}.json",
                                    repo_type="dataset", local_dir=HAM)
            vektorler = json.load(open(v_yol))
        except Exception:
            vektorler = []
        tif = Path(hf_hub_download(DEPO, f.rfilename, repo_type="dataset", local_dir=HAM))
        try:
            n, s = isle(tif, etiketler, vektorler, cikti, a.bolum, a.hedef_bina, a.doseme, sayac)
            print(f"{ad}: {f.size / 1e9:.2f} GB, ölçek {s:.3f}, {n} döşeme, toplam {sayac}", flush=True)
        except Exception as e:
            print(ad, "HATA:", repr(e), flush=True)
        finally:
            os.remove(tif)
        with open(biten, "a") as b:
            b.write(ad + "\n")
    (cikti / f"{a.bolum}.yaml").write_text(
        f"path: {cikti.as_posix()}\ntrain: {a.bolum}/images\nval: {a.bolum}/images\ntest: {a.bolum}/images\n"
        "names: {0: person, 1: destroyed, 2: major-damage, 3: minor-damage, 4: no-damage}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
