"""Okutama-Action (okutama-action.org; Barekatain vd. 2017, CVPRW) → veri/okutama_yolo.

Tam etiketli drone insan verisi (10–45 m irtifa, park/saha sahneleri), VisDrone'dan bağımsız.
* Kareler 1280x720, etiketler 3840x2160 koordinatında → 3'e bölünür (görsel olarak doğrulandı; kare N ↔ N.jpg).
* Etiket: Labels/SingleActionLabels/3840x2160/<video>.txt — "id x1 y1 x2 y2 kare lost occluded generated "Person" ...".
  lost=1 (görüş dışı) atılır. Her 180 karede iz kimlikleri yenilendiği için o karelerde aynı kişi iki kez yazılı →
  IoU>0.7 kutular tekilleştirilir.
* Bölme video/senaryo bazında: resmi test seti = test; eğitim videolarından VALID_SENARYO (iki drone aynı anda aynı
  sahneyi çektiği için senaryonun iki drone videosu birlikte) = valid; kalanı train. Ardışık kareler neredeyse aynı →
  her ADIM. kare.

Kullanım: python src/okutama_veri.py
"""
import shutil
from collections import defaultdict
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham" / "okutama"
HEDEF = KOK / "veri" / "okutama_yolo"
VALID_SENARYO = {"1.5", "2.6"}  # 1.1.5+2.1.5 (sabah), 1.2.6+2.2.6 (öğlen) → 4 video
ADIM = {"train": 10, "valid": 15, "test": 15}
W, H, OLCEK = 1280, 720, 3.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    k = ix * iy
    return k / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - k + 1e-9)


def videolar(kok):
    """{video: (kare_klasoru, etiket_dosyasi)}"""
    kareler = {p.name: p for p in kok.rglob("Extracted-Frames-1280x720/*") if p.is_dir()}
    etiketler = {p.stem: p for p in kok.rglob("SingleActionLabels/3840x2160/*.txt")}
    return {v: (kareler[v], etiketler[v]) for v in kareler if v in etiketler}


def senaryo(video):  # "1.2.10" → drone 1, zaman 2, senaryo 10 → "2.10"
    return ".".join(video.split(".")[1:])


if __name__ == "__main__":
    shutil.rmtree(HEDEF, ignore_errors=True)
    for b in ("train", "valid", "test"):
        for alt in ("images", "labels"):
            (HEDEF / b / alt).mkdir(parents=True)
    isler = [(v, kd, et, "test") for v, (kd, et) in videolar(HAM / "test").items()]
    isler += [(v, kd, et, "valid" if senaryo(v) in VALID_SENARYO else "train") for v, (kd, et) in videolar(HAM / "train").items()]
    sayac = defaultdict(lambda: [0, 0, 0])
    for v, kd, et, b in sorted(isler):
        kutular = defaultdict(list)
        for s in et.read_text().splitlines():
            p = s.split()
            if len(p) < 7 or p[6] != "0":
                continue
            x1, y1, x2, y2 = (min(max(int(t) / OLCEK, 0), lim) for t, lim in zip(p[1:5], (W, H, W, H)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            k = (x1, y1, x2, y2)
            if all(iou(k, o) <= 0.7 for o in kutular[int(p[5])]):
                kutular[int(p[5])].append(k)
        for kare in sorted(int(f.stem) for f in kd.glob("*.jpg")):
            if kare % ADIM[b]:
                continue
            ad = f"okutama_{v}_{kare:05d}"
            shutil.copy(kd / f"{kare}.jpg", HEDEF / b / "images" / f"{ad}.jpg")
            satir = [f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                     for x1, y1, x2, y2 in kutular.get(kare, [])]
            (HEDEF / b / "labels" / f"{ad}.txt").write_text("\n".join(satir))
            sayac[b][0] += 1; sayac[b][1] += len(satir); sayac[b][2] += 0 if satir else 1
    for b, (n, k, bos) in sorted(sayac.items()):
        print(f"{b}: {n} kare, {k} insan, {bos} boş kare")
