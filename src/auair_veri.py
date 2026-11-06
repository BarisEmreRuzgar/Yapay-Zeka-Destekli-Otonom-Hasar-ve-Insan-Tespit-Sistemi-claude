"""AU-AIR (Bozcan & Kayacan, ICRA 2020; CC BY 4.0) → veri/auair_yolo, yalnız insan sınıfı.

Alçak irtifa (≈5–30 m) trafik gözetimi drone videoları, 1920x1080; 8 sınıf tam etiketli, burada yalnız 'Human' (0) alınır.
* Etiket: annotations.json → her kare için bbox listesi {top, left, height, width, class} (piksel).
* Kare adı frame_<videoBaşlangıcı>_x_<sıra>.jpg → video = ikinci alan. Bölme video bazında (ardışık kareler neredeyse aynı):
  insan içeren videoların ~%15'i valid, kalanı train. İnsanlı karelerin her ADIM'ıncısı, insansız karelerin her
  BOS_ADIM'ıncısı (tam etiketli olduğu için gerçek negatif) alınır.

Kullanım: python src/auair_veri.py
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "auair_ham"
HEDEF = KOK / "veri" / "auair_yolo"
ADIM, BOS_ADIM = 2, 40
VALID_ORAN = 0.15

if __name__ == "__main__":
    ann = json.loads(next(HAM.rglob("annotations.json")).read_text(encoding="utf-8"))["annotations"]
    resimler = {p.name: p for p in HAM.rglob("*.jpg")}
    video = defaultdict(list)
    for a in ann:
        video[a["image_name"].split("_")[1]].append(a)
    insanli = sorted(v for v, l in video.items() if any(b["class"] == 0 for a in l for b in a["bbox"]))
    random.Random(0).shuffle(insanli)
    hedef_insan = VALID_ORAN * sum(b["class"] == 0 for l in video.values() for a in l for b in a["bbox"])
    valid, n = set(), 0
    for v in insanli:  # insan sayısı ~%15'e ulaşana kadar video ekle
        if n >= hedef_insan:
            break
        valid.add(v)
        n += sum(b["class"] == 0 for a in video[v] for b in a["bbox"])

    shutil.rmtree(HEDEF, ignore_errors=True)
    for b in ("train", "valid"):
        for alt in ("images", "labels"):
            (HEDEF / b / alt).mkdir(parents=True)
    sayac = defaultdict(lambda: [0, 0, 0])
    eksik = 0
    for v, liste in sorted(video.items()):
        bolum = "valid" if v in valid else "train"
        liste.sort(key=lambda a: a["image_name"])
        i_ins = i_bos = 0
        for a in liste:
            W, H = a.get("image_width:", a.get("image_width", 1920.0)), a["image_height"]
            kut = [b for b in a["bbox"] if b["class"] == 0 and b["width"] > 1 and b["height"] > 1]
            if kut:
                i_ins += 1
                if i_ins % ADIM:
                    continue
            else:
                i_bos += 1
                if i_bos % BOS_ADIM:
                    continue
            kaynak = resimler.get(a["image_name"])
            if kaynak is None:
                eksik += 1
                continue
            ad = "auair_" + Path(a["image_name"]).stem
            shutil.copy(kaynak, HEDEF / bolum / "images" / f"{ad}.jpg")
            satir = []
            for b in kut:
                x1, y1 = max(b["left"], 0), max(b["top"], 0)
                x2, y2 = min(b["left"] + b["width"], W), min(b["top"] + b["height"], H)
                satir.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            (HEDEF / bolum / "labels" / f"{ad}.txt").write_text("\n".join(satir))
            sayac[bolum][0] += 1; sayac[bolum][1] += len(satir); sayac[bolum][2] += 0 if satir else 1
    print(f"video: {len(video)}, insanlı {len(insanli)}, valid {len(valid)}; eksik resim {eksik}")
    for b, (n, k, bos) in sorted(sayac.items()):
        print(f"{b}: {n} kare, {k} insan, {bos} boş kare")
