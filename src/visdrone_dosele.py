"""VisDrone görüntülerini 960 px döşemelere böler → veri/visdrone_yolo/{train,valid}.

VisDrone görüntüleri ~2000 px; 640'a küçültülünce insanlar birkaç piksele iniyor.
Döşemenin içinde görünür alanı %40'tan az kalan kutular atılır; insan içermeyen döşemeler alınmaz.
Önce `python src/veri_indir.py --sadece visdrone` çalıştırılmış olmalı.
"""
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "ham" / "VisDrone"
CIKTI = KOK / "veri" / "visdrone_yolo"
DOSEME = 960
BOLUMLER = {"train": "VisDrone2019-DET-train", "valid": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}


def baslangiclar(uzunluk):
    if uzunluk <= DOSEME:
        return [0]
    adim = (uzunluk - DOSEME) / -(-(uzunluk - DOSEME) // DOSEME)  # eşit aralıklı, örtüşmeli
    return sorted({round(i * adim) for i in range(int((uzunluk - DOSEME) / adim) + 1)})


def main():
    for bolum, klasor in BOLUMLER.items():
        if (CIKTI / bolum / "labels").exists() and any((CIKTI / bolum / "labels").iterdir()):
            print(bolum, "zaten var, atlandı")
            continue
        (CIKTI / bolum / "images").mkdir(parents=True, exist_ok=True)
        (CIKTI / bolum / "labels").mkdir(parents=True, exist_ok=True)
        adet = 0
        for lbl in sorted((HAM / klasor / "labels").glob("*.txt")):
            satirlar = [list(map(float, s.split()[1:])) for s in lbl.read_text().splitlines() if s.strip()]
            if not satirlar:
                continue
            img = Image.open(HAM / klasor / "images" / (lbl.stem + ".jpg"))
            W, H = img.size
            kutular = [((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H) for cx, cy, w, h in satirlar]
            for oy in baslangiclar(H):
                for ox in baslangiclar(W):
                    gw, gh = min(DOSEME, W - ox), min(DOSEME, H - oy)
                    yeni = []
                    for x1, y1, x2, y2 in kutular:
                        kx1, ky1 = max(x1, ox), max(y1, oy)
                        kx2, ky2 = min(x2, ox + gw), min(y2, oy + gh)
                        if kx2 <= kx1 or ky2 <= ky1:
                            continue
                        if (kx2 - kx1) * (ky2 - ky1) < 0.4 * (x2 - x1) * (y2 - y1):
                            continue
                        yeni.append(f"0 {((kx1 + kx2) / 2 - ox) / gw:.6f} {((ky1 + ky2) / 2 - oy) / gh:.6f} "
                                    f"{(kx2 - kx1) / gw:.6f} {(ky2 - ky1) / gh:.6f}")
                    if not yeni:
                        continue
                    ad = f"{lbl.stem}_{oy}_{ox}"
                    img.crop((ox, oy, ox + gw, oy + gh)).save(CIKTI / bolum / "images" / f"{ad}.jpg", quality=92)
                    (CIKTI / bolum / "labels" / f"{ad}.txt").write_text("\n".join(yeni) + "\n")
                    adet += 1
        print(bolum, adet, "döşeme")
    # İnsan tespitinin genelleme testi (test-dev eğitimde hiç kullanılmaz)
    (KOK / "veri" / "visdrone_test.yaml").write_text(
        f"path: {CIKTI.as_posix()}\ntrain: test/images\nval: test/images\ntest: test/images\n"
        "names: {0: person, 1: destroyed, 2: major-damage, 3: minor-damage, 4: no-damage}\n")


if __name__ == "__main__":
    main()
