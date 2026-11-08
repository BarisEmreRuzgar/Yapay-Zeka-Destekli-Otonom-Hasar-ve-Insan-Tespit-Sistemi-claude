"""NOMAD (Russell Bernal, Scheirer, Cleland-Huang — WACV 2024; gizlenen/yatan insanlar, 10–90 m) → veri/nomad_yolo.

* Etiket: annotations.json (Detectron2 biçimi, bbox_mode=1 → x, y, w, h piksel), kare 5472×3078, sınıf 0 = insan,
  her kutuda 'visibility' (%10…%100 görünür). Kareler Drive'daki ActorXXX-ActorYYY.zip'lerden.
* Görünürlüğü GORUNURLUK_ALT'tan düşük kişi içeren kareler atlanır: %10–20 görünen kişi pratikte tamamen saklı
  (etiket bile tahmine yakın); bunlarla eğitmek yaprak/gölgede insan aramayı öğretir.
* Bölme OYUNCU bazında (aynı kişi + aynı yer train/val'e bölünmesin): VALID_OYUNCU = val, kalan train.
* 960 px döşeme (%15 örtüşme); train'de boş döşemelerin %10'u.

Kullanım: python src/nomad_veri.py
"""
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from doseme_yap import dosele

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "nomad_ham"
HEDEF = KOK / "veri" / "nomad_yolo"
VALID_OYUNCU = {"Actor009", "Actor010"}
GORUNURLUK_ALT = 30
ADIM = {"train": 3, "valid": 4}
BOS_ORAN = {"train": 0.10, "valid": 1.0}

if __name__ == "__main__":
    ann = json.loads((HAM / "annotations.json").read_text(encoding="utf-8"))
    resimler = {p.name: p for p in (HAM / "acik").rglob("*.jpg")}
    print("açılmış kare:", len(resimler))
    shutil.rmtree(HEDEF, ignore_errors=True)
    gruplar = defaultdict(list)  # (oyuncu, irtifa) → kareler
    for x in ann:
        if x["file_name"] in resimler:
            oyuncu, irtifa = x["file_name"].split("_")[:2]
            gruplar[(oyuncu, irtifa)].append(x)
    ozet = defaultdict(lambda: [0, 0, 0])
    for (oyuncu, irtifa), liste in sorted(gruplar.items()):
        bolum = "valid" if oyuncu in VALID_OYUNCU else "train"
        liste.sort(key=lambda x: x["file_name"])
        with tempfile.TemporaryDirectory() as t:
            g = Path(t)
            (g / "images").mkdir(); (g / "labels").mkdir()
            for i, x in enumerate(liste):
                if i % ADIM[bolum]:
                    continue
                if any(int(a["visibility"]) < GORUNURLUK_ALT for a in x["annotations"]):
                    ozet[bolum][2] += 1
                    continue
                W, H = x["width"], x["height"]
                satir = []
                for a in x["annotations"]:
                    bx, by, bw, bh = a["bbox"]
                    satir.append(f"0 {(bx + bw / 2) / W:.6f} {(by + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}")
                ad = Path(x["file_name"]).stem
                shutil.copy(resimler[x["file_name"]], g / "images" / f"{ad}.jpg")
                (g / "labels" / f"{ad}.txt").write_text("\n".join(satir))
                ozet[bolum][0] += 1; ozet[bolum][1] += len(satir)
            dosele(g, HEDEF / bolum, bos_oran=BOS_ORAN[bolum])
    for b, (n, k, atl) in sorted(ozet.items()):
        print(f"{b}: {n} kare, {k} insan; görünürlük <%{GORUNURLUK_ALT} nedeniyle {atl} kare atlandı")
