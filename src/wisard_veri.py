"""WiSARD (Broyles, Hayner, Leung — IROS 2022; yaban arazisi arama-kurtarma) → veri/wisard_yolo (yalnız görünür ışık).

* Her uçuş bir klasör: <tarih>_<yer>_<platform>_VIS_<no>/ (IR klasörleri atlanır); kare .jpeg + YOLO .txt (sınıf 0 = insan).
  count.txt gibi YOLO olmayan dosyalar ve 5 alanlı olmayan satırlar atlanır.
* Bölme YER bazında (aynı yerdeki uçuşlar birbirine çok benzer): VALID_YER = val, kalanı train. Örnek paket (MtErie) val'de.
* Kareler 4K → 960 px döşeme (%15 örtüşme), `doseme_yap.dosele` ile. Ardışık kareler benzer → train her ADIM. kare.

Kullanım: python src/wisard_veri.py
"""
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from doseme_yap import dosele

KOK = Path(__file__).resolve().parent.parent
HAM = KOK / "veri" / "wisard_ham" / "tam"
HEDEF = KOK / "veri" / "wisard_yolo"
VALID_YER = {"MtErie"}
ADIM = {"train": 3, "valid": 2}


def ucuslar(kok):
    for d in sorted(p for p in kok.rglob("*") if p.is_dir() and "_VIS_" in p.name):
        kareler = sorted(d.glob("*.jp*g"))
        if kareler:
            yield d, kareler


if __name__ == "__main__":
    shutil.rmtree(HEDEF, ignore_errors=True)
    ozet = defaultdict(lambda: [0, 0])
    for d, kareler in ucuslar(HAM):
        yer = d.name.split("_")[1]
        bolum = "valid" if yer in VALID_YER else "train"
        with tempfile.TemporaryDirectory() as t:
            gecici = Path(t)
            (gecici / "images").mkdir(); (gecici / "labels").mkdir()
            for i, k in enumerate(kareler):
                if i % ADIM[bolum]:
                    continue
                et = k.with_suffix(".txt")
                if not et.exists():  # etiketsiz kare: tam etiketli değil olabilir → atla
                    continue
                satir = [s for s in et.read_text().split("\n") if len(s.split()) == 5 and s.split()[0] == "0"]
                shutil.copy(k, gecici / "images" / (k.stem + ".jpg"))
                (gecici / "labels" / (k.stem + ".txt")).write_text("\n".join(satir))
                ozet[bolum][0] += 1; ozet[bolum][1] += len(satir)
            dosele(gecici, HEDEF / bolum)
        print(d.name, bolum)
    for b, (n, k) in sorted(ozet.items()):
        print(f"{b}: {n} kare, {k} insan")
