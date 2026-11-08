"""Hugging Face statik Space'i hazırlar: modeller tarayıcıda (onnxruntime-web, WebGPU/WASM) çalışır, sunucu gerekmez.

Klasöre web_demo/index.html, ONNX'e çevrilmiş modeller (ana model, insan uzmanı, su modeli) ve örnek görüntüler kopyalanır.
Tarayıcı sürümü Python'daki `src/tahmin.py` mantığını izler: hasar ana modelden (görüntünün ~1.33 katı ölçekte), insanlar
uzmandan (1280 px); "döşemeli çıkarım" SAHI gibi %20 örtüşen döşemeler + tam görüntü ve GREEDYNMM (IOS 0.5) birleştirmesi.
Örneklerde Python ile karşılaştırma: drone 68/68 insan (döşemeli 63/63); uydu kutularında ±1–4 fark (ölçekleme enterpolasyonu).

Kullanım (onnx paketi gerekir: pip install onnx onnxslim onnxruntime):
  python src/hf_space_hazirla.py HEDEF_KLASOR
  hf auth login                                                         # bir kez; "write" yetkili token
  hf repos create KULLANICI/disastervision --repo-type space --space-sdk static
  hf upload KULLANICI/disastervision HEDEF_KLASOR . --repo-type space
"""
import shutil
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
MODELLER = ["disastervision", "insan_uzman", "insan_su"]
ORNEKLER = ["drone_kalabalik.jpg", "uydu_hasar.jpg", "palu_sonra.jpg"]
REPO = "https://github.com/BarisEmreRuzgar/Yapay-Zeka-Destekli-Otonom-Hasar-ve-Insan-Tespit-Sistemi-claude"
BASLIK = f"""---
title: DisasterVision
emoji: 🌍
colorFrom: red
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Drone/uydu görüntüsünde bina hasarı ve insan tespiti
---

# DisasterVision — afet hasarı ve insan tespiti

Havadan (İHA/uydu) görüntüde binaları 4 hasar seviyesinde sınıflar ve insanları işaretler (YOLOv8s).
Modeller ziyaretçinin tarayıcısında çalışır (onnxruntime-web; WebGPU varsa ekran kartı, yoksa işlemci).
Kaynak kod, eğitim ve ölçümler: {REPO}
"""


def hazirla(hedef):
    from ultralytics import YOLO
    shutil.rmtree(hedef, ignore_errors=True)
    (hedef / "modeller").mkdir(parents=True)
    (hedef / "ornekler").mkdir()
    shutil.copy(KOK / "web_demo" / "index.html", hedef / "index.html")
    for f in ORNEKLER:
        shutil.copy(KOK / "ornekler" / f, hedef / "ornekler" / f)
    for m in MODELLER:
        pt = hedef / "modeller" / f"{m}.pt"
        shutil.copy(KOK / "modeller" / f"{m}.pt", pt)
        YOLO(str(pt)).export(format="onnx", dynamic=True, simplify=True, opset=17, imgsz=640, device="cpu")
        pt.unlink()
    (hedef / "README.md").write_text(BASLIK, encoding="utf-8")
    (hedef / ".gitattributes").write_text("*.onnx filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n")
    dosyalar = [p for p in hedef.rglob("*") if p.is_file()]
    print(f"{hedef}: {len(dosyalar)} dosya, {sum(p.stat().st_size for p in dosyalar) / 1e6:.0f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    hazirla(Path(sys.argv[1]))
