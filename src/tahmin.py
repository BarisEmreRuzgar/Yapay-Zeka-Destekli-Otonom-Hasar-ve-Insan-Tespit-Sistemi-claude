"""Görüntü üzerinde hasar + insan tespiti.

Büyük havadan görüntülerde insanlar birkaç piksel kalır; --dilimle ile SAHI döşemeli
çıkarım yapılır (görüntü örtüşen parçalara bölünür, sonuçlar birleştirilir).

Kullanım:
  python src/tahmin.py foto.jpg
  python src/tahmin.py foto.jpg --dilimle --model modeller/disastervision.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
VARSAYILAN_MODEL = KOK / "modeller" / "disastervision.pt"
# Yalnız insan sınıfı, 1280 px'te VisDrone + arama-kurtarma (drone-leak, SARD) ile eğitildi.
# Val insan AP50 (ana model → uzman v2): VisDrone 0.607 → 0.690, SAR 0.260 → 0.890, SARD 0.315 → 0.864;
# harici HERIDAL (eğitimde yok) 0.253 → 0.628.
INSAN_MODEL = KOK / "modeller" / "insan_uzman.pt"
INSAN_BOYUT, INSAN_DILIM = 1280, 960

# BGR; insan dikkat çeksin diye parlak camgöbeği, hasar yeşilden kırmızıya
RENKLER = {"person": (255, 255, 0), "destroyed": (0, 0, 220), "major-damage": (0, 120, 255),
           "minor-damage": (0, 220, 255), "no-damage": (60, 180, 60)}
TR = {"person": "insan", "destroyed": "yıkık", "major-damage": "ağır hasar",
      "minor-damage": "hafif hasar", "no-damage": "hasarsız"}


def cikarim_boyutu(uzun_kenar: int) -> int:
    """Modelin en iyi çalıştığı ölçek: görüntünün ~1.33 katı (eğitimdeki ölçeğe yakın), 640–1280 arası.
    Ölçüm: VisDrone 960 px döşemelerde insan AP50 640→0.343, 960→0.380, 1280→0.407;
    xBD 512 px döşemelerde hasar mAP50 640→0.410, 960→0.389, 1280→0.336. Sabit 640 ikisinden birini bozar."""
    return int(min(1280, max(640, round(uzun_kenar * 4 / 3 / 32) * 32)))


def tespit(goruntu_bgr: np.ndarray, model_yolu=VARSAYILAN_MODEL, conf=0.3, dilimle=False, dilim=640,
           insan_modeli=INSAN_MODEL):
    """[(sınıf_adı, güven, x1, y1, x2, y2), ...] döndürür. insan_modeli verilirse (ve dosya varsa) insanlar
    ana modelden değil, ondan alınır; hasar her zaman ana modelden gelir."""
    t = _calistir(goruntu_bgr, model_yolu, conf, dilimle, dilim, None)
    if insan_modeli and Path(insan_modeli).exists():
        t = [x for x in t if x[0] != "person"] + _calistir(goruntu_bgr, insan_modeli, conf, dilimle, INSAN_DILIM, INSAN_BOYUT)
    return t


def _calistir(goruntu_bgr, model_yolu, conf, dilimle, dilim, boyut):
    if dilimle:
        from sahi.predict import get_sliced_prediction

        model = _sahi_model(str(model_yolu), conf, boyut or cikarim_boyutu(dilim))
        rgb = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2RGB)
        sonuc = get_sliced_prediction(rgb, model, slice_height=dilim, slice_width=dilim,
                                      overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        return [(p.category.name, p.score.value, *map(int, p.bbox.to_xyxy())) for p in sonuc.object_prediction_list]

    model = _yolo_model(str(model_yolu))
    boyut = boyut or cikarim_boyutu(max(goruntu_bgr.shape[:2]))
    r = model.predict(goruntu_bgr, conf=conf, imgsz=boyut, verbose=False)[0]
    return [(r.names[int(c)], float(s), *map(int, b)) for b, s, c in
            zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


_onbellek = {}


def _yolo_model(yol):
    from ultralytics import YOLO
    if ("yolo", yol) not in _onbellek:
        _onbellek[("yolo", yol)] = YOLO(yol)
    return _onbellek[("yolo", yol)]


def _sahi_model(yol, conf, boyut):
    from sahi import AutoDetectionModel
    anahtar = ("sahi", yol, conf, boyut)
    if anahtar not in _onbellek:
        _onbellek[anahtar] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yol, confidence_threshold=conf, image_size=boyut)
    return _onbellek[anahtar]


def ciz(goruntu_bgr, tespitler):
    cikti = goruntu_bgr.copy()
    kalinlik = max(1, round(sum(cikti.shape[:2]) / 1200))
    for ad, guven, x1, y1, x2, y2 in tespitler:
        renk = RENKLER.get(ad, (255, 255, 255))
        cv2.rectangle(cikti, (x1, y1), (x2, y2), renk, kalinlik)
        etiket = f"{ad} {guven:.2f}"
        (w, h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * kalinlik, 1)
        cv2.rectangle(cikti, (x1, max(0, y1 - h - 4)), (x1 + w + 2, y1), renk, -1)
        cv2.putText(cikti, etiket, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * kalinlik, (0, 0, 0), 1)
    return cikti


def ozet(tespitler):
    say = Counter(t[0] for t in tespitler)
    return {TR[k]: say.get(k, 0) for k in RENKLER}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynak")
    ap.add_argument("--model", default=str(VARSAYILAN_MODEL))
    ap.add_argument("--insan-modeli", default=str(INSAN_MODEL), help='"" verilirse insanlar da ana modelden')
    ap.add_argument("--conf", type=float, default=0.3, help="val F1 eşik taraması: VisDrone 0.28, SAR 0.44, SARD 0.41 en iyi")
    ap.add_argument("--dilimle", action="store_true", help="SAHI döşemeli çıkarım (küçük nesneler için)")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    img = cv2.imread(a.kaynak)
    if img is None:
        raise SystemExit(f"Görüntü okunamadı: {a.kaynak}")
    t = tespit(img, a.model, a.conf, a.dilimle, insan_modeli=a.insan_modeli)
    cikti = a.cikti or str(Path(a.kaynak).with_name(Path(a.kaynak).stem + "_tespit.jpg"))
    cv2.imwrite(cikti, ciz(img, t))
    print(ozet(t))
    print("kaydedildi:", cikti)
