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

# BGR; insan dikkat çeksin diye parlak camgöbeği, hasar yeşilden kırmızıya
RENKLER = {"person": (255, 255, 0), "destroyed": (0, 0, 220), "major-damage": (0, 120, 255),
           "minor-damage": (0, 220, 255), "no-damage": (60, 180, 60)}
TR = {"person": "insan", "destroyed": "yıkık", "major-damage": "ağır hasar",
      "minor-damage": "hafif hasar", "no-damage": "hasarsız"}


def tespit(goruntu_bgr: np.ndarray, model_yolu=VARSAYILAN_MODEL, conf=0.25, dilimle=False, dilim=640):
    """[(sınıf_adı, güven, x1, y1, x2, y2), ...] döndürür."""
    if dilimle:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        model = _sahi_model(str(model_yolu), conf)
        rgb = cv2.cvtColor(goruntu_bgr, cv2.COLOR_BGR2RGB)
        sonuc = get_sliced_prediction(rgb, model, slice_height=dilim, slice_width=dilim,
                                      overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        return [(p.category.name, p.score.value, *map(int, p.bbox.to_xyxy())) for p in sonuc.object_prediction_list]

    model = _yolo_model(str(model_yolu))
    r = model.predict(goruntu_bgr, conf=conf, imgsz=640, verbose=False)[0]
    return [(r.names[int(c)], float(s), *map(int, b)) for b, s, c in
            zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


_onbellek = {}


def _yolo_model(yol):
    from ultralytics import YOLO
    if ("yolo", yol) not in _onbellek:
        _onbellek[("yolo", yol)] = YOLO(yol)
    return _onbellek[("yolo", yol)]


def _sahi_model(yol, conf):
    from sahi import AutoDetectionModel
    anahtar = ("sahi", yol, conf)
    if anahtar not in _onbellek:
        _onbellek[anahtar] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yol, confidence_threshold=conf)
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
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dilimle", action="store_true", help="SAHI döşemeli çıkarım (küçük nesneler için)")
    ap.add_argument("--cikti", default=None)
    a = ap.parse_args()

    img = cv2.imread(a.kaynak)
    if img is None:
        raise SystemExit(f"Görüntü okunamadı: {a.kaynak}")
    t = tespit(img, a.model, a.conf, a.dilimle)
    cikti = a.cikti or str(Path(a.kaynak).with_name(Path(a.kaynak).stem + "_tespit.jpg"))
    cv2.imwrite(cikti, ciz(img, t))
    print(ozet(t))
    print("kaydedildi:", cikti)
