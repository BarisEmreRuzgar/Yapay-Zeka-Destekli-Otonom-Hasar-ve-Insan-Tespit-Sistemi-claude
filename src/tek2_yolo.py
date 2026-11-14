"""YOLO biçimli bir hasar setinde (ör. RescueNet val) öncesiz iki aşamayı değerlendirir.

  tek      : dedektörün kendi hasar sınıfları
  tek2     : dedektör kutuları sınıf-bağımsız konum olarak (skor = en yüksek sınıf güveni) + tek sınıflandırıcı
  tek2+tek : ikisinin sınıf bazında birleşimi (iki_asama.birlesik ile aynı mantık)
AP50 hesabı iki_asama.ap50 ile aynı (COCO 101 nokta).

  python src/tek2_yolo.py --veri veri/rescuenet_yolo/valid --dedektor runs/hasar_v2/weights/best.pt \
         --siniflandirici runs/tek2d/weights/best.pt --imgsz 1024
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from iki_asama import HASAR, TEK_AGIRLIK, Topluluk, ap50, nms

KOK = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", required=True, help="images/ ve labels/ içeren klasör")
    ap.add_argument("--dedektor", required=True)
    ap.add_argument("--siniflandirici", default="", help="virgülle; boşsa yalnız 'tek'")
    ap.add_argument("--sicaklik", default="")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tek-agirlik", type=float, default=TEK_AGIRLIK)
    a = ap.parse_args()
    k = KOK / a.veri
    det = YOLO(a.dedektor)
    sinif = Topluluk(a.siniflandirici, a.sicaklik) if a.siniflandirici else None
    gercek = {h: {} for h in HASAR}
    tah = {kurgu: {h: [] for h in HASAR} for kurgu in ("tek", "tek2", "tek2+tek")}
    for e in sorted((k / "labels").glob("*.txt")):
        g = cv2.imread(str(k / "images" / (e.stem + ".jpg")))
        H, W = g.shape[:2]
        s = np.array([list(map(float, x.split())) for x in e.read_text().splitlines() if x.strip()]).reshape(-1, 5)
        kut = np.c_[(s[:, 1] - s[:, 3] / 2) * W, (s[:, 2] - s[:, 4] / 2) * H, (s[:, 1] + s[:, 3] / 2) * W,
                    (s[:, 2] + s[:, 4] / 2) * H]
        for j, h in enumerate(HASAR):
            gercek[h][e.stem] = kut[s[:, 0] == j + 1]
        r = det.predict(g, conf=0.001, iou=0.7, imgsz=a.imgsz, verbose=False, classes=[1, 2, 3, 4])[0]
        K, S, C = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        for j, h in enumerate(HASAR):
            m = C == j + 1
            tah["tek"][h] += [(e.stem, float(si), ki) for ki, si in zip(K[m], S[m])]
        if sinif is None:
            continue
        # konum: sınıf-bağımsız NMS ile tek kutu kümesi
        t = nms(K, S, 0.6) if len(K) else np.zeros(0, int)
        KK, SS = K[t][:300], S[t][:300]
        P = sinif(None, g[..., ::-1], KK) if len(KK) else np.zeros((0, 4))
        for j, h in enumerate(HASAR):
            tah["tek2"][h] += [(e.stem, float(si * p), ki) for ki, si, p in zip(KK, SS, P[:, j])]
            m = C == j + 1
            KB = np.r_[KK.reshape(-1, 4), K[m].reshape(-1, 4)]
            SB = np.r_[SS * (P[:, j] if len(P) else 0), S[m] * a.tek_agirlik]
            tb = nms(KB, SB) if len(KB) else np.zeros(0, int)
            tah["tek2+tek"][h] += [(e.stem, float(si), ki) for ki, si in zip(KB[tb], SB[tb])]
    sonuc = {}
    for kurgu, t in tah.items():
        if not any(t.values()):
            continue
        ap_ = {h: ap50(t[h], gercek[h]) for h in HASAR}
        sonuc[kurgu] = {"mAP50": float(np.mean(list(ap_.values()))), "AP50": ap_}
        print(f"{kurgu:9s} mAP50={sonuc[kurgu]['mAP50']:.3f}  " + "  ".join(f"{h}={v:.3f}" for h, v in ap_.items()))
    (KOK / "runs" / f"tek2_yolo_{Path(a.veri).parent.name}_{Path(a.veri).name}.json").write_text(
        json.dumps(sonuc, indent=1, ensure_ascii=False))
