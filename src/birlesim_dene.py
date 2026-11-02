"""Kaydedilmiş ham tahminler (iki_asama --kaydet) üzerinde birleştirme yöntemlerini dener (yalnız CPU).

Yöntemler:
  nms      : bugünkü iki+tek (konum kutuları + tek-model kutuları, ağırlıklı skor, NMS)
  yumusak  : konum kutusunun j. sınıf olasılığı, örtüşen (IoU>=esik) tek-model kutusunun j. sınıf skoruyla harmanlanır:
             p' = (1-w)·p_j + w·s_tek ; sonra yine eşleşmeyen tek kutuları eklenip NMS.
Kullanım: python src/birlesim_dene.py runs/deg/val_son.pkl
"""
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iki_asama import HASAR, ap50, iou, nms  # noqa: E402


def sinif_kutulari(d, j, w_tek, w_yum, esik=0.3):
    m = d["tek_sinif"] == j + 1
    tk, ts = d["tek_kutu"][m].reshape(-1, 4), d["tek_skor"][m]
    kk = d["kon_kutu"].reshape(-1, 4)
    p = d["kon_olas"][:, j].copy() if len(kk) else np.zeros(0)
    if w_yum and len(kk) and len(tk):
        M = iou(kk, tk)
        en = np.where(M >= esik, M, 0) * ts[None, :]  # örtüşen tek kutularının skoru
        s = en.max(1)
        p = (1 - w_yum) * p + w_yum * s
    K = np.r_[kk, tk]
    S = np.r_[d["kon_skor"] * p if len(kk) else np.zeros(0), ts * w_tek]
    t = nms(K, S) if len(K) else np.zeros(0, int)
    return K[t], S[t]


def degerlendir(ham, j, w_tek, w_yum):
    h = HASAR[j]
    gercek = {g: d["gt"][d["gc"] == j + 1] for g, d in ham.items()}
    tahmin = []
    for g, d in ham.items():
        K, S = sinif_kutulari(d, j, w_tek, w_yum)
        tahmin += [(g, float(s), k) for k, s in zip(K, S)]
    return ap50(tahmin, gercek)


if __name__ == "__main__":
    ham = pickle.load(open(sys.argv[1], "rb"))
    print(len(ham), "döşeme")
    ozet = {}
    for j, h in enumerate(HASAR):
        sonuc = {}
        for w_tek, w_yum in product([0.3, 0.6, 1.0], [0.0, 0.2, 0.4, 0.6]):
            sonuc[(w_tek, w_yum)] = degerlendir(ham, j, w_tek, w_yum)
        en = max(sonuc, key=sonuc.get)
        ozet[h] = (sonuc[(0.6, 0.0)], en, sonuc[en])
        print(h, " ".join(f"{k}:{v:.3f}" for k, v in sonuc.items()), flush=True)
    print("\nsınıf: bugünkü(0.6,0) -> en iyi (w_tek, w_yum)")
    for h, (b, en, v) in ozet.items():
        print(f"{h}: {b:.3f} -> {en} {v:.3f}")
    print(f"mAP50: {np.mean([o[0] for o in ozet.values()]):.3f} -> {np.mean([o[2] for o in ozet.values()]):.3f}")
