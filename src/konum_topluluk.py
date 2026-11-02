"""İki konum modelinin kayıtlı val tahminlerini ağırlıklı kutu birleştirme (WBF) ile birleştirir (yalnız CPU).

Her iki pickle'da aynı döşemeler için kon_kutu / kon_skor / kon_olas (aynı sınıflandırıcı topluluğu) ve tek-model
kutuları vardır. WBF: iki modelin kutuları skor sırasıyla kümelere atanır (IoU >= esik); küme kutusu skor ağırlıklı
ortalama, skoru toplam/model sayısı (tek modelin bulduğu kutu yarı skorla kalır), sınıf olasılığı skor ağırlıklı ortalama.
Kullanım: python src/konum_topluluk.py runs/deg/val_son.pkl runs/deg/val_bina_m.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iki_asama import iou, puanla  # noqa: E402


def wbf(kutular, skorlar, olaslar, n_model, esik=0.55):
    sira = np.argsort(-skorlar)
    kumeler = []  # [kutu_toplam, skor_toplam, olas_toplam, temsil_kutu, adet]
    for i in sira:
        k, s, p = kutular[i], skorlar[i], olaslar[i]
        if kumeler:
            temsil = np.array([c[3] for c in kumeler])
            o = iou(k[None], temsil)[0]
            j = int(o.argmax())
            if o[j] >= esik:
                c = kumeler[j]
                c[0] += k * s; c[1] += s; c[2] += p * s; c[4] += 1
                c[3] = c[0] / c[1]
                continue
        kumeler.append([k * s, s, p * s, k.copy(), 1])
    if not kumeler:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, olaslar.shape[1] if olaslar.ndim == 2 else 4))
    K = np.array([c[0] / c[1] for c in kumeler])
    S = np.array([c[1] / max(n_model, c[4]) for c in kumeler])
    P = np.array([c[2] / c[1] for c in kumeler])
    return K, S, P


def birlestir(a, b, esik):
    ham = {}
    for g, da in a.items():
        db = b.get(g)
        if db is None:
            continue
        d = dict(da)
        K = np.r_[da["kon_kutu"].reshape(-1, 4), db["kon_kutu"].reshape(-1, 4)]
        S = np.r_[da["kon_skor"], db["kon_skor"]]
        P = np.r_[da["kon_olas"].reshape(-1, 4), db["kon_olas"].reshape(-1, 4)]
        d["kon_kutu"], d["kon_skor"], d["kon_olas"] = wbf(K, S, P, 2, esik) if len(K) else (K, S, P)
        ham[g] = d
    return ham


if __name__ == "__main__":
    a = pickle.load(open(sys.argv[1], "rb"))
    b = pickle.load(open(sys.argv[2], "rb"))
    ortak = set(a) & set(b)
    print(len(ortak), "ortak döşeme")
    a = {g: a[g] for g in ortak}; b = {g: b[g] for g in ortak}
    print("== A"); puanla(a, ["iki+tek"])
    print("== B"); puanla(b, ["iki+tek"])
    for esik in (0.5, 0.6):
        print(f"== WBF esik={esik}"); puanla(birlestir(a, b, esik), ["iki+tek"])
