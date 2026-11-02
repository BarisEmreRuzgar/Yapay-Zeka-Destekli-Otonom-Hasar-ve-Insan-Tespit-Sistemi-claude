"""Afet türüne göre sınıf olasılığı kalibrasyonu (yalnız CPU, kayıtlı val tahminleri üzerinde).

xBD'de etiketleme alışkanlığı afet türüne göre değişiyor (ör. kasırgada sel hasarı çatıdan görünmüyor). Uygulamada
kullanıcı afet türünü bilir; iki aşamanın olasılıkları tür başına 4 çarpanla (p_j · w_tür,j) ölçeklenir.
Çarpanlar val'in bir yarısında koordinat aramasıyla öğrenilir, diğer yarısında ölçülür (çapraz).
Kullanım: python src/tur_kalibrasyon.py runs/deg/val_son.pkl
"""
import contextlib
import io
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iki_asama import HASAR, ap50, birlesik, puanla  # noqa: E402

TUR = {"hurricane": "kasirga", "flooding": "sel", "fire": "yangin", "wildfire": "yangin", "bushfire": "yangin",
       "earthquake": "deprem_tsunami", "tsunami": "deprem_tsunami", "tornado": "hortum", "volcano": "volkan"}
TURLER = sorted(set(TUR.values()))
ADIMLAR = [0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0]


def tur(g):
    afet = g.split("_")[1]
    return next(v for k, v in TUR.items() if k in afet)


def uygula(ham, W):
    out = {}
    for g, d in ham.items():
        d2 = dict(d)
        if len(d["kon_kutu"]):
            d2["kon_olas"] = d["kon_olas"] * W[tur(g)][None, :]
        out[g] = d2
    return out


def map50(ham, W):
    with contextlib.redirect_stdout(io.StringIO()):
        return puanla(uygula(ham, W), ["iki+tek"])["iki+tek"]["mAP50"]


def sinif_ap(ham, j, w):
    """j. sınıfın AP50'si; w: tür → çarpan (çarpan j yalnız bu sınıfı etkiler, sınıflar bağımsız aranır)."""
    tahmin, gercek = [], {}
    for g, d in ham.items():
        gercek[g] = d["gt"][d["gc"] == j + 1]
        d2 = d
        if len(d["kon_kutu"]):
            d2 = dict(d); P = d["kon_olas"].copy(); P[:, j] *= w[tur(g)]; d2["kon_olas"] = P
        K, S = birlesik(d2, j)
        tahmin += [(g, float(si), ki) for ki, si in zip(K, S)]
    return ap50(tahmin, gercek)


def ogren(ham, tur_sayisi=2):
    W = {t: np.ones(4) for t in TURLER}
    aplar = []
    for j in range(4):
        w = {t: 1.0 for t in TURLER}
        en = sinif_ap(ham, j, w)
        for _ in range(tur_sayisi):
            for t in TURLER:
                for c in ADIMLAR:
                    w2 = dict(w); w2[t] = c
                    m = sinif_ap(ham, j, w2)
                    if m > en + 1e-4:
                        en, w = m, w2
        for t in TURLER:
            W[t][j] = w[t]
        aplar.append(en)
        print(f"  {HASAR[j]}: {en:.3f}", flush=True)
    return W, float(np.mean(aplar))


if __name__ == "__main__":
    ham = pickle.load(open(sys.argv[1], "rb"))
    # görüntü adına göre iki kat (aynı görüntünün döşemeleri aynı katta)
    kat = {g: random.Random("kat_" + "_".join(g.split("_")[1:3])).random() < 0.5 for g in ham}
    A = {g: d for g, d in ham.items() if kat[g]}
    B = {g: d for g, d in ham.items() if not kat[g]}
    bir = {t: np.ones(4) for t in TURLER}
    print(f"kat A {len(A)} / B {len(B)} döşeme; kalibrasyonsuz A {map50(A, bir):.3f}  B {map50(B, bir):.3f}  tümü {map50(ham, bir):.3f}", flush=True)
    WA, ma = ogren(A); print(f"A'da öğren: A {ma:.3f} → B'de {map50(B, WA):.3f}", flush=True)
    WB, mb = ogren(B); print(f"B'de öğren: B {mb:.3f} → A'da {map50(A, WB):.3f}", flush=True)
    W, m = ogren(ham); print(f"tümünde öğrenilen (uygulama için): {m:.3f}")
    for t in TURLER:
        print(t, "A:", WA[t].round(2), "B:", WB[t].round(2), "tümü:", W[t].round(2))
    Path(KOK := Path(__file__).resolve().parent.parent, "runs", "tur_kalibrasyon.json").write_text(
        json.dumps({t: W[t].tolist() for t in TURLER}, indent=1))
