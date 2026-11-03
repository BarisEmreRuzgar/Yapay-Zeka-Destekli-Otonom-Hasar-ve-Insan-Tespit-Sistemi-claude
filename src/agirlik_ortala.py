"""Aynı soydan YOLOv8 ağırlıklarının ağırlıklı ortalaması (model çorbası / WiSE-FT).

Kullanım: python src/agirlik_ortala.py cikti.pt A.pt:0.3 B.pt:0.7 [C.pt:w ...]   (ağırlıklar 1'e normalize edilir)
Son bileşenin kontrol noktası (args vb.) çıktıya taşınır.
"""
import sys

import torch


def model(yol):
    ck = torch.load(yol, map_location="cpu", weights_only=False)
    return ck, (ck.get("ema") or ck["model"]).float()


if __name__ == "__main__":
    cikti, parcalar = sys.argv[1], [p.rsplit(":", 1) for p in sys.argv[2:]]
    toplam = sum(float(w) for _, w in parcalar)
    ortalama, son = None, None
    for yol, w in parcalar:
        ck, m = model(yol)
        w = float(w) / toplam
        sd = m.state_dict()
        if ortalama is None:
            ortalama = {k: v * w if v.dtype.is_floating_point else v for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    ortalama[k] += v * w
        son = (ck, m)
    ck, m = son
    m.load_state_dict(ortalama)
    ck["model"] = m.half(); ck["ema"] = None; ck["optimizer"] = None
    torch.save(ck, cikti)
    print("kaydedildi:", cikti, [(y, w) for y, w in parcalar])
