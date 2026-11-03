"""Aynı soydan iki YOLOv8 ağırlığının doğrusal ortalaması (model çorbası / WiSE-FT): w = (1-a)·A + a·B.

Kullanım: python src/agirlik_ortala.py A.pt B.pt 0.5 cikti.pt
"""
import sys

import torch

if __name__ == "__main__":
    a_yol, b_yol, alfa, cikti = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    A = torch.load(a_yol, map_location="cpu", weights_only=False)
    B = torch.load(b_yol, map_location="cpu", weights_only=False)
    ma = (A.get("ema") or A["model"]).float()
    mb = (B.get("ema") or B["model"]).float()
    sb = mb.state_dict()
    yeni = {k: ((1 - alfa) * v + alfa * sb[k]) if v.dtype.is_floating_point else sb[k] for k, v in ma.state_dict().items()}
    mb.load_state_dict(yeni)
    B["model"] = mb.half(); B["ema"] = None; B["optimizer"] = None
    torch.save(B, cikti)
    print("kaydedildi:", cikti)
