"""tek2d sınıflandırıcı verisi: xBD tek1 panelleri + drone panelleri (RescueNet train, CRASAR train), sabit bağlantılarla.

Drone panelleri xBD'ye göre az olduğu için --cogalt kez (farklı adla) bağlanır. Doğrulama: drone panelleri
(RescueNet val + CRASAR valid); xBD tek1 val ayrıca ölçülür (unutma kontrolü).

  python src/tek2d_veri.py --drone veri/tek_drone_rn_train veri/tek_drone_cr_train --cikti veri/tek2d
"""
import argparse
import os
import shutil
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent


def bagla(k: Path, h: Path):
    h.parent.mkdir(parents=True, exist_ok=True)
    if h.exists():
        return
    try:
        os.link(k, h)
    except OSError:
        shutil.copy2(k, h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd", default="veri/tek1")
    ap.add_argument("--drone", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", default=["veri/tek_drone_rn_val", "veri/tek_drone_cr_val"])
    ap.add_argument("--cikti", default="veri/tek2d")
    ap.add_argument("--cogalt", type=int, default=3)
    a = ap.parse_args()
    c = KOK / a.cikti
    say = {}
    for g in (KOK / a.xbd / "train").rglob("*.jpg"):
        bagla(g, c / "train" / g.parent.name / ("x_" + g.name))
        say["xbd"] = say.get("xbd", 0) + 1
    for d in a.drone:
        for g in (KOK / d).rglob("*.jpg"):
            for i in range(a.cogalt):
                bagla(g, c / "train" / g.parent.name / f"d{i}_{g.name}")
            say[d] = say.get(d, 0) + 1
    for d in a.val:
        for g in (KOK / d).rglob("*.jpg"):
            bagla(g, c / "val" / g.parent.name / f"{Path(d).name}_{g.name}")
            say["val"] = say.get("val", 0) + 1
    print(say, {s.name: len(list(s.iterdir())) for s in (c / "train").iterdir()})
