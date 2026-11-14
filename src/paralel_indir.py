"""Büyük dosyayı birden çok HTTP Range bağlantısıyla indirir (bağlantı başına hız sınırı olan sunucular için,
ör. Figshare ~270 KB/s). Yarıda kalırsa kaldığı yerden sürer (parça dosyaları .parcaN).

  python src/paralel_indir.py URL hedef.zip [--baglanti 8]
"""
import argparse
import sys
import threading
import time
import urllib.request
from pathlib import Path

BLOK = 1 << 20


def son_url_ve_boyut(url):
    istek = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "paralel-indir/1.0"})
    with urllib.request.urlopen(istek, timeout=60) as y:
        return y.geturl(), int(y.headers["Content-Length"])


def parca_indir(url, bas, son, yol: Path, durum, i):
    while True:
        mevcut = yol.stat().st_size if yol.exists() else 0
        if bas + mevcut > son:
            return
        try:
            istek = urllib.request.Request(url, headers={"Range": f"bytes={bas + mevcut}-{son}",
                                                         "User-Agent": "paralel-indir/1.0"})
            with urllib.request.urlopen(istek, timeout=120) as y, open(yol, "ab") as f:
                while True:
                    b = y.read(BLOK)
                    if not b:
                        break
                    f.write(b)
                    durum[i] += len(b)
        except Exception as e:  # ağ kesintisi: bekle ve kaldığı yerden sür (imzalı URL süresi dolarsa yenile)
            print(f"parça {i}: {e!r}, yeniden deneniyor", flush=True)
            time.sleep(15)
            try:
                url = son_url_ve_boyut(durum["kaynak"])[0]
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("hedef")
    ap.add_argument("--baglanti", type=int, default=8)
    a = ap.parse_args()
    hedef = Path(a.hedef)
    if hedef.exists():
        print("zaten var:", hedef); return
    url, boyut = son_url_ve_boyut(a.url)
    n = a.baglanti
    sinirlar = [(k * boyut // n, (k + 1) * boyut // n - 1) for k in range(n)]
    parcalar = [hedef.with_name(hedef.name + f".parca{k}") for k in range(n)]
    durum = {k: (p.stat().st_size if p.exists() else 0) for k, p in enumerate(parcalar)}
    durum["kaynak"] = a.url
    isler = [threading.Thread(target=parca_indir, args=(url, b, s, p, durum, k), daemon=True)
             for k, ((b, s), p) in enumerate(zip(sinirlar, parcalar))]
    for t in isler:
        t.start()
    onceki, t0 = sum(durum[k] for k in range(n)), time.time()
    while any(t.is_alive() for t in isler):
        time.sleep(60)
        simdi = sum(durum[k] for k in range(n))
        print(f"{time.strftime('%H:%M')} {simdi / 1e9:.2f}/{boyut / 1e9:.2f} GB  {(simdi - onceki) / 60 / 1e6:.2f} MB/s",
              flush=True)
        onceki = simdi
    for k, ((b, s), p) in enumerate(zip(sinirlar, parcalar)):
        if p.stat().st_size != s - b + 1:
            sys.exit(f"parça {k} eksik: {p.stat().st_size} / {s - b + 1}")
    with open(hedef.with_suffix(".tmp"), "wb") as f:
        for p in parcalar:
            with open(p, "rb") as g:
                while b := g.read(64 * BLOK):
                    f.write(b)
    hedef.with_suffix(".tmp").rename(hedef)
    for p in parcalar:
        p.unlink()
    print("tamam", hedef, f"{(time.time() - t0) / 60:.0f} dk")


if __name__ == "__main__":
    main()
