"""DisasterVision Gradio arayüzü: havadan görüntü yükle → hasar ve insan tespiti.

Kullanım: python app.py   (tarayıcıda http://127.0.0.1:7860)
"""
import sys
from pathlib import Path

import cv2
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tahmin import INSAN_MODEL, SU_MODEL, TR, VARSAYILAN_MODEL, ciz, ozet, tespit  # noqa: E402


def calistir(goruntu_rgb, conf, dilimle, uydu=False, su=False):
    if goruntu_rgb is None:
        return None, "Önce bir görüntü yükleyin."
    bgr = cv2.cvtColor(goruntu_rgb, cv2.COLOR_RGB2BGR)
    t = tespit(bgr, VARSAYILAN_MODEL, conf, dilimle, insan_modeli=SU_MODEL if su else INSAN_MODEL)
    if uydu:  # uydu çözünürlüğünde (~0.5 m/piksel) insan görünmez; bulunan her "insan" yanlış alarmdır
        t = [x for x in t if x[0] != "person"]
    say = ozet(t)
    satirlar = [f"| {ad} | {n} |" for ad, n in say.items()]
    rapor = "| Sınıf | Adet |\n|---|---|\n" + "\n".join(satirlar)
    if say["insan"]:
        rapor = f"⚠️ **{say['insan']} olası insan tespit edildi.**\n\n" + rapor
    return cv2.cvtColor(ciz(bgr, t), cv2.COLOR_BGR2RGB), rapor


def cift_calistir(once_rgb, sonra_rgb, conf):
    if once_rgb is None or sonra_rgb is None:
        return None, "Afet öncesi ve sonrası görüntülerin ikisini de yükleyin."
    if once_rgb.shape != sonra_rgb.shape:
        sonra_rgb = cv2.resize(sonra_rgb, (once_rgb.shape[1], once_rgb.shape[0]))
    from iki_asama import cift_tespit
    t = cift_tespit(once_rgb, sonra_rgb, conf)
    bgr = cv2.cvtColor(sonra_rgb, cv2.COLOR_RGB2BGR)
    say = {TR[h]: sum(x[0] == h for x in t) for h in ("destroyed", "major-damage", "minor-damage", "no-damage")}
    rapor = "| Hasar | Bina |\n|---|---|\n" + "\n".join(f"| {ad} | {n} |" for ad, n in say.items())
    return cv2.cvtColor(ciz(bgr, t), cv2.COLOR_BGR2RGB), rapor


with gr.Blocks(title="DisasterVision") as demo:
    gr.Markdown("# 🌍 DisasterVision — Afet Hasar ve İnsan Tespiti\n"
                "İHA veya uydu görüntüsü yükleyin. Model binaları 4 hasar seviyesinde sınıflandırır ve insanları işaretler.\n\n"
                "*Karar destek aracıdır; sonuçlar saha ekiplerince doğrulanmalıdır.*")
    with gr.Tab("Tek görüntü (İHA / uydu)"), gr.Row():
        with gr.Column():
            giris = gr.Image(label="Görüntü", type="numpy")
            conf = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Güven eşiği (0.30: val F1 dengesi)")
            dilimle = gr.Checkbox(label="Döşemeli çıkarım (SAHI) — büyük görüntülerde küçük insanları yakalar", value=False)
            uydu = gr.Checkbox(label="Uydu görüntüsü — insan aramayı kapat (bu çözünürlükte insan görünmez)", value=False)
            su = gr.Checkbox(label="Su / sel sahnesi — sudaki insanlar için özel model", value=False)
            buton = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            cikti = gr.Image(label="Sonuç")
            rapor = gr.Markdown()
        buton.click(calistir, [giris, conf, dilimle, uydu, su], [cikti, rapor])
        ornekler = sorted(p for p in (Path(__file__).parent / "ornekler").glob("*.jpg") if "_once" not in p.stem)
        if ornekler:
            gr.Examples([[str(p), "uydu" in p.stem] for p in ornekler], inputs=[giris, uydu])
    with gr.Tab("Afet öncesi + sonrası (uydu)"):
        gr.Markdown("Aynı bölgenin hizalı **afet öncesi** ve **sonrası** uydu görüntüsü varsa: binalar öncesinde bulunur, "
                    "hasar iki zaman karşılaştırılarak sınıflanır. Tamamen yıkılmış (artık görünmeyen) binaları da yakalar.")
        with gr.Row():
            with gr.Column():
                once = gr.Image(label="Afet öncesi", type="numpy")
                sonra = gr.Image(label="Afet sonrası", type="numpy")
                conf2 = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Güven eşiği")
                buton2 = gr.Button("Hasar tespit et", variant="primary")
            with gr.Column():
                cikti2 = gr.Image(label="Sonuç (sonrası üzerinde)")
                rapor2 = gr.Markdown()
        buton2.click(cift_calistir, [once, sonra, conf2], [cikti2, rapor2])
        kok = Path(__file__).parent / "ornekler"
        cift_ornekler = [[str(o), str(o.with_name(o.name.replace("_once", "_sonra")))] for o in sorted(kok.glob("*_once.jpg"))]
        if cift_ornekler:
            gr.Examples(cift_ornekler, inputs=[once, sonra])

if __name__ == "__main__":
    demo.launch()
