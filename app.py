"""DisasterVision Gradio arayüzü: havadan görüntü yükle → hasar ve insan tespiti.

Kullanım: python app.py   (tarayıcıda http://127.0.0.1:7860)
"""
import sys
from pathlib import Path

import cv2
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tahmin import VARSAYILAN_MODEL, ciz, ozet, tespit  # noqa: E402


def calistir(goruntu_rgb, conf, dilimle):
    if goruntu_rgb is None:
        return None, "Önce bir görüntü yükleyin."
    bgr = cv2.cvtColor(goruntu_rgb, cv2.COLOR_RGB2BGR)
    t = tespit(bgr, VARSAYILAN_MODEL, conf, dilimle)
    say = ozet(t)
    satirlar = [f"| {ad} | {n} |" for ad, n in say.items()]
    rapor = "| Sınıf | Adet |\n|---|---|\n" + "\n".join(satirlar)
    if say["insan"]:
        rapor = f"⚠️ **{say['insan']} olası insan tespit edildi.**\n\n" + rapor
    return cv2.cvtColor(ciz(bgr, t), cv2.COLOR_BGR2RGB), rapor


with gr.Blocks(title="DisasterVision") as demo:
    gr.Markdown("# 🌍 DisasterVision — Afet Hasar ve İnsan Tespiti\n"
                "İHA veya uydu görüntüsü yükleyin. Model binaları 4 hasar seviyesinde sınıflandırır ve insanları işaretler.\n\n"
                "*Karar destek aracıdır; sonuçlar saha ekiplerince doğrulanmalıdır.*")
    with gr.Row():
        with gr.Column():
            giris = gr.Image(label="Görüntü", type="numpy")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Güven eşiği")
            dilimle = gr.Checkbox(label="Döşemeli çıkarım (SAHI) — büyük görüntülerde küçük insanları yakalar", value=False)
            buton = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            cikti = gr.Image(label="Sonuç")
            rapor = gr.Markdown()
    buton.click(calistir, [giris, conf, dilimle], [cikti, rapor])
    ornekler = sorted((Path(__file__).parent / "ornekler").glob("*.jpg"))
    if ornekler:
        gr.Examples([[str(p)] for p in ornekler], inputs=[giris])

if __name__ == "__main__":
    demo.launch()
