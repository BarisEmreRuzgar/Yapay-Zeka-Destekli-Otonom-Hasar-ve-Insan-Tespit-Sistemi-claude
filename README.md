<div align="center">

# 🌍 DisasterVision AI: Yapay Zeka Destekli Otonom Hasar ve İnsan Tespit Sistemi

<p align="center">
  <img src="https://img.shields.io/badge/PYTHON-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-ULTRALYTICS-green?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/GPU-RTX_3050_6GB-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/STATUS-TAMAMLANDI-brightgreen?style=for-the-badge" />
</p>

Afet bölgelerinde havadan (İHA/drone ve uydu) çekilen görüntüler üzerinden bina hasar durumunu tespit eden ve insan varlığını algılayan bilgisayarla görü projesi.

</div>

---

## 📌 Proje Durumu

✅ **Tamamlandı.** Proje daha önce *"mevcut veri setlerinin çeşitlilik ve miktar olarak yetersiz kalması"* nedeniyle durdurulmuştu. Bu sürümde veri sorunu çözüldü:

* **İnsan sınıfı** için 8.600+ drone görüntüsü içeren **VisDrone2019-DET** eklendi.
* **Hasar sınıfları** için 850.000+ bina etiketli **xBD (xView2)** veri setinden ~5.400 döşeme eklendi.
* Orijinal veri hattındaki iki hata düzeltildi (aşağıda).
* Model yeniden eğitildi; eğitimde **hiç görülmemiş** iki bağımsız test setinde ölçüldü.
* Döşemeli çıkarım (SAHI) ve Gradio arayüzü eklendi.

<p align="center"><img src="docs/karsilastirma_drone_kalabalik.jpg" width="900"/><br/><em>VisDrone test-dev (eğitimde kullanılmadı): orijinal model 15, yeni model 79 kişi buluyor (gerçekte 91).</em></p>

<p align="center"><img src="docs/karsilastirma_uydu_hasar.jpg" width="900"/><br/><em>xBD test, Hurricane Harvey (eğitimde kullanılmadı): orijinal model 2 bina görüyor; yeni model binaları buluyor ve çoğunu doğru sınıflıyor.</em></p>

---

## 📊 Sonuçlar

Üç model aynı test setlerinde karşılaştırıldı (AP50, `conf=0.001`, `imgsz=640`):

| Test seti | Orijinal yöntem¹ | + VisDrone + xBD tier3 | **Son model** |
| :--- | :---: | :---: | :---: |
| **Gerçek drone görüntüsünde insan** (VisDrone test-dev, 3.057 döşeme) | 0.022 | 0.345 | **0.343** |
| **Farklı afetlerde bina hasarı** (xBD test, 337 döşeme), mAP50 | 0.054 | 0.130 | **0.294** |
| &nbsp;&nbsp;↳ yıkık / ağır / hafif / hasarsız | .014 / .004 / .005 / .192 | .048 / .050 / .057 / .363 | **.122 / .312 / .244 / .497** |
| Orijinal Roboflow test seti, mAP50 | 0.497 | 0.516 | **0.520** |

¹ Orijinal defterdeki düzen (YOLOv8s, yalnız Roboflow verisi) aynı donanımda yeniden eğitildi.

**Yorum:** Orijinal model yalnızca eğitildiği veriye benzeyen görüntülerde çalışıyordu (Roboflow testi 0.50). Gerçek drone görüntüsünde insanları neredeyse hiç bulamıyor (0.02), başka afetlerde de hasarı göremiyordu (0.05). Yeni model orijinal testte aynı seviyede kalırken bağımsız testlerde **5–15 kat** daha iyi.

### Orijinal veri hattında bulunan hatalar
1. **Kopya veri seti:** `damaged-buildings-vrcqx` (v8), `damaged-building-detection-zenbx` (v2) setinin birebir alt kümesi (aynı döşemeler, aynı bölümler). Defter ikisini birden ekliyor, aynı görüntüleri iki kez eğitiyordu. Benzersiz hasar verisi aslında yalnızca 531 döşemeydi.
2. **Açıkta API anahtarı:** Roboflow anahtarı herkese açık defterde düz metin olarak duruyordu. Artık `ROBOFLOW_API_KEY` ortam değişkeninden okunuyor.

### Afet öncesi görüntü varsa: iki aşamalı hasar tespiti

Yıkık sınıfı tek görüntülü modelin en zayıf noktasıydı (AP50 0.12). Teşhis, sorunun sınıflandırma değil
**konum** olduğunu gösterdi: yıkık binaların yalnız %15'i herhangi bir kutuyla eşleşiyordu, ama
eşleşenlerin %71'i doğru sınıflanıyordu. Bina yıkılınca sonrası görüntüde "bina" kalmıyor.

Uydu görüntüsünde aynı bölgenin afet öncesi hâli genelde vardır. Bu durumda:
1. **Konum:** bina, afet *öncesi* görüntüde tek sınıflı bir dedektörle bulunur (`modeller/bina_konum.pt`, xBD'nin tüm öncesi görüntülerinden 13,5k döşeme; doğrulama mAP50 0.79).
2. **Hasar:** her bina için öncesi | sonrası | fark | maske panellerinden oluşan bir görsel iki YOLOv8m-cls sınıflandırıcısına verilir, olasılıklar ortalanır (`modeller/hasar_cift.pt` + `hasar_cift_v4.pt`, ikincisi sıcaklık T=2 ile yumuşatılır). Paneller konuma bağlı olduğu için eğitimde yatay çevirme ve rastgele kırpma kapalı.
3. **Birleştirme:** tek görüntülü modelin sonrası görüntüdeki hasar kutuları 0.6 ağırlıkla eklenir, sınıf bazında NMS.

Bütün seçimler xBD eğitim dosyalarından ayrılmış bir doğrulama setinde yapıldı (ayrıntılı günlük: [`docs/ilerleme.md`](docs/ilerleme.md)); test setine yalnız sonda bir kez bakıldı.
xBD testinde (aynı 337 döşeme, aynı AP50 hesabı, `src/iki_asama.py`):

| | yıkık | ağır | hafif | hasarsız | **mAP50** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tek görüntü (son model) | 0.121 | 0.328 | 0.195 | 0.507 | 0.288 |
| İki aşama v1 (bina dedektörü + YOLOv8s-cls) | 0.446 | 0.388 | 0.168 | 0.616 | 0.404 |
| **İki aşama, son** (konum v2 + 2 sınıflandırıcı + birleştirme) | **0.546** | **0.490** | **0.303** | **0.700** | **0.510** |
| Gerçek kutular + son sınıflandırıcılar (tavan) | 0.880 | 0.743 | 0.450 | 0.957 | 0.758 |

Yıkık sınıfı **4.5 kat** (0.12 → 0.55), genel hasar mAP50'si **0.29 → 0.51**. Hafif hasar çatıdan zor görüldüğü için
hâlâ en zayıf sınıf. Tavanla aradaki fark, kalan darboğazın konum (bina bulma) olduğunu gösteriyor.

<p align="center"><img src="docs/iki_asama_ornek.jpg" width="600"/><br/><em>xBD test, Palu tsunamisi (eğitimde kullanılmadı), son sistemin sonrası görüntüsü üzerindeki çıktısı: zemin sıvılaşmasıyla binaların tamamen kaybolduğu bölge yıkık (kırmızı), sağlam mahalle hasarsız (yeşil) işaretleniyor.</em></p>

```bash
python src/cift_veri.py && python src/cift_veri.py --test   # öncesi/sonrası panel veri seti
python src/cift_egit.py                                     # hasar sınıflandırıcı → runs/cift
python src/bina_veri.py                                     # öncesi görüntüde bina veri seti
python src/egit.py --ad bina --veri veri/bina/data.yaml --model modeller/disastervision.pt --epoch 30
python src/iki_asama.py                                     # değerlendirme (modeller/bina_konum.pt + sınıflandırıcı topluluğu)
python src/iki_asama.py --once ornekler/palu_once.jpg --sonra ornekler/palu_sonra.jpg   # tek çift, çizimli çıktı
```

### Sınırlamalar
* Tek görüntülü hasar tespitinde sel ve hafif hasar çatıdan çoğu zaman görünmüyor; yıkık sınıfı zayıf. Afet öncesi görüntü varsa iki aşamalı yöntem (yukarıda) kullanılmalı; drone ile sahada çekilen görüntüde öncesi genelde olmadığı için insan tespiti ve drone senaryosu tek görüntülü modelle çalışır.
* Roboflow insan test seti, eğitimle aynı videonun karelerinden oluştuğu için iyimser sonuç veriyor (0.92). Gerçekçi ölçüt VisDrone testi.
* Sonuçlar karar desteği içindir; saha ekiplerince doğrulanmalıdır.

---

## 🎯 Projenin Amacı ve Kapsamı

| Özellik | Açıklama |
| :--- | :--- |
| 🏢 **Çok Sınıflı Hasar** | Binaların durumunu 4 kategoride sınıflandırma (`destroyed`, `major-damage`, `minor-damage`, `no-damage`). |
| 👤 **İnsan Tespiti** | Afet sahasındaki arama-kurtarma faaliyetlerine destek olmak amacıyla insan siluetlerinin tespiti (`person`). |
| 🔄 **Veri Mühendisliği** | Dört kaynağın ortak şemada birleştirilmesi; sınıf kimliklerinin **ada göre** eşlenmesi. |
| 🔍 **Döşemeli Çıkarım** | Büyük görüntülerde küçük insanları kaçırmamak için SAHI ile parçalı tahmin. |

---

## 🗂️ Veri Kaynakları

| Kaynak | Kullanım | Eğitimdeki pay | Lisans |
| :--- | :--- | :--- | :--- |
| [Roboflow: disaster](https://universe.roboflow.com/akash-gnbdg/disaster-dnaqz) | insan (eğik açılı drone) | 911 görüntü | Roboflow Universe |
| [Roboflow: damaged-building-detection](https://universe.roboflow.com/damaged-building-detection/damaged-building-detection-zenbx) | hasar (Hurricane Maria) | 1.119 görüntü | Roboflow Universe |
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | insan (`pedestrian` + `people`) | 4.000 döşeme (960 px) | yalnızca akademik |
| [xBD / xView2](https://xview2.org) ([HF aynası](https://huggingface.co/datasets/EVER-Z/torchange_xView2)) | hasar (19 afet) | 5.387 döşeme (512 px) | CC BY-NC-SA 4.0 |

* xBD maskelerinden her hasar seviyesinin bağlı bileşenleri bina kutusuna çevrilir. Yalnız hasarsız bina içeren döşemelerin %85'i atılarak sınıf dengesi iyileştirilir.
* **Sızıntı önlemi:** Roboflow hasar seti xBD'nin Hurricane Maria görüntülerinden türetildiği için Maria görüntüleri eğitime alınmaz. xBD testi, resmî `test` bölümünden gelir.
* Eğitim seti: 11.417 görüntü; 56.791 insan, 12.476 yıkık, 14.503 ağır, 16.401 hafif, 68.479 hasarsız bina.

> ⚠️ xBD ve VisDrone ticari olmayan kullanım içindir; bu verilerle eğitilen model de aynı kısıtı taşır.

---

## 📊 Sınıf Hiyerarşisi

| ID | Sınıf | Açıklama |
|:-:|---|---|
| 0 | `person` | İnsan |
| 1 | `destroyed` | Yıkık bina |
| 2 | `major-damage` | Ağır hasarlı |
| 3 | `minor-damage` | Hafif hasarlı |
| 4 | `no-damage` | Hasarsız |

---

## 🚀 Kurulum ve Kullanım

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Hazır modelle tahmin
Eğitilmiş ağırlıklar `modeller/disastervision.pt` dosyasında (YOLOv8s, 22 MB).

```bash
python app.py                                      # Gradio arayüzü → http://127.0.0.1:7860
python src/tahmin.py ornekler/drone_kalabalik.jpg  # komut satırı
python src/tahmin.py buyuk_goruntu.jpg --dilimle   # büyük görüntü: SAHI döşemeli çıkarım
```

```python
from ultralytics import YOLO
model = YOLO("modeller/disastervision.pt")
model.predict(source="fotograf.jpg", save=True, imgsz=640, conf=0.25)
```

### Modeli baştan üretmek
```bash
set ROBOFLOW_API_KEY=...                       # Linux/Mac: export ROBOFLOW_API_KEY=...
python src/veri_indir.py                       # Roboflow + VisDrone
# xBD: EVER-Z/torchange_xView2 deposundaki train-* ve tier3-* parquet dosyalarını veri/ham/xbd/ altına,
#      test-00000 dosyasını veri/ham/xbd_test/ altına indirin
python src/xbd_cevir.py --hasarsiz-oran 0.15   # xBD → YOLO döşemeleri
python src/xbd_cevir.py --test                 # bağımsız xBD test seti
python src/visdrone_dosele.py                  # VisDrone → 960 px döşemeler (+ test-dev)
python src/birlestir.py --visdrone-train 4000  # birleşik 5 sınıflı veri seti → veri/birlesik/

python src/egit.py --ad genis --veri veri/birlesik/data.yaml --epoch 60
python src/egit.py --ad genis2 --veri veri/birlesik/data.yaml --model runs/genis/weights/best.pt --epoch 40

python src/degerlendir.py runs/genis2/weights/best.pt --veri veri/xbd_test.yaml
python src/degerlendir.py runs/genis2/weights/best.pt --veri veri/visdrone_test.yaml
```
Eğitim RTX 3050 6 GB'ta yapıldı (batch 12, ~3 GB VRAM; toplam ~7 saat).

### Roboflow bulut dağıtımı
Model Roboflow'a `disastervision-yolov8s` adıyla yüklendi ve `disastervision-tespit` Workflow'u
üzerinden sunuluyor (tespit → kutu + etiket çizimi → insan sayısı). Anahtar `.env` içindeki
`ROBOFLOW_API_KEY` satırından okunur.

```bash
# Yükleme: Roboflow YOLOv8 paketlemesi ultralytics==8.0.196 istediği için ayrı ortam
py -3.11 -m venv .venv_yukle
.venv_yukle\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv_yukle\Scripts\pip install ultralytics==8.0.196 "roboflow>=1.3.13" python-dotenv
.venv_yukle\Scripts\python src/roboflow_yukle.py

python src/roboflow_test.py   # Workflow'u ornekler/ üzerinde çalıştırır, yerel modelle karşılaştırır
```

Workflow girdileri: `image`, `confidence` (varsayılan 0.25), `iou` (varsayılan 0.7). Roboflow'un
varsayılan NMS eşiği 0.3'tür; kalabalıkta yan yana duran insanları birleştirdiği için (örnek drone
görüntüsünde 79 → 58) eşik, yerel değerlendirmeyle aynı olacak şekilde 0.7'ye çekildi. Bu ayarla
örneklerde bulut/yerel sonuçlar: drone 76/79 insan, uydu görüntüsünde hasar sınıfları birebir aynı.

---

## 📁 Proje Yapısı

```text
├── app.py                  # Gradio arayüzü
├── modeller/
│   ├── disastervision.pt   # ana model: insan + hasar (tek görüntü)
│   ├── bina_konum.pt       # afet öncesi görüntüde bina konumu
│   └── hasar_cift.pt       # öncesi/sonrası çiftiyle hasar sınıflandırıcı
├── src/
│   ├── veri_indir.py       # Roboflow + VisDrone indirme
│   ├── xbd_cevir.py        # xBD parquet → YOLO döşemeleri
│   ├── visdrone_dosele.py  # VisDrone → insan döşemeleri
│   ├── birlestir.py        # 5 sınıflı birleşik veri seti
│   ├── egit.py             # eğitim
│   ├── degerlendir.py      # karşılaştırmalı değerlendirme
│   ├── tahmin.py           # tahmin + SAHI + çizim
│   ├── cift_veri.py        # öncesi/sonrası panel veri seti
│   ├── cift_egit.py        # hasar sınıflandırıcı eğitimi
│   ├── bina_veri.py        # öncesi görüntüde bina veri seti
│   ├── iki_asama.py        # iki aşamalı hasar tespiti + değerlendirme
│   ├── roboflow_yukle.py   # ağırlıkları Roboflow'a yükleme
│   └── roboflow_test.py    # Roboflow Workflow testi
├── ornekler/               # örnek görüntüler
├── docs/                   # karşılaştırma görselleri
└── Yapay-Zeka-...ipynb     # orijinal Colab defteri (tarihçe)
```

---

<div align="center">

**Geliştirici:** Barış Emre Rüzgar

</div>
