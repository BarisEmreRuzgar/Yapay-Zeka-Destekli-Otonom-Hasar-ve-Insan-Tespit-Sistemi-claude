# Hasar tespiti iyileştirme günlüğü

Kural: tüm kararlar **doğrulama setinde** (xBD eğitim dosyalarının görüntü adına göre ayrılmış %10'u;
hiçbir model görmedi) verilir. Bağımsız xBD testine yalnız seçilen son sistem için bakılır.
Bu dosya 30 dakikada bir otomatik olarak commit edilip gönderilir (`otomatik_kayit.ps1`).

## Başlangıç (test, 337 döşeme)
| Sistem | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Tek görüntü (disastervision.pt) | 0.121 | 0.328 | 0.195 | 0.507 | 0.288 |
| İki aşama v1 (bina_konum + hasar_cift) | 0.446 | 0.388 | 0.168 | 0.616 | 0.404 |

## Doğrulama seti denemeleri (aynı 930 döşeme)
| # | Değişiklik | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | İki aşama v1 | 0.635 | 0.428 | 0.221 | 0.671 | 0.489 |
| 1 | + tek-model kutularını NMS ile ekle (ağırlık 0.6) | 0.639 | 0.442 | 0.254 | 0.710 | 0.511 |
| 2 | Sınıflandırıcı v2: YOLOv8m-cls, 256 px, maske = oynatılmış kutu, 10 epoch | 0.677 | 0.516 | 0.340 | 0.730 | 0.566 |
| 3 | 2 + birleştirme | 0.677 | 0.521 | 0.360 | 0.737 | **0.574** |

Skor birleşimi (det^a · olasılık^b) denendi: a=2 yalnız +0.006, şimdilik değiştirilmedi.

## Teşhisler
* Konum (sınıf-bağımsız AP50 0.79): 256 px²'den küçük binaların yalnız %58'i bulunuyor (orta boy %85–94).
* Sınıflandırıcı v1 eğitimde gerçek bina şeklini maske olarak görüyor, çıkarımda dikdörtgen alıyordu (uyumsuzluk) → v2'de düzeltildi.

## Sıradaki
* Tüm tier3 (21/21) indirildi. Konum verisi 9.006 → 13.502 döşeme.
* Sınıflandırıcı v3 verisi: tüm veri, sınıf başına ≤30k, **panel içi rastgele çevirme/döndürme** (8 kare simetrisi; panel düzeni korunur) → yön değişmezliği.
* TTA eklendi: `--tta N` (sınıflandırıcıda N panel-içi çevirmenin ortalaması), `--konum-tta` (ultralytics ölçek+çevirme).
* Konumu 1024 px'te çalıştırma (eğitim 640): val 1.348 döşemede iki aşama 0.564 → 0.563, konum AP 0.782 → 0.774.
  Küçük binalarda bulunma %58 → %61, diğer boylarda fark yok → saatler sürecek 1024 eğitimine değmez; vazgeçildi.
* ~~Paralel eğitimler~~ — **hata:** ikisi birlikte VRAM'i doldurdu (5.8/6 GB), Windows paylaşımlı belleğe taştı ve
  RAM 1 GB'a indi; adım süreleri 0.2 s → 1.3 s (sınıflandırıcı) ve 7 s (konum). 1 saat içinde neredeyse ilerleme yok.
  Konum eğitimi durduruldu (`--devam` ile sürecek), her şey tek sıralı kuyrukta (`egitim_zinciri10`):
  sınıflandırıcı v3 → VisDrone 640/960/1280 → konum v2 devam → val değerlendirmesi. Plan:
  * Sınıflandırıcı v3 (`runs/cift3`): YOLOv8m-cls 256, ~83k örnek (yıkık 11.4k → 17.3k), panel-içi çevirme, 15 epoch.
  * Konum v2 (`runs/bina2`): bina_konum.pt'den devam, 13.5k döşeme, 640 px, 30 epoch.

Sınıflandırıcı v3 eğitimi: val top1 epoch 2'den sonra ~0.76'da sabit, eğitim kaybı 0.05'e iniyor (ezber).

### Sınıflandırıcı v3 sonucu: daha fazla veri ≠ daha iyi (val, ortak 1.348 döşeme, iki aşama, konum v1)
| Sınıflandırıcı | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| v2 (10 epoch, en iyisi 4.) | 0.652 | 0.502 | 0.379 | 0.724 | 0.564 |
| v2, T=2 | 0.655 | 0.505 | 0.371 | 0.728 | 0.565 |
| v3 (15 epoch, tüm veri, panel çevirme) | 0.637 | 0.428 | 0.281 | 0.684 | 0.507 |
| v3, T=4 | 0.670 | 0.490 | 0.339 | 0.712 | 0.553 |
| **v2 + v3(T=4) olasılık ortalaması** | **0.677** | **0.521** | **0.384** | **0.727** | **0.577** |

Teşhis: v3 aşırı emin — gerçek kutularda olasılıkların %71'i > 0.99 (v2'de %27). AP kutuları olasılığa göre
sıraladığı için bu bilgi kaybı doğruluk aynıyken (top1 ~0.76) mAP'i düşürüyor. Sıcaklık ölçekleme (p^(1/T),
logit/T ile eşdeğer) kısmen düzeltiyor. `iki_asama.py --siniflandirici a.pt,b.pt --sicaklik 1,4` eklendi.
v4 = v3 verisi, 5 epoch, dropout 0.3: eğitim kaybı 0.22'de kaldı (v3: 0.05), ama tek başına v2'yi geçemedi.

| Sınıflandırıcı (val, ortak 1.348 döşeme, konum v1) | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| v2 | 0.652 | 0.502 | 0.379 | 0.724 | 0.564 |
| v4, T=2 | 0.662 | 0.501 | 0.332 | 0.714 | 0.552 |
| **v2 + v4(T=2)** | 0.673 | 0.524 | 0.389 | 0.727 | **0.578** |
| v2 + v3(T=4) + v4(T=2) | 0.681 | 0.528 | 0.387 | 0.727 | 0.581 |

Seçim: **v2 + v4(T=2)** (üçüncü model yalnız +0.003; çıkarım maliyetine değmez).
Sınıflandırıcı tavanı (gerçek kutular) 0.77, tahmin kutularıyla 0.58 → kalan boşluk **konumda**.

### Konum v2 ara durum
Yeni (büyük) bina val setinde v1 = **0.782** mAP50. v2 (v1'den yüksek LR ile devam, +%50 veri) epoch 1–9: 0.753–0.772,
yani şimdilik v1'in altında (yeniden ısınma). Kosinüs LR + son 10 epoch mozaiksiz ile yükselmesi bekleniyor; v1'i
geçemezse v1 kalır (karar iki aşama val sonucuna göre).
Sonuç: epoch 15'te 0.781, 20'de 0.790, **30'da 0.793** (v1: 0.782). İki aşama val değerlendirmesi çalışıyor.

## Arayüz
Gradio uygulamasına **"Afet öncesi + sonrası (uydu)"** sekmesi eklendi (`app.py`): iki hizalı görüntü yüklenir,
iki aşamalı sistem + tek-model birleşimi çalışır. Mantık `iki_asama.cift_tespit()` içinde (modeller önbellekte).

## İnsan tespiti (projenin ana amacı)
VisDrone test döşemeleri 960 px ama eğitim ve değerlendirme `imgsz=640` ile yapılmış: döşeme ~1.5 kat
küçültülüyor, zaten birkaç piksellik insanlar daha da küçülüyor. Eğitimsiz ilk deneme: aynı modeli
640 / 960 / 1280'de değerlendirmek.

| `imgsz` (VisDrone test-dev, 3.057 döşeme) | insan AP50 | AP50-95 |
| :---: | :---: | :---: |
| 640 (şimdiki) | 0.343 | 0.130 |
| **960** | **0.380** | 0.145 |
| **1280** | **0.407** | **0.155** |

Ama aynı şey uydu hasarına zarar veriyor (xBD val 544 döşeme, 512 px, hasar mAP50): 640 → **0.410**, 960 → 0.389, 1280 → 0.336.
Ortak kural: model görüntünün **~1.33 katında** en iyi (eğitimdeki ölçek: VisDrone 960→640, xBD 512→640).
`tahmin.py` artık sabit 640 yerine `cikarim_boyutu(uzun_kenar) = clip(1.33·uzun_kenar, 640, 1280)` kullanıyor
(SAHI dilimleri için de). Örnek drone görüntüsünde bulunan insan 79 → 103 (gerçek ~91).
Not: Roboflow'daki model sabit 640 girdiyle çalışıyor; bu kazanç orada yok.

Sırada (`egitim_zinciri12`, kuyruk 11 bitince): **insan uzmanı** — yalnız insan sınıfı, 1280 px'te eğitim
(VisDrone 18k döşemeden rastgele 8.000; disastervision.pt'den başlangıç, 20 epoch). Hasar 640'taki ana modelde kalır.
Karar VisDrone **val** üzerinde: ana model @1280 ile karşılaştırılacak.

Not: 09:00–09:50 arası iki iş başlatma komutundaki yol hatası yüzünden hiç başlamamış; fark edilip yeniden başlatıldı.
