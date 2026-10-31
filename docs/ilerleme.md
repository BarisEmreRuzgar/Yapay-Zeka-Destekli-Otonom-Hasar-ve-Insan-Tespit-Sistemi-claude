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
* Paralel eğitimler (GPU 6 GB'a ikisi sığıyor):
  * Sınıflandırıcı v3 (`runs/cift3`): YOLOv8m-cls 256, ~83k örnek (yıkık 11.4k → 17.3k), panel-içi çevirme, 15 epoch.
  * Konum v2 (`runs/bina2`): bina_konum.pt'den devam, 13.5k döşeme, 640 px, 30 epoch.

## İnsan tespiti (projenin ana amacı)
VisDrone test döşemeleri 960 px ama eğitim ve değerlendirme `imgsz=640` ile yapılmış: döşeme ~1.5 kat
küçültülüyor, zaten birkaç piksellik insanlar daha da küçülüyor. Eğitimsiz ilk deneme: aynı modeli
640 / 960 / 1280'de değerlendirmek (`egitim_zinciri9`, sınıflandırıcı eğitimi bitince; RAM yetersiz olduğu için sırada).

Not: 09:00–09:50 arası iki iş başlatma komutundaki yol hatası yüzünden hiç başlamamış; fark edilip yeniden başlatıldı.
