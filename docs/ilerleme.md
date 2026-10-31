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
* Tüm tier3 (21/21) ile veri: konum 9.006 → 13.502 döşeme; sınıflandırıcı v3 verisi.
* Konumu 1024 px'te çalıştırma denemesi (küçük binalar).
