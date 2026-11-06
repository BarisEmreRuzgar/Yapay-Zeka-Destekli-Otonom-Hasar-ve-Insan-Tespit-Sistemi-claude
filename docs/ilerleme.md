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
Kıyas: ana model VisDrone val @1280 insan AP50 = **0.607** (val, test-dev'den belirgin şekilde kolay; test @1280 = 0.407).

### Hasar: son yapılandırma
Konum v2 (`runs/bina2`) + sınıflandırıcı topluluğu v2 + v4(T=2) + tek-model birleşimi (`iki+tek`).
Konum v2 tek başına (cift2 ile, ortak 1.348 val döşeme): 0.564 → 0.571.

Tam val (1.446 döşeme):

| Sistem | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| tek görüntü | 0.329 | 0.396 | 0.300 | 0.569 | 0.398 |
| konum v2 + cift2, iki+tek | — | — | — | — | 0.578 |
| **konum v2 + (v2 + v4 T=2), iki+tek** | 0.691 | 0.525 | 0.403 | 0.731 | **0.588** |
| aynı, gerçek kutularla (tavan) | 0.884 | 0.710 | 0.506 | 0.984 | 0.771 |

**Tek seferlik bağımsız test (337 döşeme):**

| Sistem | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| tek görüntü | 0.121 | 0.328 | 0.195 | 0.507 | 0.288 |
| iki aşama v1 | 0.446 | 0.388 | 0.168 | 0.616 | 0.404 |
| **son sistem (iki+tek)** | **0.546** | **0.490** | **0.303** | **0.700** | **0.510** |
| gerçek kutularla tavan | 0.880 | 0.743 | 0.450 | 0.957 | 0.758 |

Val'deki kazanç (0.489 → 0.588) testte de korundu (0.404 → 0.510). Modeller `modeller/`'e kopyalandı
(bina_konum.pt = v2, hasar_cift.pt = v2, hasar_cift_v4.pt), `iki_asama` ve uygulama varsayılanı bu topluluk.

Not: 09:00–09:50 arası iki iş başlatma komutundaki yol hatası yüzünden hiç başlamamış; fark edilip yeniden başlatıldı.

## İnsan tespiti: arama-kurtarma (SAR) drone verisi
Roboflow Universe'ten indirildi (`src/sar_indir.py`, anahtar `.env`):
* **drone-leak/search-and-rescue-8ywn7** (CC BY 4.0): 12 videodan 2.429 adet 3840×2160 kare, nehir kenarı / tarla / 50–150 ft
  yükseklik / 45° eğik çekim, insanlar ~70 px. Roboflow'un bölmesi rastgele kare bazında (ardışık neredeyse aynı kareler
  train ile valid'e dağılmış → sızıntı); bu yüzden **video bazında** ayrıldı: 45DEG_150FT, BIV_RIVER_100FT, NEGATIVE-_100FT
  yalnız doğrulamada. 960 px döşemeler (`src/sar_veri.py`): eğitim 3.493 döşeme / 3.340 insan, doğrulama 1.584 / 1.362.
* ilinsk-univerzita-v-iline/search-and-rescue: 5 alçak irtifa karlı video, 640'a gerdirilmiş + 3× artırılmış, büyük insanlar.
  VisDrone/afet senaryosundan uzak → şimdilik kullanılmadı.
* trash-32e51/search-rescue (yatan/oturan/sürünen insan): yerden 1280×720 çekim, insanlar ~180×500 px, havadan değil → kullanılmadı.

Plan (`egitim_zinciri14`): ana modelin SAR doğrulamada sıfır-atış başarısı (1280) → VisDrone 8k + SAR eğitim ile
1280 px insan uzmanı (12 epoch) → VisDrone val (kıyas 0.607) ve SAR val üzerinde karşılaştırma.

Ana model SAR doğrulamada (sıfır-atış, 1280): insan AP50 **0.260**, AP50-95 0.113 — VisDrone val (0.607) ile kıyasla belirgin alan farkı.

* **SARD** (sandr/drone-aerial, CC BY 4.0): moloz/kaya/tarlada kısmen gizli insanlar, tek uzun videonun 1.981 karesi
  (gss1–2104), Roboflow 640×640'a germiş → 16:9'a geri açıldı (`src/sard_veri.py`). Ayrım kare numarasıyla:
  gss ≥ 1800 doğrulama (301 görüntü / 1.118 insan), 1750–1799 tampon atıldı; eğitimden rastgele 2.000 görüntü.
* **HERIDAL** (dağlık arazi SAR, Hırvatistan) iniyor (4,2 GB) → hiç eğitimde kullanılmayacak, **harici test** olacak.

Eğitim yeniden başlatıldı (`egitim_zinciri15`): VisDrone 8k + drone-leak 3,5k + SARD 2k, 1280 px, 10 epoch.
Değerlendirme: VisDrone val (kıyas 0.607), drone-leak SAR val (kıyas 0.260), SARD val (kıyas: ana model sıfır-atış AP50 **0.315**, AP50-95 0.141).

### İnsan uzmanı sonucu (`runs/insan_sar` → `modeller/insan_uzman.pt`)
Eğitim içi VisDrone val epoch 1→10: 0.575 → 0.663 (yükselme son epoch'a kadar sürdü; daha uzun eğitim fayda verebilir).

| İnsan AP50 (1280) | Ana model | İnsan uzmanı |
| :--- | :---: | :---: |
| VisDrone val | 0.607 | **0.666** |
| drone-leak SAR val (3 görülmemiş video) | 0.260 | **0.860** |
| SARD val (gss ≥ 1800) | 0.315 | **0.840** |
| VisDrone test-dev (tek seferlik) | 0.407 | **0.458** |

`tahmin.py`: insanlar uzmandan (her zaman 1280; SAHI'de 960 dilim), hasar ana modelden. Kalabalık örnek: 103 → 86 (gerçek ~91).
Uydu örneğinde uzman 1 yanlış insan buldu (ana model 0) — uydu çözünürlüğünde insan görünmez; ileride uydu girdisinde insan kapatılabilir.
HERIDAL ilk indirmede bağlantı koptu (2,6/4,3 GB), yeniden indiriliyor → harici test.

### Sıradaki: uzman v2 + HERIDAL harici testi
* `egitim_zinciri17`: uzman v1'den aynı veriyle 12 epoch daha (`runs/insan_sar2`); karar val setlerinde (v1: 0.666 / 0.860 / 0.840).
* HERIDAL (new-workspace-qsuag/heridal-t428z, CC BY 4.0): 13 bölgeden 1.546 adet 4000×3000 dağ/orman karesi →
  `src/heridal_veri.py` ile 5.808 döşeme / 4.555 insan. **Hiç eğitimde yok**; ana model, uzman v1 ve v2 karşılaştırılacak (`egitim_zinciri18`).
* Uzman v2 (devam eğitimi) VisDrone val: epoch 1–7 = 0.625, 0.632, 0.615, 0.606, 0.620, 0.622, 0.654 (v1 0.666); yeniden ısınma düşüşü sonrası toparlanıyor.
* Kuyrukta (`egitim_zinciri19`, zincir18 bitince): **YOLOv8m** insan uzmanı (COCO başlangıç, 1280, batch 2, 10 epoch, aynı veri) →
  model kapasitesinin küçük insanlarda fark yaratıp yaratmadığı; aynı val setleri + HERIDAL.

### Uzman v2 sonucu (`runs/insan_sar2`)
Devam eğitimi epoch 8–12: 0.668, 0.673, 0.678, 0.683, 0.686 — yeniden ısınma düşüşünden (0.606) sonra v1'i geçti.

| İnsan AP50 (1280, val) | v1 | **v2** |
| :--- | :---: | :---: |
| VisDrone val | 0.666 | **0.690** |
| drone-leak SAR val | 0.860 | **0.890** |
| SARD val | 0.840 | **0.864** |

Üçünde de daha iyi → v2 seçildi, `modeller/insan_uzman.pt` ve Roboflow `disastervision-insan` güncellendi.

**HERIDAL harici test** (hiç eğitimde yok, 5.808 döşeme / 4.555 insan, 1280): ana model **0.253**, uzman v1 **0.621**, uzman v2 **0.628**
(AP50-95: 0.129 / 0.326 / 0.328). VisDrone test-dev v2 ölçümü YOLOv8m eğitiminden sonra (`egitim_zinciri20`).

### YOLOv8m denemesi: durduruldu
COCO ağırlıklarından, 1280, batch 2: VisDrone val epoch 1–3 = 0.313, 0.310, 0.364 (epoch başına ~46 dk). Eğilimle 10 epoch'ta
~0.6'ya ulaşabilir, uzman v2'nin (0.690) altında → 3. epoch'ta durduruldu. Daha büyük model bu VRAM'de (batch 2) ve bu sürede kazanç sağlamıyor.

### Uzman v3 (`egitim_zinciri21`)
v2'den başlayarak **VisDrone'un tüm 18.098 eğitim döşemesi** (önce 8.000) + drone-leak SAR + SARD 2k = 23.591 görüntü, 1280, 8 epoch (~5 saat).
Karar val setleri + HERIDAL'de v2'ye göre.

### Güven eşiği (uzman v2, val setlerinde F1 taraması, `src/esik_bul.py`, CPU)
| conf | VisDrone F1 | SAR F1 | SARD F1 |
| :---: | :---: | :---: | :---: |
| 0.10 | 0.577 | 0.732 | 0.685 |
| 0.20 | 0.665 | 0.804 | 0.781 |
| 0.25 (eski varsayılan) | 0.679 | 0.823 | 0.813 |
| **0.30 (yeni varsayılan)** | 0.679 | 0.838 | 0.833 |
| en iyi | 0.680 @0.28 | 0.849 @0.44 | 0.846 @0.41 |

0.30 üç sette de en iyiye 0.013'ten yakın; duyarlılık tarafında kalındı (arama-kurtarmada kaçırma pahalı).
`tahmin.py`, uygulama ve Roboflow workflow varsayılanı 0.30.
* v3 epoch 1–3: 0.660, 0.659, 0.657. Epoch 3'te adım süresi 0.33 → 2.7 s'ye çıktı: tam VisDrone'daki çok kalabalık döşemeler
  (yüzlerce insan) kutu atama belleğini şişirdi, VRAM 6.47 GB'a çıkıp paylaşımlı belleğe taştı. Epoch sonunda durdurulup
  `--devam` ile sürdürüldü (`egitim_zinciri22`) → 3.67 GB, 3.0 it/s. Ders: veri büyüyünce VRAM'i epoch içinde de izle.
* v3 epoch 4–5: 0.668, 0.678. Epoch 6 ortasında yine yavaşlama (2.4 s/it) → durdurulup `--devam` +
  `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:128` ile sürdürüldü.

## Oturum devri (2026-11-02)
**Uzman v3 sonucu: v2 korunur.** v3 epoch 6'da CUDA OOM ile çöktü (`max_split_size_mb` ayarı taşmayı yavaşlama yerine
hataya çevirdi — bu ayarı kullanma). Epoch 5 en iyisiyle (VisDrone val 0.678) değerlendirme: SAR **0.864** (v2 0.890),
SARD **0.843** (v2 0.864), HERIDAL **0.600** (v2 0.628). VisDrone'u 8k → 18k büyütmek SAR verisinin payını düşürüp
arama-kurtarma başarısını azalttı → bu yön bırakıldı. Etkin model: `modeller/insan_uzman.pt` = v2 (yerel + Roboflow `disastervision-insan-v2`).

Arka planda çalışan iş yok (yalnız `otomatik_kayit.ps1`). Sıradaki adaylar:
1. Uzman v2'nin **tek seferlik** VisDrone test-dev ölçümü (`degerlendir.py runs\insan_sar2\weightsest.pt --veri veri\insan.yaml --bolum test --imgsz 1280`) → README tablosundaki "ölçülecek".
2. Fikir: SAR payını koruyarak veri büyütme (SAR'ı çoğaltma/oversample) veya hafif hasar sınıfı (en zayıf, test 0.30).

## 2026-11-02 öğleden sonra
* `egitim_zinciri23`: uzman v2'nin tek seferlik VisDrone test-dev ölçümü (çalışıyor).
* `egitim_zinciri24` (23 bitince): **YOLOv8m bina konum modeli** (`runs/bina_m`, COCO başlangıç, 640, batch 8, 30 epoch).
  Gerekçe: hasar sisteminde tahmin kutularıyla val 0.588, gerçek kutularla 0.771 → boşluk konumda; küçük binaların
  (<256 px²) yalnız %58'i bulunuyor. Kıyas: konum v2 (YOLOv8s) bina val mAP50 **0.793**; karar iki aşama val (iki+tek 0.588) ile.
* **Uzman v2 VisDrone test-dev (tek seferlik): AP50 0.469**, AP50-95 0.180 (v1 0.458, ana model @1280 0.407). README güncellendi.
* Yumuşak füzyon denemesi (`src/birlesim_dene.py`, val_son.pkl, CPU): konum kutusunun sınıf olasılığını örtüşen tek-model
  skoruyla harmanlama + sınıf başına tek-model ağırlığı. En iyi sınıf başına seçimle val mAP50 0.588 → 0.594 (hafif 0.403 → 0.412).
  8 ayarla +0.006 aşırı uyum payı içinde → **benimsenmedi**; mevcut NMS birleşimi kalır.
* `bina_m` eğitimi başladı: 1.688 adım/epoch, 2.6 it/s (~11 dk/epoch), VRAM 3.4 GB.
* Uygulama: tek görüntü sekmesine **"Uydu görüntüsü — insan aramayı kapat"** seçeneği (uydu örneğinde 1 yanlış insan → 0;
  örneklerde uydu görüntüsü seçilince otomatik işaretli).
* Sudaki insanlar verisi (sigmoidal-ivbj6/search-and-rescue-y6x1c, 8.930 görüntü, kamu malı, `swimmer` sınıfı) iniyor →
  sel senaryosu için uzman v2 sıfır-atış ölçümü.

### Sudaki insanlar (sel senaryosu) — harici veri
`src/su_veri.py`: sigmoidal-ivbj6/search-and-rescue-y6x1c (kamu malı, 2728×1816 deniz/göl kareleri; SeaDronesSee benzeri).
Roboflow her kareyi 4 artırılmış kopyayla vermiş → her kareden **tek kopya**; bölümler arasında ortak kare yok.
Yalnız `swimmer` → insan; 960 px döşemeler: test 893 kare → 830 döşeme, valid 1.786 kare → 1.686 döşeme / 2.729 insan.
İnsanlar çok küçük: kutu genişliği medyan 34 px (p10 15, p90 85). Uzman hiç su görmedi → sıfır-atış ölçümü (CPU, eğitim GPU'da).
* bina_m (YOLOv8m konum) epoch 1–5: 0.625, 0.660, 0.664, 0.701, 0.716 (kıyas: konum v2 0.793).
* **Sıfır-atış (Roboflow'un rastgele test bölmesi):** uzman v2 AP50 **0.147** (AP50-95 0.051), ana model 0.075 → sudaki
  insanlar büyük bir boşluk.
* **Sızıntı bulundu:** Roboflow bölmesi kare bazlı rastgele; test karelerinin %84'ünün ±1 komşusu eğitimde. `su_veri.py`
  bölümleri birleştirip kare numarası bloklarıyla yeniden ayırıyor (valid 3000–4000 ve 11000–12000, test 7000–8000 ve
  14000–15000, kenarlarda 30 kare tampon): eğitim 6.902 döşeme / 11.840 insan, valid 807 / 494, test 642 / 1.646.
* `egitim_zinciri25` (24 bitince): **uzman v4** = v2'den, v2 verisi (13.493) + su 4.300 döşeme (insanlı 4.000 + boş 300;
  SAR payını korumak için tümü değil), 1280, 8 epoch → VisDrone/SAR/SARD/su val + HERIDAL. Su testi yalnız sonda.
* bina_m epoch 6–13: 0.722, 0.738, 0.745, 0.743, 0.750, 0.748, 0.757, 0.760 (yavaş ama düzenli yükseliş).
* bina_m epoch 14–24: 0.767 → 0.791 (mozaiksiz son 10 epoch başladı; konum v2 0.793 seviyesine ulaştı).
* bina_m eğitimi bitti: bina val mAP50 epoch 30 = **0.793** (konum v2 YOLOv8s ile aynı). İki aşama val değerlendirmesi çalışıyor.
* Hazır: `src/konum_topluluk.py` — iki konum modelinin kayıtlı val tahminlerini WBF ile birleştirir (CPU, 1.446 döşeme ~10 s).
  Aynı mAP'li iki farklı mimari farklı binaları kaçırıyorsa topluluk konum boşluğunu (0.588 ↔ 0.771) daraltabilir.

### Hasar: konum hata analizi (val_son.pkl, CPU)
Afet bazında konum AP50 (sınıf-bağımsız): moore/joplin/florence ~0.91–0.93, ama **sunda-tsunami 0.34, mexico-earthquake 0.43,
nepal-flooding 0.60**, palu 0.71. Bunların iki+tek mAP'leri de en düşükler.
* Boyut değil: Meksika medyan bina kenarı 36 px (Moore 37.5 px, orada bulunma %91).
* Öncesi/sonrası kayması değil: eşleşen kutularda ortalama merkez farkı < 1 px.
* Maskeden kutu çıkarımında bitişik binaların birleşmesi (bağlı bileşen) küçük bir etken: kaçırılanların yalnız %2.4'ü.
  Görsel incelemede Meksika'da **etiketsiz binalar** ve çatıdan küçük GT kutuları da görüldü (etiket kalitesi).
* **Asıl neden:** bu afetler eğitimde en az temsil edilenler (sunda 234, mexico 417 döşeme; portugal 1.939) ve bina yoğunlukları
  en yüksek; kaçırılan binaların ~%32'si 0.05–0.25 güvenle zaten bulunuyor → model bu alanlarda az emin.
* Çare: **zayıf afetleri çoğaltarak ince ayar** — `veri/bina_os` (sabit bağlantı; sunda ×4, mexico/palu ×3, nepal/matthew ×2;
  17.715 döşeme), AdamW lr0 0.0005, 10 epoch (`egit.py --lr0` eklendi). Karar iki aşama val'de, afet bazında da bakılacak.
* `degerlendir.py --tta` eklendi; `egitim_zinciri27`: uzman v2'de TTA (SAR/SARD/VisDrone/su val).

### YOLOv8m konum sonucu (val, 1.446 döşeme, sınıflandırıcı v2+v4 T=2, iki+tek)
| Konum | konum AP50 | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| v2 (YOLOv8s) | 0.788 | 0.691 | 0.525 | 0.403 | 0.731 | 0.588 |
| bina_m (YOLOv8m, COCO'dan 30 epoch) | 0.786 | 0.686 | 0.532 | 0.402 | 0.730 | 0.587 |
| WBF (iki model) | 0.776–0.785 | | | | | 0.540–0.557 |
| **eşleşmeli topluluk** (NMS + iki modelin eşleşen skor ortalaması) | **0.801** | 0.693 | 0.534 | 0.406 | 0.737 | **0.592** |

Büyük model tek başına kazanç getirmedi (kapasite değil veri/alan sorunu — hata analiziyle uyumlu). Topluluk +0.004 ama
çıkarımı ~3 kat ağırlaştırıyor → şimdilik benimsenmedi (`konum_topluluk.eslesik`, ileride son birleştirmede düşünülebilir).
Sıradaki: `egitim_zinciri28` — konum v2'den çoğaltmalı ince ayar (`runs/bina_os`), 27 bitince.
* Uzman v4 (`runs/insan_su`) eğitimi başladı: 4.449 adım/epoch, 3.0 it/s (~25 dk/epoch), VRAM 3.6 GB.

### Sınıflandırıcı hata analizi + afet türü kalibrasyonu (CPU, val_son.pkl)
Gerçek kutularda sınıf doğruluğu: yıkık 0.83, ağır 0.71, hafif 0.60, hasarsız 0.88. Hatalar kasırgalarda yoğun:
Matthew'da hasarsızların %94'ü "hafif", Harvey'de yıkık ve hafif yalnız %9 doğru (sel hasarı çatıdan görünmüyor;
xBD'de etiketleme alışkanlığı afet türüne göre değişiyor). Meksika/Sunda'da sınıflandırma iyi (0.92–0.98) — orada sorun yalnız konum.
**Deneme:** afet türü (6 tür) başına sınıf çarpanı (`src/tur_kalibrasyon.py`), val görüntü adına göre iki kata bölünüp çapraz:
A'da öğren → B: 0.574 → **0.575**; B'de öğren → A: 0.590 → **0.584**. Kat içi kazanç (+0.007/+0.010) aktarılmıyor → aşırı uyum,
**reddedildi**. (Not: val seçimi `Random(ad) < 0.1` olduğu için kat tohumu farklı olmalı — ilk denemede herkes A'ya düşmüştü.)
* Uzman v4 epoch 1–2 VisDrone val: 0.662, 0.622 (v2 devam eğitimindeki yeniden ısınma eğrisine benzer).
* `egitim_zinciri29` (28'den sonra): hasar iki aşamada hiç ölçülmemiş çıkarım artırmaları — konum TTA (`--konum-tta`) ve sınıflandırıcı panel TTA (`--tta 4`), val.

### HERIDAL bölge ayrımı (uzman v5 hazırlığı)
HERIDAL dosya adlarında 13 bölge kodu var. İnsan tarafındaki en düşük metrik HERIDAL (0.628) olduğu için dağ/orman alanını
eğitime katmak üzere bölgeye göre ayrıldı (`src/heridal_bolge.py`): eğitim ZRI, VRD, JAS, BRA, BRS, SB → 3.178 döşeme / 2.928 insan;
**hiç görülmeyen bölgeler** GOR, MED, BRK, TRS, BLA, CAP, RAK → 2.630 döşeme / 1.627 insan (`veri/heridal_bolge_test.yaml`).
Kıyas için v2 ve v4 bu test bölgelerinde ayrıca ölçülecek (eski tam HERIDAL sayıları bununla doğrudan kıyaslanamaz).

### Uzman v4 (su verisiyle) sonucu: reddedildi
Eğitim içi VisDrone val epoch 1–8: 0.662, 0.622, 0.613, 0.652, 0.652, 0.667, 0.675, 0.680.

| İnsan AP50 (val, 1280) | v2 | v4 |
| :--- | :---: | :---: |
| VisDrone val | **0.690** | 0.666 |
| drone-leak SAR val | **0.890** | 0.768 |
| SARD val | **0.864** | 0.762 |
| HERIDAL (tam, harici) | **0.628** | 0.597 |
| su val (blok ayrımı) | (zincir 26) | 0.558 |

Su verisi su sahnesinde kazanç sağladı ama SAR'da −0.12, SARD'da −0.10. Olası neden: veri setinde yalnız `swimmer`
etiketli, tekne/kıyıdaki insanlar etiketsiz → model "bazı bağlamlarda insanı görmezden gel" öğreniyor; nehir kıyısı SAR
sahneleri tam da bu bağlam. **Ders: kısmi etiketli (yalnız bir alt tür insan) veri genel insan uzmanını bozar.** v2 etkin kalır.
* `egitim_zinciri30` (29'dan sonra): v2 ve v4'ü HERIDAL görülmemiş bölgelerde ölç → **uzman v5** = v2'den, v2 verisi + HERIDAL
  eğitim bölgeleri (16.671 görüntü), 1280, 8 epoch → görülmemiş HERIDAL bölgeleri + VisDrone/SAR/SARD val.
* v2 su val (blok ayrımı, eski tek-kopya veri): **0.188** (v4 0.558) → v4 su sahnelerinde 3 kat iyi, karada kötü. Çözüm:
  v4 ayrı **su / sel modu** olarak sunuldu (`modeller/insan_su.pt`; `tahmin.py --su`, uygulamada "Su / sel sahnesi" kutusu).
  Normal mod v2 olarak kalır.
* **İnsan TTA reddedildi** (uzman v2, ultralytics augment): SAR 0.890 → 0.849, SARD 0.864 → 0.861, VisDrone 0.690 → 0.670,
  su 0.188 → 0.217. Ultralytics TTA ölçekleri 1 / 0.83 / 0.67 — küçültme zaten birkaç piksellik insanları yok ediyor.
* **Su verisi hatası düzeltildi:** aynı kare numarası ~4 *farklı* görüntüde geçiyor (farklı kaynak dizileri; parlaklıkları
  165/169/108/111, içerikleri farklı — biri yüzücüler, biri şamandıra). "Her numaradan tek kopya" verinin %75'ini atmıştı.
  `su_veri.py` artık hepsini kullanıyor (`veri/su_yolo2`; aynı numaralılar aynı bölümde, blok ayrımı korunuyor).
  Siyah dikdörtgenler veri setinin kendi yok sayılan bölgeleri (artırma değil; README: artırma yok).
* `su_yolo2`: test 2.660 / valid 3.304 döşeme hazır (eski tek-kopya: 642 / 807). Eğitim bölümünün derlemesi bellek baskısıyla
  yarıda kaldı → GPU kuyruğu boşalınca `egitim_zinciri31` yeniden derliyor, ardından **su modu modeli v2** (`runs/insan_su2`):
  v2'den, tüm su eğitim verisi + drone-leak SAR eğitim döşemeleri (nehir kıyısı insanlar tam etiketli; kısmi etiket etkisini
  dengelemek için), 1280, 8 epoch → su valid2'de v2 / v4 / su2 kıyası + SAR val.

### Çoğaltmalı konum ince ayarı (`runs/bina_os`) sonucu: değişiklik yok
Bina val mAP50 epoch 1–10: 0.773 → 0.794 (v2 0.793). İki aşama val: **0.589** (v2 0.588), konum AP 0.789 (0.788).
Zayıf afetlerde konum: sunda 0.344 → 0.356, mexico 0.432 → 0.438, nepal 0.604 → 0.608, palu 0.709 → 0.704.
Bu afetlerin 3–4 kat çoğaltılması bile konumu kıpırdatmadı → sorun veri miktarı değil; görsel incelemede görülen
**etiketsiz binalar / çatıdan küçük GT kutuları** (etiket kalitesi) en olası açıklama — veri setinin kendi tavanı.
Karar: konum v2 korunur. Konum tarafında denenenler (YOLOv8m, 1024 çıkarım, WBF, eşleşmeli topluluk, çoğaltma) en fazla +0.004.

### Hasar TTA (zincir 29, val, konum v2 + v2/v4 T=2, iki+tek)
| Kurgu | yıkık | ağır | hafif | hasarsız | mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| temel | 0.691 | 0.525 | 0.403 | 0.731 | 0.588 |
| **konum TTA** (ultralytics ölçek+çevirme) | 0.698 | 0.534 | 0.406 | 0.734 | **0.593** |
| sınıflandırıcı panel TTA ×4 | 0.694 | 0.524 | 0.407 | 0.733 | 0.590 |
Konum TTA dört sınıfta da artış → **varsayılan yapıldı** (`cift_tespit(konum_tta=True)`, CLI `--konum-tta` varsayılan açık).
Tek çift görüntüde maliyeti küçük. Sınıflandırıcı TTA (+0.002, 4 kat maliyet) benimsenmedi.

### Uzman v5 (v2 + HERIDAL eğitim bölgeleri): **benimsendi**
Eğitim içi VisDrone val epoch 1–8: 0.645, 0.635, 0.598, 0.625, 0.650, 0.664, 0.668, 0.678.

| İnsan AP50 (1280) | v2 | v4 (su) | **v5** |
| :--- | :---: | :---: | :---: |
| **HERIDAL görülmemiş 7 bölge** | 0.622 | 0.568 | **0.793** |
| VisDrone val | 0.690 | 0.666 | 0.682 |
| drone-leak SAR val | 0.890 | 0.768 | 0.870 |
| SARD val | 0.864 | 0.762 | 0.865 |
| 4 set ortalaması | 0.767 | 0.691 | **0.803** |

Dağ/orman aramasında +0.17, diğerlerinde ±0.02 → `modeller/insan_uzman.pt` = v5. Kalabalık örnekte aynı eşikte v2 69 / v5 70 insan.
Sırada: v5 için eşik taraması (esik_bul), Roboflow'a v5 yükleme, su modu v2 (zincir 31) sonucu.

### Su modu v2 (`runs/insan_su2`): benimsendi
Tüm su verisi (`su_yolo2` eğitim: 27.334 döşeme / 46.472 insan; eskisinin ~4 katı) + drone-leak SAR eğitim döşemeleri, v2'den, 8 epoch.
| AP50 (1280) | v2 | su v1 (v4) | **su v2** |
| :--- | :---: | :---: | :---: |
| su valid2 (3.304 döşeme) | 0.187 | 0.571 | **0.703** |
| drone-leak SAR val | 0.890 | 0.768 | 0.820 |
Tam etiketli nehir kıyısı SAR verisi kısmi etiket zararını büyük ölçüde giderdi. `modeller/insan_su.pt` = su v2 (yalnız su/sel modu).
* `egitim_zinciri33`: **uzman v6** = v5'ten aynı veriyle 8 epoch daha (v1→v2'de bu +0.02–0.03 getirmişti; v5 eğrisi son epoch'ta hâlâ yükseliyordu).
* Roboflow: v5 `disastervision-insan-v5` olarak yüklendi (`roboflow_yukle.py --insan`; `--kosu/--ad` eklendi) ve
  `disastervision-tespit` workflow'u v5'e geçirildi (sunucu önbelleği ~8 dk).
* v5 eşik taraması (val F1): en iyi VisDrone 0.679 @0.27, SAR 0.829 @0.35, SARD 0.849 @0.38, HERIDAL görülmemiş 0.760 @0.17.
  0.30'da 0.678 / 0.825 / 0.837 / 0.743 (ortalama 0.771; 0.25'te 0.768) → varsayılan 0.30 korunur.

### Uzman v6 (v5'ten aynı veriyle 8 epoch daha): reddedildi
Eğitim içi VisDrone val epoch 1–8: 0.653, 0.652, 0.620, 0.636, 0.655, 0.674, 0.670, 0.679 (v5 son: 0.678).
| AP50 | v5 | v6 |
| :--- | :---: | :---: |
| HERIDAL görülmemiş | **0.793** | 0.792 |
| VisDrone val | 0.682 | 0.682 |
| SAR val | **0.870** | 0.866 |
| SARD val | **0.865** | 0.854 |
Ek eğitim doygun (v1→v2'deki kazanç tekrarlanmadı) → v5 etkin kalır.
* `egitim_zinciri34`: v5'i eğitimsiz olarak **1536 px** çıkarımda ölç (küçük insanlar; eğitim 1280).
* v5 @1536 (eğitimsiz): HERIDAL 0.768 (1280: 0.793), SAR 0.865 (0.870), SARD 0.860 (0.865), VisDrone val **0.700** (0.682).
  Yoğun VisDrone kalabalığında +0.018, arama-kurtarma setlerinde zarar; ortalama 0.803 → 0.798 → **1280 korunur**.
* **Tek seferlik xBD testi, konum TTA'lı son sistem:** mAP50 **0.511** (önce 0.510): yıkık 0.561 (0.546), ağır 0.483 (0.490),
  hafif 0.308 (0.303), hasarsız 0.693 (0.700); konum AP 0.759; gerçek kutularla tavan 0.758. Val'deki +0.005 testte ~0.
* v5 VisDrone test-dev (tek seferlik): **0.471** (v2 0.469).

### Lacmus / LADD (yaban arazisi arama-kurtarma) — uzman v7 hazırlığı
Universe 2-8hh1e/lacmus v4 (CC BY 4.0): 14.176 dosya = 1.356 numara × ~4 kaynak görüntü × 3 artırma (pozlama ±%4, tuz-biber).
Roboflow kareleri 2048×2048'e germiş (aslı 4:3) → `src/lacmus_veri.py` 2048×1536'ya geri açıyor, her kaynaktan tek kopya
(aynı numara + aynı etiket = aynı kaynak), numara bloklarıyla ayırıyor (valid 300–400 + 900–1000, test 600–700 + 1200–1300,
±10 tampon), 960 px döşeme. Tarla, orman, kar, kalabalık meydan sahneleri; insanlar ~50–150 px.
`egitim_zinciri36`: v5 sıfır-atış Lacmus val → **uzman v7** (v5'ten, v5 verisi + Lacmus eğitim, 8 epoch) → Lacmus val + HERIDAL +
VisDrone/SAR/SARD val. Lacmus testi yalnız sonda. Ayrıca indiriliyor: search-and-rescue-drone (orman örtüsü altı, 7.835 görüntü).
* Lacmus döşemeleri: eğitim 2.508 kaynak → 4.500 döşeme / 4.701 insan; valid 511 → 827 / 1.416; test 549 → 973 / 1.018.
  **v5 sıfır-atış Lacmus val: 0.591** → yeni bir zayıf alan (yaban arazisi / tarla / kar).
* search-and-rescue-drone (7.835): sınıflar "0/1/2" (anlamı belirsiz; "2" 12.573 kutu), "person-like", 640'a gerdirilmiş,
  karma dosya adları (dizi bazlı ayrım imkânsız) → v4 dersine göre (belirsiz/kısmi etiket genel modeli bozar) **kullanılmadı**.
* **Uzman v7 sonuçları** (val; v5 → v7): Lacmus val 0.591 → **0.717**, HERIDAL görülmemiş bölgeler 0.793 → **0.852**,
  SAR 0.870 → **0.887**, SARD 0.865 → 0.857, VisDrone val 0.682 → 0.679. Eğitim VisDrone val epoch'ları 0.649…0.675.
* **Tek seferlik Lacmus testi:** v5 0.702 → **v7 0.897**. **Karar: v7 benimsendi** (`modeller/insan_uzman.pt`).
* v7 eşik taraması (val F1, en iyi conf / 0.30'da): VisDrone 0.674@0.25 / 0.670, SAR 0.834@0.33 / 0.832, SARD 0.838@0.41 / 0.834,
  HERIDAL görülmemiş 0.796@0.27 / 0.794, Lacmus 0.726@0.28 / 0.720 → **0.30 korundu**.
* v7 VisDrone test-dev (tek seferlik): **0.461** (v5 0.471). Kentsel kalabalıkta −0.010, yaban arazisi/dağda +0.06…+0.20 → v7 kalır.
* Roboflow: disastervision-insan-v7 yüklendi, workflow `disastervision-tespit` insan adımı v7'ye geçirildi.

### Model çorbası (ağırlık ortalaması, `src/agirlik_ortala.py`) — uzman v8
v7, v5'ten ince ayarlandığı için ağırlıkları aynı havzada; doğrusal ortalama (WiSE-FT) ek eğitim olmadan denendi (val, 1280):
| Model | VisDrone | SAR | SARD | HERIDAL görülmemiş | Lacmus val |
| :--- | :---: | :---: | :---: | :---: | :---: |
| v5 | 0.682 | 0.870 | 0.865 | 0.793 | 0.591 |
| v7 | 0.679 | 0.887 | 0.857 | 0.852 | 0.717 |
| 0.5·v5 + 0.5·v7 | **0.691** | 0.899 | **0.866** | 0.854 | 0.718 |
| **0.3·v5 + 0.7·v7 (= v8)** | 0.689 | **0.902** | 0.863 | **0.861** | **0.722** |
α=0.7 beş setin hepsinde v7'den iyi → **uzman v8 benimsendi** (`runs/insan_v8`). Zincir 39: v8 tek seferlik Lacmus testi +
VisDrone test-dev; zincir 40: v6'yı da katan çorbalar (v5:v6:v7 = 1:1:2, 0.3·v6 + 0.7·v7, 0.4·v5 + 0.6·v7).
* **v8 tek seferlik testler:** Lacmus test **0.886** (v7 0.897), VisDrone test-dev **0.473** (v7 0.461, v5 0.471). Seçim val'e göre → v8 kalır.
* Zincir 40, ek çorbalar (val; VisDrone / SAR / SARD / HERIDAL / Lacmus):
  v5:v6:v7 = 1:1:2 → 0.693 / 0.904 / 0.863 / 0.860 / 0.712; 0.3·v6 + 0.7·v7 → 0.690 / 0.905 / 0.859 / 0.863 / 0.715;
  0.4·v5 + 0.6·v7 → 0.690 / 0.901 / 0.864 / 0.859 / 0.721. Hiçbiri v8'i açıkça geçmiyor (Lacmus −0.010'a kadar) → **reddedildi**.
* Roboflow: disastervision-insan-v8 yüklendi, workflow insan adımı v8'e geçirildi.
* Sırada: aynı çorba fikri hasar konum modelinde (bina2 ← bina v1'den, bina_os ← bina2'den ince ayar → aynı havza).
* Konum modeli çorbaları (bina val mAP50, 640): bina2 0.793; 0.5·bina2 + 0.5·bina_os 0.794; 0.3·bina + 0.7·bina2 0.792;
  bina:bina_os:bina2 = 1:1:2 0.792 → fark yok, **reddedildi** (konum tavanı yine etiket kalitesi).
* Sınıflandırıcı çorbaları: cift2/cift3/cift4 aynı ImageNet yolov8m-cls başlangıcından → ağırlık ortalaması anlamlı.
  Zincir 42: 0.5·cift2 + 0.5·cift4 ve cift2:cift3:cift4 = 1:1:1 tek model olarak iki aşama val (kıyas topluluk 0.593).
  Topluluk düzeyinde çıkarsa çıkarım maliyeti yarıya iner.
* **Zincir 42 sonucu** (iki aşama val, iki+tek; topluluk cift2 + cift4(T=2) = **0.593**):
  0.5·cift2 + 0.5·cift4 çorbası tek model **0.591** (yıkık 0.688, ağır 0.531, hafif 0.408, hasarsız 0.738) — topluluğa
  denk, sınıflandırıcı maliyeti yarı; cift2:cift3:cift4 çorbası 0.544 (aşırı emin cift3 bozuyor).
  Kayıtlı olasılıklarla çevrimdışı: topluluk ⊕ çorba (0.5/0.5) 0.598 (+0.005), ama ağırlık val'de seçildi, üç model gerekiyor
  ve önceki +0.005'lik val kazançları teste geçmedi (konum TTA: val +0.005 → test +0.001) → **varsayılan değişmedi**.
  Hızlı mod gerekirse `--siniflandirici runs/corba/cift24.pt --sicaklik 1` (−0.002).
* Panel çözünürlüğü: paneller 128 px, kırpma ~1.6 × bina (xBD binaları 10–40 px) → kaynak zaten büyütülüyor; daha büyük panel
  bilgi eklemez, denenmedi. Hasar tarafı bu donanımda doymuş görünüyor.
* **Zincir 43:** uzman v9 = v8'den devam (v7 verisi, AdamW lr0 5e-4, 6 epoch, ~3 saat) → v8+v9 çorbaları (0.5, 0.3) → beş val seti.
  (v6 tek başına reddedilmişti ama çorbada SAR/VisDrone'a katkı verdi; "devam + çorba" tarifi.)
* **Zincir 43 sonucu — uzman v9** (v8'den devam, 6 epoch; eğitim VisDrone val 0.651 → 0.687). Val (VisDrone / SAR / SARD / HERIDAL / Lacmus):
  | Model | VisDrone | SAR | SARD | HERIDAL | Lacmus | ort. |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | v8 | 0.689 | 0.902 | 0.863 | 0.861 | 0.722 | 0.807 |
  | v9 | 0.685 | 0.896 | 0.859 | 0.838 | 0.721 | 0.800 |
  | 0.5·v8 + 0.5·v9 | 0.690 | 0.914 | 0.856 | 0.851 | 0.722 | 0.807 |
  | 0.7·v8 + 0.3·v9 | 0.691 | 0.917 | 0.852 | 0.856 | 0.721 | 0.807 |
  SAR kazancı (tek veri seti, 3 video) SARD/HERIDAL kaybıyla dengeleniyor → **reddedildi**, v8 kalır. Mevcut veriyle insan
  uzmanı doygunluğa yakın; sonraki kazanç yeni veri alanından gelmeli.

### Tambo nehir/sal drone verisi (uzman v10 hazırlığı)
Universe tiny-person-public/tiny-person-detection-stwdp v6 (CC BY 4.0): adı "Tiny Person" ama TinyPerson değil — 8 nehir/kıyı
drone videosunun ("Tambo") 24.002 kopyası; sallardaki, kıyıdaki ve sudaki çok küçük insanlar (sel senaryosuna yakın).
Roboflow 640×640'a germiş + her kareden 3 kopya (±10° döndürme, yatay çevirme). `src/tambo_veri.py`: her kareden en az siyah
köşeli (en az döndürülmüş) kopya, 1138×640'a geri açma, **video bazında** ayrım (valid New-Tambo-5 + 8-17-23-Tambo-2,
test New-Tambo-2, eğitim kalan 5 video her 3. kare): eğitim 2.677 kare / 9.022 insan, valid 1.177 / 4.260, test 778 / 1.533.
Etiketler görsel olarak doğrulandı (sallardaki insanlar doğru kutulu).
* **Sıfır-atış Tambo val: v8 0.141, su modu 0.000** → çok zayıf yeni alan (düşük çözünürlük, sal üstü insanlar).
* Zincir 44: **uzman v10** = v8'den, v7 verisi + Tambo eğitim, 8 epoch → v8+v10 çorbaları (0.5, 0.7) → Tambo val + beş val seti.
* **Zincir 44 sonucu — uzman v10** (v8'den, + Tambo eğitim, 8 epoch; eğitim VisDrone val 0.654 → 0.674). Val:
  | Model | Tambo | VisDrone | SAR | SARD | HERIDAL | Lacmus |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | v8 | 0.141 | 0.689 | 0.902 | 0.863 | 0.861 | 0.722 |
  | v10 | **0.885** | 0.681 | 0.866 | 0.838 | 0.829 | 0.723 |
  | 0.5·v8 + 0.5·v10 | 0.558 | 0.693 | 0.907 | 0.859 | 0.863 | 0.737 |
  | **0.3·v8 + 0.7·v10 (= v11)** | **0.806** | 0.689 | 0.895 | 0.856 | 0.855 | 0.732 |
  v10 tek başına yeni alanı öğreniyor ama eski SAR setlerinde −0.03 (unutma). Çorba 0.7 Tambo'nun çoğunu koruyup diğerlerinde
  ≤0.007 kaybediyor, Lacmus +0.010 → **uzman v11 benimsendi** (`runs/insan_v11`, Roboflow disastervision-insan-v11).
  Zincir 45: tek seferlik Tambo testi (v8 ile), Lacmus testi, VisDrone test-dev.
* **v11 tek seferlik testler:** Tambo test (New-Tambo-2) v8 0.123 → **0.423**; Lacmus test **0.900** (v8 0.886);
  VisDrone test-dev 0.470 (v8 0.473). Tambo val 0.806 ile test 0.423 arasındaki fark büyük: 5 eğitim videosu yeni videoya
  sınırlı genelleşiyor (val videoları eğitim videolarına benziyor olabilir). Nehir/sal hâlâ en zayıf alan.
* Roboflow: disastervision-insan-v11 yüklendi, workflow insan adımı v11'e geçirildi.
* Tambo test neden düşük: test videosunda insanlar daha küçük (medyan yükseklik 13 px; eğitim 17, val 20) ve daha az
  (1.97/kare; val 3.62) — daha yüksek irtifa, 640'a indirilmiş kaynakta birkaç piksellik noktalar. Veri çözünürlüğü sınırı.
* **Su modu çorbaları** (v11 ⊕ su2; su2 de v2 soyundan). Su val / Tambo val: su2 0.703 / 0.000; v11 0.390 / 0.806;
  0.3·su2 0.586 / 0.147; 0.5·su2 0.656 / 0.005; 0.7·su2 0.689 / 0.000 → su2 nehir/sal sahnelerini bastırıyor, **reddedildi**.
  Not: nehir/sal sahnelerinde su modu değil varsayılan uzman (v11) kullanılmalı.
* Zincir 46: **su modu v3** = v11'den, su verisi + drone-leak + Tambo eğitim, 8 epoch → su val (kıyas 0.703) + Tambo val.
* **Zincir 46 sonucu — su modu v3 benimsendi** (`runs/insan_su3`, eğitim içi su val 0.646 → 0.715, 8 epoch ~6,6 saat).
  Su val: su2 0.703 → **0.715**; Tambo val: su2 0.000 → **0.887** (v11 0.806). `modeller/insan_su.pt` = su3
  (eski su2 yedeği `runs/insan_su2_yedek.pt`). README'deki "nehir/sal'da su modunu açmayın" uyarısı kaldırıldı.
* Zincir 47: su3 v11 soyundan → v11⊕su3 çorbaları (su3 payı 0.5, 0.7) + su3 tek başına; su, Tambo ve beş kara val seti.
  Amaç: karada v11 düzeyini koruyan tek bir model (su modu anahtarı gereksiz olur).
* **Zincir 47 sonucu — birleşik model reddedildi.** Val mAP50:
  | Model | Su | Tambo | VisDrone | SAR | SARD | HERIDAL | Lacmus |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | v11 | 0.390 | 0.806 | 0.689 | 0.895 | 0.856 | 0.855 | 0.732 |
  | su3 | **0.715** | **0.887** | 0.071 | 0.799 | 0.377 | 0.254 | 0.177 |
  | 0.5·v11 + 0.5·su3 | 0.665 | 0.874 | 0.444 | 0.898 | 0.743 | 0.721 | 0.573 |
  | 0.3·v11 + 0.7·su3 | 0.701 | 0.886 | 0.255 | 0.869 | 0.631 | 0.567 | 0.404 |
  su3 karada (VisDrone/SARD/HERIDAL/Lacmus) çökmüş: su verisindeki kısmi etiket (yalnız yüzücüler) karadaki insanları
  bastırmayı öğretiyor. Çorbalar kara setlerinde 0.1–0.25 kaybediyor → su modu ayrı kalır; README'ye "karada açmayın" eklendi.
* Zincir 48: tek seferlik su testi (su2 ile su3) + su3 Tambo testi (v11 0.423).
* **Zincir 48 (tek seferlik):** su test su2 0.611 → **su3 0.639**; Tambo test su3 **0.448** (v11 0.423).
* Zincir 49: Tambo val çıkarım boyutu taraması (v11, su3; 1600/1920/2272). Tambo kareleri 1138×640, test insanları ~13 px →
  büyütülmüş çıkarım yardımcı olursa tahmin.py'de "düşük çözünürlüklü kare → büyük imgsz" kuralı düşünülecek
  (diğer val setleri 960 döşeme; 1536 daha önce VisDrone val +0.018, HERIDAL −0.025 vermişti).
* **Zincir 49 sonucu — reddedildi.** Tambo val çıkarım boyutu (v11 / su3): 1280 0.806 / 0.887; 1600 0.810 / 0.893;
  1920 0.790 / 0.886; 2272 0.763 / 0.857. Büyütme kazancı ≤0.006 → kural eklenmedi (küçük insan sorunu çıkarım boyutuyla çözülmüyor).
* Zincir 50: v11 @1536 tüm kara val setleri + Tambo (v11 @1280: VisDrone 0.689, SAR 0.895, SARD 0.856, HERIDAL 0.855,
  Lacmus 0.732, Tambo 0.806). Ortalama belirgin artmazsa 1280 kalır.
* **Zincir 50 sonucu — reddedildi.** v11 @1536 (1280'e göre): VisDrone 0.708 (+0.019), SAR 0.894, SARD 0.855,
  HERIDAL 0.830 (−0.025), Lacmus 0.721 (−0.011), Tambo 0.807 → ortalama −0.003. 1280 kalır.

### Durum değerlendirmesi (2026-11-05)
Arama-kurtarma setleri ~0.9'a ulaştı: Lacmus test 0.900, SAR val 0.895, SARD 0.856, HERIDAL 0.855; su modu v3 Tambo val 0.887.
Son denemelerin hepsi (v9 devam, konum/sınıflandırıcı/su çorbaları, birleşik su+kara çorbası, çıkarım boyutu 1536/1600+)
≤+0.006 ya da zararlı → mevcut veri ve 6 GB GPU ile tavan. Kalan zayıf alanlar yapısal:
* VisDrone test-dev 0.470: şehir kalabalığı, çok küçük insanlar; yayınlanmış YOLO sonuçları da bu bandda. Daha büyük model
  (YOLOv8m) 6 GB VRAM'de batch 2 ile v2'nin altında kaldı.
* xBD hasar 0.511: tavan etiket kalitesi (bina sınırı/hasar sınıfı tutarsız, hafif hasar zayıf).
* Tambo test 0.423/0.448: 5 eğitim videosu, test videosunda ~13 px insanlar.
Bunları aşmak için yeni ve tam etiketli veri ya da daha büyük GPU gerekir. Otonom döngü burada durduruldu.

### Kullanıcı "devam" (2026-11-05) — zincir 51: 1536 ince ayar
* Universe'te yeni drone insan verisi arandı: büyük setlerin çoğu kaynağı belirsiz (VisDrone türevi olabilir → test-dev
  sızıntısı riski) → kullanılmadı.
* Zincir 50'de 1536 çıkarım VisDrone'a +0.019 verdi ama HERIDAL −0.025 (model 1280'de eğitilmiş). Hipotez: 1536'da ince ayar
  bu kaybı kapatır. **v12** = v11'den, v10 verisi, imgsz 1536, batch 2, 6 epoch → v11⊕v12 çorbaları (0.5, 0.7) →
  6 val seti @1280 ve @1536. Ölçüt: ortalama ≥ +0.01 ve hiçbir set > 0.01 düşmesin.
* **Zincir 51 sonucu — v12 reddedildi.** Val mAP50 (v12 / 0.5·v11+0.5·v12 / 0.3·v11+0.7·v12):
  | | VisDrone | SAR | SARD | HERIDAL | Lacmus | Tambo |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | v11 @1280 | 0.689 | 0.895 | 0.856 | 0.855 | 0.732 | 0.806 |
  | @1280 | 0.645 / 0.679 / 0.668 | 0.842 / 0.890 / 0.876 | 0.763 / 0.815 / 0.792 | 0.814 / 0.853 / 0.840 | 0.743 / 0.744 / 0.744 | 0.656 / 0.785 / 0.758 |
  | @1536 | 0.670 / 0.700 / 0.693 | 0.823 / 0.890 / 0.871 | 0.778 / 0.824 / 0.809 | 0.787 / 0.827 / 0.815 | 0.718 / 0.729 / 0.727 | — |
  1536 ince ayar (batch 2) SARD/Tambo'yu bozdu; en iyi çorba bile SARD −0.04. v11 korunur.
* Zincir 52: **YOLO11s** (ultralytics 8.4, COCO'dan) v10 veri karışımı, 1280, batch 4, 12 epoch (~12 saat) → 6 val seti.
  Farklı mimari → çorba yok; doğrudan v11 ile kıyas. Not: Roboflow yüklemesi için .venv_yukle (ultralytics 8.0.196) YOLO11'i
  tanımaz → benimsenirse yükleme yolu ayrıca çözülecek.
* **Zincir 52 durduruldu — YOLO11s reddedildi.** COCO'dan VisDrone val: epoch 1–3 0.365, 0.397, 0.397 (~36 dk/epoch).
  YOLOv8m denemesiyle aynı tablo (epoch 3 0.364): v11'in arkasında yüzlerce epoch'luk soy (genis → genis2 → v1 … v11) var;
  yeni mimari 12 epoch'ta yetişemez, yetişmesi ~40+ saat ister ve kazanç belirsiz → durduruldu.

### Durum (2026-11-05 öğlen) — tavan
"devam" sonrası iki deneme daha (1536 ince ayar, YOLO11s) kazanç vermedi. Hasar tarafında konum (YOLOv8m, WBF, eşleşmeli
topluluk, çoğaltma, 1024) ve sınıflandırıcı (çorba, üçlü, panel çözünürlüğü) seçenekleri daha önce tükenmişti.
Mevcut donanım ve veriyle anlamlı iyileştirme kalmadı. Sonraki ilerleme için gerekenler:
1. VisDrone 0.47: tam etiketli yeni şehir/kalabalık drone verisi (kaynağı belli, VisDrone türevi olmayan) ya da ≥12 GB GPU
   (YOLOv8m/YOLO11m 1280, batch ≥8).
2. xBD 0.51: hasar etiketlerinin temizlenmesi (özellikle hafif hasar) — elle etiket düzeltme.
3. Tambo test 0.45: daha fazla nehir/sal videosu (yüksek irtifa, küçük insan).

### Okutama-Action (2026-11-06) — zincir 53
* Universe'teki sel setleri ("Flood" 8.9k, "Drone People Flood") incelendi: yerden çekilmiş haber fotoğrafları → elendi.
* **Okutama-Action** (okutama-action.org, Dropbox; eğitim kareleri 5.77 GB, test 1.65 GB): 43 dk 4K drone videosu, 10–45 m,
  1280×720 kareler. Etiketler 3840×2160 koordinatında → /3 (görsel doğrulandı). `src/okutama_veri.py`:
  SingleActionLabels, lost=1 atılır, her 180 karede iz kimliği yenilendiği için çift kutular IoU>0.7 ile tekilleştirilir;
  video/senaryo bazında ayrım (aynı senaryoyu iki drone çektiği için ikisi birlikte): valid senaryo 1.5 + 2.6 (4 video),
  test = resmî test seti (10 video), train = kalan 29 video. Train her 10. kare 5.348 kare / 31.287 insan; valid 447 / 2.255;
  test 1.160 / 5.149 (her 15. kare).
* Dikkat: yalnız eylem oyuncuları etiketli, kenardaki seyirciler etiketsiz (seyrek) → kısmi etiket riski; çorba ve
  diğer val setleri karar verir.
* Zincir 53: v11 sıfır-atış Okutama val → v13 = v11'den, v10 verisi + Okutama train, 8 epoch → 0.5/0.7 çorbaları → 7 val seti.
* **v11 sıfır-atış Okutama val: 0.637** (mAP50-95 0.245). v13 eğitimi ~55 dk/epoch (8 epoch ~7,5 saat).
* **v13 eğitim ara sonucu** (Okutama val, eğitim içi): epoch 1–6: 0.653, 0.643, 0.625, 0.644, 0.657, **0.666** (v11 sıfır-atış 0.637).
* 2026-11-06 ~14:10 bilgisayar kapandı/uyudu (epoch 7/8 %95'te; otomatik kayıt da durdu). 17:50 **zincir 53b**:
  `egit.py --devam` ile v13 sürdürüldü → v11⊕v13 çorbaları (0.5, 0.7) → 7 val seti (Okutama + altı eski set).
* v13 epoch 7 (devamdan sonra): Okutama val **0.673** (v11 sıfır-atış 0.637).
* v13 epoch 8: Okutama val **0.680** (eğitim bitti; best.pt = ep8). Sırada çorbalar + 7 val seti.
* Okutama val (degerlendir.py, 1280): v11 0.637 → **v13 0.773**, 0.5 çorba 0.723, 0.7 çorba 0.750.
* **Zincir 53 sonucu — uzman v14 = 0.3·v11 + 0.7·v13 benimsendi.** Val mAP50 (1280):
  | Model | Okutama | VisDrone | SAR | SARD | HERIDAL | Lacmus | Tambo | ort. |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | v11 | 0.637 | 0.689 | 0.895 | 0.856 | 0.855 | 0.732 | 0.806 | 0.781 |
  | v13 | 0.773 | 0.683 | 0.914 | 0.860 | 0.875 | 0.738 | 0.888 | 0.819 |
  | 0.5·v11 + 0.5·v13 | 0.723 | 0.693 | 0.926 | 0.863 | 0.879 | 0.744 | 0.877 | 0.815 |
  | **0.3·v11 + 0.7·v13 (= v14)** | 0.750 | 0.690 | 0.924 | 0.861 | 0.882 | 0.740 | **0.890** | **0.820** |
  Kısmi etiket (seyirciler etiketsiz) korkusu gerçekleşmedi: Okutama'daki tam etiketli eylem oyuncuları (yürüyen, koşan,
  yatan, oturan insanlar, 10–45 m) **her** sette kazanç verdi — SAR 0.924, HERIDAL 0.882 ve Tambo 0.890 şimdiye kadarki en iyi.
  `runs/insan_v14`. Zincir 54: tek seferlik testler (Okutama test v11'le kıyas, Lacmus, Tambo, VisDrone test-dev).
* **Zincir 54 — v14 tek seferlik testler:** Okutama resmî test v11 0.693 → **0.820**; Lacmus test 0.900 → **0.914**;
  Tambo test 0.423 → **0.435**; VisDrone test-dev 0.470 → 0.470. Hiçbir test düşmedi.
* **v14 ETKİN:** `modeller/insan_uzman.pt` = v14 (v11: `runs/insan_v11`). Roboflow disastervision-insan-v14 yüklendi,
  workflow disastervision-tespit insan adımı v14'e geçirildi. README güncellendi.
* Ders: kısmi etiketli görünen veri (seyirciler etiketsiz) bu kez zarar vermedi — etiketsizler kenarda ve seyrek; çoğunluk tam
  etiketli. Su verisinde ise kişilerin çoğu (teknedekiler) etiketsizdi. Ölçüt: etiketsiz insan oranı.

### Yeni tam etiketli insan verisi arayışı (2026-11-06 akşam)
* **NOMAD** (Notre Dame, WACV 2024; 100 oyuncu, 10–90 m, gizlenen/yatan insanlar, 42.8k kare, YOLO etiketli) — ideal SAR verisi
  ama Google Drive'da 10 × 38 GB zip (5.4K kareler) → bu bağlantıyla (≤2 MB/s) pratik değil; annotations.json kotaya takıldı.
  İleride tek zip (10 oyuncu) gece indirmesi düşünülebilir.
* **AU-AIR** (ICRA 2020, CC BY 4.0; Drive aynası fmi-data-index): 32.8k kare 1920×1080, 5–30 m trafik videoları, 8 sınıf tam
  etiketli; Human 5.158 kutu / 4.712 kare, medyan yükseklik 68 px. İndiriliyor (2.2 GB). `src/auair_veri.py`: yalnız insan,
  video bazlı bölme (~%15 valid), insanlı karelerin yarısı + insansız karelerin 1/40'ı (gerçek negatif).
* **Zincir 55 — v14 eşik taraması:** en iyi F1 eşikleri 0.27–0.36 aralığında; 0.30 her sette en iyinin ≤0.004 altında → varsayılan conf 0.30 korunur.
* **AU-AIR reddedildi:** v14 sıfır-atış val 0.148; hata analizi etiket sorununu gösterdi — arabaya "Human" etiketi, birçok gerçek
  yaya etiketsiz (modelin "yanlış alarm"larının çoğu gerçek insan). Trafik veri seti, insan sınıfı ikincil ve eksik → kısmi
  etiketle eğitim riskli, kullanılmadı.
* **v14 eşik taraması** yukarıda (0.30 korunur).
* **SeaDronesSee ODv2** (resmî link: POST /dataset/ODv2_redirecter → cloud.cs.uni-tuebingen.de/s/ZZxX65FGnQ8zjBP, WebDAV):
  eğitim 8.930 görüntü = mevcut su verisiyle (Roboflow sigmoidal) aynı sayı → aynı kaynak; yalnız `swimmer` etiketli, teknedeki
  insanlar yine yok → yeni bilgi yok, kullanılmadı.
* **NOMAD** zip 1 indirmesi Drive kotasına takıldı; `nomad_indir.ps1` 10 dk arayla 20 kez yeniden deniyor.
* **WiSARD** (UW, IROS 2022; yaban arazisi SAR, görünür + termal, ~26.8k görünür etiketli görüntü; Drive: WiSARDv1.zip 41 GB,
  örnek 972 MB): örnek iniyor → etiket formatı/kalitesi kontrol edilecek.
* **WiSARD örnek** (tek uçuş, 210417_MtErie_Enterprise_VIS_0003, 264 kare 3840×2160, YOLO etiket, tek sınıf): etiketler görsel
  olarak doğru ve tam (kayalıkta yatan/oturan, ağaç arasında küçük insanlar). v14 tam kare @1280: 0.369 (60 karede 234 insanın
  177'si kaçırılıyor, yalnız 19 yanlış alarm → gerçek zayıflık, etiket sorunu değil). `src/doseme_yap.py` (960 px döşeme, %15
  örtüşme) ile 3.960 döşeme / 1.926 kutu: **v11 0.618, v14 0.661** — diğer SAR setlerinin (~0.88) çok altında → WiSARD'ın
  tamamı yüksek değerli. WiSARDv1.zip (41 GB) Drive kotasına takıldı; `wisard_indir.ps1` 5 dk arayla 30 kez deniyor.
  Not: bu örnek uçuş tam sette de var → eğitimde MtErie uçuşları val'e ayrılmalı.
* Universe aynaları (Drive kotası beklenirken) — ikisi de **reddedildi**:
  - helmetdetection-426oy/nomad-aerial-human-2 (2.058 kare, NOMAD alt kümesi): 640×640'a ezilmiş, yansıtma dolgulu
    (kutular aynalanmış bölgede kopyalanıyor), yalnız a10/a30, oyuncu karelerinin yarısı etiketsiz → bozuk.
  - roboflow-public/sarnet-search-and-rescue v5: mozaik + 3× çoğaltılmış, çok yüksek irtifa, `target` sınıfı insan dışı
    nesneleri de içeriyor, hedefler birkaç piksel → uyumsuz.
  Sıradaki gerçek kaynaklar: WiSARD (41 GB) ve NOMAD (zip başına 38 GB), Drive kotası açılınca kendiliğinden inecek.
