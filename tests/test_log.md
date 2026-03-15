# White Box Test Raporu

Backend kod yapısı (FastAPI ve model yükleme metotları) referans alınarak kritik endpoint'ler ve yardımcı fonksiyonlar için White-Box testler tasarlanmış ve yürütülmüştür.
Testlerde Python pytest kütüphanesi kullanılmıştır.

## Kullanılan Araçlar
- Python
- pytest
- FastAPI TestClient

## Gerçekleştirilen Testler

### Test Case 1: predict_code Fonksiyonu Testi
- **Amaç:** Sisteme gönderilen Python kodunun 3 model ile analiz edilmesini test etmek
- **Test Edilen:** `/predict` endpoint fonksiyonu
- **Beklenen Sonuç:** 
  - HTTP 200 döndürmeli
  - 3 model sonucu olmalı (Logistic Regression, Naive Bayes, Random Forest)
  - Final decision "ai" veya "human" olmalı
  - Average sonuçları hesaplanmış olmalı
- **Sonuç:** ✅ PASS

### Test Case 2: get_prediction Fonksiyonu Testi
- **Amaç:** Model olasılık hesaplamalarının doğruluğunu test etmek
- **Test Edilen:** `get_prediction()` yardımcı fonksiyonu
- **Beklenen Sonuç:**
  - Prediction "ai" veya "human" olmalı
  - prob_ai ve prob_human 0-1 arasında olmalı
  - prob_ai + prob_human = 1 olmalı
- **Sonuç:** ✅ PASS

### Test Case 3: health Endpoint Testi
- **Amaç:** Backend'in durumunu ve modellerin yüklü olup olmadığını kontrol etmek
- **Test Edilen:** `/health` endpoint fonksiyonu
- **Beklenen Sonuç:**
  - HTTP 200 döndürmeli
  - Status "ok" olmalı
  - Modeller yüklü olmalı (models_loaded: true)
  - Vectorizer yüklü olmalı (vectorizer_loaded: true)
  - 3 model listelenmeli
- **Sonuç:** ✅ PASS

## Test Sonuçları Özeti

| Test Case | Test Adı | Sonuç | Durum |
|-----------|----------|-------|-------|
| 1 | predict_code Fonksiyonu | PASS | ✅ |
| 2 | get_prediction Fonksiyonu | PASS | ✅ |
| 3 | health Endpoint | PASS | ✅ |

**Toplam Test Sayısı:** 3  
**Başarılı Test:** 3  
**Başarısız Test:** 0  
**Başarı Oranı:** 100%

## Sonuç

Tüm testler başarıyla çalıştırılmıştır. Sistem beklenen davranışı göstermektedir.

**Test Tarihi:** 2025-12-09  
**Test Versiyonu:** V.1.0  
**Test Ortamı:** Windows 10/11, Python 3.x, pytest
