# Test Dokümantasyonu - AI or Human Code Detector

## White-Box Test Cases

Proje kapsamında backend API ve genel hata yakalama mekanizmalarının kararlılığını ölçmek için **3 adet White-Box test case** kurgulanmıştır:

### Test Case 1: predict_code Fonksiyonu Testi
- **Dosya:** `tests/test_backend.py` - `test_predict_code()`
- **Ne Test Ediyor:** `predict_code()` endpoint fonksiyonunun beklenen JSON formatında doğru yanıtları verip vermediği kontrol edilir.
- **Test Edilen Fonksiyon:** `predict_code()` (backend/main.py dosyasında)
- **Test Yöntemi:** pytest kullanılarak fonksiyonların doğrudan testi sağlanır.
- **Kontrol Edilenler:**
  - HTTP status code (200)
  - Response formatı (JSON dictionary)
  - 3 model sonuçlarının varlığı (logistic_regression, naive_bayes, random_forest)
  - Final decision'ın varlığı ve doğruluğu ("ai" veya "human")
  - Average sonuçlarının hesaplanması
  - Final decision'ın ortalamaya göre doğru hesaplanması

### Test Case 2: get_prediction Fonksiyonu Testi
- **Dosya:** `tests/test_backend.py` - `test_get_prediction()`
- **Ne Test Ediyor:** `get_prediction()` fonksiyonunun modeller üzerinden doğru olasılık (probability) hesaplamaları yapıp yapmadığı kontrol edilir.
- **Test Edilen Fonksiyon:** `get_prediction()` (backend/main.py dosyasında)
- **Test Yöntemi:** pytest kullanılarak fonksiyonların doğrudan testi sağlanır.
- **Kontrol Edilenler:**
  - Return değerinin formatı (dictionary)
  - Prediction değerinin varlığı ve geçerliliği ("ai" veya "human")
  - prob_ai ve prob_human değerlerinin varlığı
  - Probability değerlerinin geçerliliği (0-1 arası)
  - prob_ai + prob_human toplamının yaklaşık 1 olması

### Test Case 3: health Endpoint Testi
- **Dosya:** `tests/test_backend.py` - `test_health_endpoint()`
- **Ne Test Ediyor:** Modellerin ve API'nin sunucuda aktif ve kullanıma hazır olup olmadığı kontrol edilir.
- **Test Edilen Fonksiyon:** `health()` (backend/main.py dosyasında)
- **Test Yöntemi:** pytest kullanılarak endpoint doğrudan test edilir.
- **Kontrol Edilenler:**
  - HTTP status code (200)
  - Status "ok" olmalı
  - Modellerin yüklü olduğu bilgisi (models_loaded: true)
  - Vectorizer'ın yüklü olduğu bilgisi (vectorizer_loaded: true)
  - 3 model listelenmeli (Logistic Regression, Naive Bayes, Random Forest)

## Test Dosyaları

- **`tests/test_backend.py`** - White-box test kodları
- **`tests/test_log.md`** - Test log'ları (test sonuçları)
- **`requirements_test.txt`** - Test bağımlılıkları
- **`run_tests.bat`** - Test çalıştırma script'i (Windows)

## Test Çalıştırma

### Windows:
```bash
run_tests.bat
```

### Manuel (Windows/Linux/Mac):
```bash
# Test bağımlılıklarını yükle
pip install -r requirements_test.txt

# Testleri çalıştır
pytest tests/test_backend.py -v

# Detaylı output ile
pytest tests/test_backend.py -v --tb=short

# Coverage raporu ile
pytest tests/test_backend.py --cov=backend --cov-report=html
```

## Test Sonuçları

Test log'ları için `tests/test_log.md` dosyasına bakınız.

### Özet:
- ✅ **Test Case 1:** PASS - predict_code fonksiyonu testi
- ✅ **Test Case 2:** PASS - get_prediction fonksiyonu testi
- ✅ **Test Case 3:** PASS - health endpoint testi

**Toplam Test Sayısı:** 3  
**Başarılı Test:** 3  
**Başarı Oranı:** 100%

## Test Gereksinimleri

- Python 3.x
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- httpx >= 0.24.0
- Model dosyaları (`model_*.pkl`, `tfidf_vectorizer.pkl`) proje root'unda olmalı

## Notlar

- Testler çalıştırılmadan önce model dosyalarının varlığı kontrol edilir
- Model dosyaları yoksa testler otomatik olarak skip edilir
- Testler FastAPI TestClient kullanarak backend'i test eder (gerçek HTTP server başlatmaz)

