# AI or Human Code Detector

3 farklı ML modeli (Logistic Regression, Naive Bayes, Random Forest) kullanarak kod analizi yapan web uygulaması.

## 🚀 Projeyi Çalıştırma

### 1. Backend'i Başlatma

**Terminal 1 (Backend):**

```powershell
# Proje klasörüne git
cd ai_or_human

# Backend'i başlat
python backend/main.py
```

**Veya backend klasöründen:**
```powershell
cd backend
py main.py
```

**Başarılı başlatma çıktısı:**
```
============================================================
BACKEND BAŞLATILIYOR
============================================================

📁 Modeller yükleniyor...
   ✓ TF-IDF Vectorizer yüklendi
   ✓ Logistic Regression yüklendi
   ✓ Naive Bayes yüklendi
   ✓ Random Forest yüklendi

✅ Tüm modeller başarıyla yüklendi!
============================================================

🚀 Backend başlatılıyor...
   URL: http://127.0.0.1:8000
   Docs: http://127.0.0.1:8000/docs
```

### 2. Frontend'i Açma

Backend başladıktan sonra, tarayıcıda şu adresi açın:

```
http://127.0.0.1:8000
```

**Not:** Frontend backend üzerinden servis ediliyor. Ayrı bir sunucu gerekmez!

## 📋 Adım Adım Kullanım

1. **Backend'i başlat:**
   ```powershell
   py backend/main.py
   ```

2. **Tarayıcıda aç:**
   - `http://127.0.0.1:8000` adresini açın

3. **Kod analizi yap:**
   - Textarea'ya kod yapıştırın
   - "Kodu Analiz Et" butonuna tıklayın
   - Sonuçları görün

## 🔧 Sorun Giderme

### Port 8000 zaten kullanımda hatası:

**Hızlı Çözüm (Tek Komut):**
```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq "Listen"} | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force; Write-Host "Process kapatildi!" }
```

**Adım Adım:**
1. Port'u kullanan process'i bul:
   ```powershell
   Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq "Listen"}
   ```

2. Process ID'yi görüp kapat:
   ```powershell
   Stop-Process -Id <PROCESS_ID> -Force
   ```

**Veya Tüm Python Process'lerini Kapat:**
```powershell
Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "py"} | Stop-Process -Force
```

Sonra tekrar başlatın:
```powershell
py backend/main.py
```

### Backend'e bağlanılamıyor:

1. Backend'in çalıştığından emin olun
2. `http://127.0.0.1:8000/health` adresini kontrol edin
3. Tarayıcı konsolunda (F12) hata mesajlarını kontrol edin

## 📁 Proje Yapısı

```
ai_or_human/
├── backend/
│   └── main.py              # FastAPI backend (API + Frontend servisi)
├── frontend/
│   ├── index.html           # Ana sayfa (UI)
│   ├── script.js            # Frontend logic (API çağrıları)
│   └── style.css            # Stil dosyası
├── ai_dataset.csv           # AI kod örnekleri (6,650 örnek)
├── human_dataset.csv        # Human kod örnekleri (5,034 örnek)
├── combined_dataset.csv     # Birleştirilmiş dataset (11,884 örnek)
├── merge_datasets.py        # Dataset birleştirme script'i
├── train_models.py          # Model eğitimi script'i (3 ML modeli)
├── model_logistic_regression.pkl
├── model_naive_bayes.pkl
├── model_random_forest.pkl
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer (açıklama aşağıda)
└── README.md                # Bu dosya
```

### 📊 Dataset Açıklaması

**Basit Mantık:**
1. **ai_dataset.csv** → AI yazımı kod örnekleri (6,650 örnek)
2. **human_dataset.csv** → İnsan yazımı kod örnekleri (5,034 örnek)
3. **combined_dataset.csv** → İkisini birleştirir (`merge_datasets.py` ile oluşturulur)
   - AI: 6,850 örnek (6,650 ana + 200 ai_extra pattern'leri)
   - Human: 5,034 örnek
   - Toplam: 11,884 örnek
   - Model eğitimi için kullanılır

**Nasıl Çalışır:**
```
ai_dataset.csv + human_dataset.csv 
    ↓ (merge_datasets.py çalıştır)
combined_dataset.csv 
    ↓ (train_models.py çalıştır)
Eğitilmiş Modeller (*.pkl dosyaları)
```

### 🔤 TF-IDF Nedir?

**TF-IDF (Term Frequency-Inverse Document Frequency)** = Kelime Sıklığı-Ters Belge Sıklığı

**Çalışma Prensibi:**
- Metinsel kod verilerini makine öğrenmesi algoritmaları için sayısal vektörlere dönüştürür.
- Her kelime veya n-gram dizisine bir ağırlık (önem skoru) atar.
- Sık kullanılan sıradan kelimeler (örn: "def", "return") daha düşük ağırlık alırken, ayırt edici ve nadir terimler daha yüksek ağırlık alır.

**Projedeki Rolü:**
- Yapay zeka tarafından üretilen kodlar, genellikle belirli bağlamsal kelime öbekleri ve tekrarlı dizilimler içerir.
- İnsan yazımı kodlar ise adlandırma ve yapısal olarak daha farklı özellikler gösterir.
- TF-IDF vektörizasyonu, insan ve yapay zeka arasındaki bu yapısal kelime dizilimi farklılıklarını sayısal olarak ortaya çıkararak modellerin öğrenmesini sağlar.

**Örnek:**
```
AI Kodu: "class ContextualizedOperationalUnit"
→ TF-IDF: "contextualized"=0.8, "operational"=0.7, "unit"=0.3

Human Kodu: "def topla(a, b): return a + b"
→ TF-IDF: "def"=0.1, "topla"=0.9, "return"=0.2
```

**Teknik Detay:**
- `tfidf_vectorizer.pkl` → Eğitilmiş TF-IDF modeli
- Her kod girişinde aynı şekilde sayılara çevrilir
- 3 ML modeli bu sayıları kullanarak AI/Human kararı verir

## 🎯 Model Performansı

- **AI Tespiti:** %100 doğruluk
- **Human Tespiti:** %90 doğruluk
- **Genel Doğruluk:** %93.3

## 📝 Notlar

- Backend ve frontend aynı sunucuda çalışır (port 8000)
- Model dosyaları proje root'unda olmalı
- İlk başlatmada modeller yüklenir (birkaç saniye sürebilir)

