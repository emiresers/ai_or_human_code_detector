"""
FastAPI Backend - AI/Human Code Detection
3 model kullanarak kod analizi yapar ve sonuçları döner
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys
import io

# Windows terminal emoji encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ======================
# MODEL ve TF-IDF YÜKLEME
# ======================
print("=" * 60)
print("BACKEND BAŞLATILIYOR")
print("=" * 60)

try:
    print("\n📁 Modeller yükleniyor...")
    
    import os
    # Root klasörüne git (backend klasöründen bir üst)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Vectorizer yükle
    vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
    vectorizer = joblib.load(vectorizer_path)
    print("   ✓ TF-IDF Vectorizer yüklendi")
    
    # Modelleri yükle
    model_lr_path = os.path.join(root_dir, "model_logistic_regression.pkl")
    model_lr = joblib.load(model_lr_path)
    print("   ✓ Logistic Regression yüklendi")
    
    model_nb_path = os.path.join(root_dir, "model_naive_bayes.pkl")
    model_nb = joblib.load(model_nb_path)
    print("   ✓ Naive Bayes yüklendi")
    
    model_rf_path = os.path.join(root_dir, "model_random_forest.pkl")
    model_rf = joblib.load(model_rf_path)
    print("   ✓ Random Forest yüklendi")
    
    print("\n✅ Tüm modeller başarıyla yüklendi!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ HATA: Model yükleme başarısız!")
    print(f"   Hata: {str(e)}")
    print("\n⚠️  Lütfen önce train_models.py çalıştırın!")
    raise

# ======================
# FASTAPI UYGULAMASI
# ======================
app = FastAPI(
    title="AI or Human Code Detector",
    description="3 ML modeli kullanarak kod analizi yapan API",
    version="1.0.0"
)

# CORS ayarları (Frontend'den erişim için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler kullan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend'i servis et (Static files)
# Root klasörüne git (backend klasöründen bir üst)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(root_dir, "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    def serve_frontend():
        """Frontend'i ana sayfa olarak servis et"""
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend bulunamadı"}
    
    print(f"   ✓ Frontend servis ediliyor: {frontend_dir}")

# ======================
# REQUEST/RESPONSE MODELLERİ
# ======================
class CodeInput(BaseModel):
    code: str

# ======================
# YARDIMCI FONKSİYONLAR
# ======================
def get_prediction(model, X_vectorized):
    """
    Model'den tahmin ve olasılık alır
    
    Returns:
        dict: {
            "prediction": "ai" veya "human",
            "prob_ai": float (0-1 arası),
            "prob_human": float (0-1 arası)
        }
    """
    # Olasılık tahminleri
    proba = model.predict_proba(X_vectorized)[0]
    
    # Sınıf tahmini
    pred = model.predict(X_vectorized)[0]
    
    # Sınıf indekslerini bul
    classes = list(model.classes_)
    try:
        ai_idx = classes.index("ai")
        human_idx = classes.index("human")
    except ValueError:
        # Eğer sınıf isimleri farklıysa
        ai_idx = 0
        human_idx = 1
    
    return {
        "prediction": pred,
        "prob_ai": float(proba[ai_idx]),
        "prob_human": float(proba[human_idx])
    }

# ======================
# API ENDPOINT'LERİ
# ======================
@app.post("/predict")
def predict_code(data: CodeInput):
    """
    Kod analizi yapar ve 3 modelin sonuçlarını döner
    
    Returns:
        dict: {
            "logistic_regression": {...},
            "naive_bayes": {...},
            "random_forest": {...},
            "average": {
                "prob_ai": float,
                "prob_human": float
            },
            "final_decision": "ai" veya "human"
        }
    """
    # Kod metnini al
    text = data.code
    
    # TF-IDF ile vektörize et
    X = vectorizer.transform([text])
    
    # Her model için tahmin yap
    lr_result = get_prediction(model_lr, X)
    nb_result = get_prediction(model_nb, X)
    rf_result = get_prediction(model_rf, X)
    
    # Ortalama hesapla (her model eşit ağırlıkta)
    avg_ai = (lr_result["prob_ai"] + nb_result["prob_ai"] + rf_result["prob_ai"]) / 3
    avg_human = (lr_result["prob_human"] + nb_result["prob_human"] + rf_result["prob_human"]) / 3
    
    # Final karar (ortalamaya göre)
    final_decision = "ai" if avg_ai > avg_human else "human"
    
    # Response oluştur
    response = {
        "logistic_regression": {
            "prediction": lr_result["prediction"],
            "prob_ai": round(lr_result["prob_ai"], 4),
            "prob_human": round(lr_result["prob_human"], 4)
        },
        "naive_bayes": {
            "prediction": nb_result["prediction"],
            "prob_ai": round(nb_result["prob_ai"], 4),
            "prob_human": round(nb_result["prob_human"], 4)
        },
        "random_forest": {
            "prediction": rf_result["prediction"],
            "prob_ai": round(rf_result["prob_ai"], 4),
            "prob_human": round(rf_result["prob_human"], 4)
        },
        "average": {
            "prob_ai": round(avg_ai, 4),
            "prob_human": round(avg_human, 4)
        },
        "final_decision": final_decision
    }
    
    return response

@app.get("/api")
def api_root():
    """API health check endpoint"""
    return {
        "status": "ok",
        "message": "Backend çalışıyor!",
        "models_loaded": True
    }

@app.get("/health")
def health():
    """Detaylı health check"""
    return {
        "status": "ok",
        "models_loaded": True,
        "vectorizer_loaded": True,
        "models": ["Logistic Regression", "Naive Bayes", "Random Forest"]
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Backend başlatılıyor...")
    print("   URL: http://127.0.0.1:8000")
    print("   Docs: http://127.0.0.1:8000/docs")
    print("\n" + "=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)

