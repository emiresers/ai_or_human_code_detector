"""
White-Box Test Cases for AI or Human Code Detector Backend
Python pytest kullanılarak yazılmıştır.

Test Case 1: predict_code Fonksiyonu Testi
Test Case 2: get_prediction Fonksiyonu Testi
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# Backend modülünü import etmek için path ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Backend'i import et
try:
    from main import app, get_prediction, CodeInput
    import joblib
    import numpy as np
    
    # Modelleri yükle (test için)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl")
    model_lr_path = os.path.join(root_dir, "model_logistic_regression.pkl")
    model_nb_path = os.path.join(root_dir, "model_naive_bayes.pkl")
    model_rf_path = os.path.join(root_dir, "model_random_forest.pkl")
    
    # Dosyaların varlığını kontrol et
    models_exist = all(os.path.exists(p) for p in [
        vectorizer_path, model_lr_path, model_nb_path, model_rf_path
    ])
    
    if models_exist:
        vectorizer = joblib.load(vectorizer_path)
        model_lr = joblib.load(model_lr_path)
        model_nb = joblib.load(model_nb_path)
        model_rf = joblib.load(model_rf_path)
    else:
        models_exist = False
        
except Exception as e:
    models_exist = False
    print(f"Model yükleme hatası: {e}")

# Test client oluştur
client = TestClient(app)


# ============================================================================
# TEST CASE 1: predict_code Fonksiyonu Testi
# ============================================================================
@pytest.mark.skipif(not models_exist, reason="Model dosyaları bulunamadı")
def test_predict_code():
    """
    Test Case 1: predict_code Fonksiyonu Testi
    
    Bu test, predict_code endpoint fonksiyonunun doğru çalışıp çalışmadığını test eder.
    Fonksiyon kod girişi alır, 3 model ile analiz yapar ve sonuçları döner.
    """
    # Test verilerini hazırla
    test_code = """
def topla(a, b):
    return a + b

def cikar(a, b):
    return a - b

x = 10
y = 5
sonuc = topla(x, y)
print(sonuc)
"""
    
    # Endpoint'i çağır
    response = client.post("/predict", json={"code": test_code})
    
    # Assert: HTTP status code 200 olmalı
    assert response.status_code == 200, \
        f"predict_code endpoint HTTP 200 döndürmeli (şu an: {response.status_code})"
    
    # Response JSON formatında olmalı
    result = response.json()
    assert isinstance(result, dict), "predict_code fonksiyonu dictionary döndürmeli"
    
    # Assert: 3 model sonuçları olmalı
    assert "logistic_regression" in result, "logistic_regression sonucu olmalı"
    assert "naive_bayes" in result, "naive_bayes sonucu olmalı"
    assert "random_forest" in result, "random_forest sonucu olmalı"
    
    # Assert: Final decision olmalı ve "ai" veya "human" olmalı
    assert "final_decision" in result, "final_decision olmalı"
    assert result["final_decision"] in ["ai", "human"], \
        f"final_decision 'ai' veya 'human' olmalı (şu an: {result['final_decision']})"
    
    # Assert: Average sonuçları olmalı
    assert "average" in result, "average sonucu olmalı"
    assert "prob_ai" in result["average"], "average prob_ai olmalı"
    assert "prob_human" in result["average"], "average prob_human olmalı"
    
    # Final decision'ın doğru hesaplandığını kontrol et
    avg_ai = result["average"]["prob_ai"]
    avg_human = result["average"]["prob_human"]
    expected_decision = "ai" if avg_ai > avg_human else "human"
    
    assert result["final_decision"] == expected_decision, \
        f"final_decision doğru hesaplanmalı (beklenen: {expected_decision}, alınan: {result['final_decision']})"
    
    # Test başarılı mesajı
    print(f"\n✅ Test Case 1: PASS - predict_code fonksiyonu başarıyla çalıştı")
    print(f"   Final Decision: {result['final_decision']}")
    print(f"   Average AI: {result['average']['prob_ai']:.2%}")
    print(f"   Average Human: {result['average']['prob_human']:.2%}")


# ============================================================================
# TEST CASE 2: get_prediction Fonksiyonu Testi
# ============================================================================
@pytest.mark.skipif(not models_exist, reason="Model dosyaları bulunamadı")
def test_get_prediction():
    """
    Test Case 2: get_prediction Fonksiyonu Testi
    
    Bu test, get_prediction fonksiyonunun doğru çalışıp çalışmadığını test eder.
    Fonksiyon vektörize edilmiş kod alır, model ile tahmin yapar ve sonuçları döner.
    """
    # Test verilerini hazırla
    test_code = """
class ContextualizedOperationalUnit:
    def __init__(self, contextual_parameters):
        self.operational_context = contextual_parameters
        self.initialization_pipeline = self._construct_pipeline()
    
    def _construct_pipeline(self):
        return {
            'preprocessing': self._preprocess_contextual_data,
            'orchestration': self._orchestrate_operations,
            'postprocessing': self._postprocess_results
        }
"""
    
    # Kodu vektörize et
    X_vectorized = vectorizer.transform([test_code])
    
    # Logistic Regression model ile test et
    result = get_prediction(model_lr, X_vectorized)
    
    # Assert: Sonuç bir dictionary olmalı
    assert isinstance(result, dict), "get_prediction fonksiyonu dictionary döndürmeli"
    
    # Assert: Gerekli alanlar olmalı
    assert "prediction" in result, "get_prediction sonucunda prediction olmalı"
    assert "prob_ai" in result, "get_prediction sonucunda prob_ai olmalı"
    assert "prob_human" in result, "get_prediction sonucunda prob_human olmalı"
    
    # Assert: Prediction "ai" veya "human" olmalı
    assert result["prediction"] in ["ai", "human"], \
        f"prediction 'ai' veya 'human' olmalı (şu an: {result['prediction']})"
    
    # Assert: Probability'ler 0-1 arasında olmalı
    assert 0 <= result["prob_ai"] <= 1, \
        f"prob_ai 0-1 arasında olmalı (şu an: {result['prob_ai']})"
    assert 0 <= result["prob_human"] <= 1, \
        f"prob_human 0-1 arasında olmalı (şu an: {result['prob_human']})"
    
    # AI ve Human probability'leri toplamı yaklaşık 1 olmalı
    total_prob = result["prob_ai"] + result["prob_human"]
    expected_total = 1.0
    assert abs(total_prob - expected_total) < 0.01, \
        f"prob_ai + prob_human yaklaşık 1 olmalı (beklenen: {expected_total}, alınan: {total_prob})"
    
    # Test başarılı mesajı
    print(f"\n✅ Test Case 2: PASS - get_prediction fonksiyonu başarıyla çalıştı")
    print(f"   Prediction: {result['prediction']}")
    print(f"   AI Probability: {result['prob_ai']:.2%}")
    print(f"   Human Probability: {result['prob_human']:.2%}")


# ============================================================================
# TEST CASE 3: health Endpoint Testi
# ============================================================================
@pytest.mark.skipif(not models_exist, reason="Model dosyaları bulunamadı")
def test_health_endpoint():
    """
    Test Case 3: health Endpoint Testi
    
    Bu test, health endpoint'inin doğru çalışıp çalışmadığını test eder.
    Health endpoint backend'in durumunu ve modellerin yüklü olup olmadığını kontrol eder.
    """
    # Health endpoint'ini çağır
    response = client.get("/health")
    
    # Assert: HTTP status code 200 olmalı
    assert response.status_code == 200, \
        f"health endpoint HTTP 200 döndürmeli (şu an: {response.status_code})"
    
    # Response JSON formatında olmalı
    result = response.json()
    assert isinstance(result, dict), "health endpoint dictionary döndürmeli"
    
    # Assert: Status "ok" olmalı
    assert "status" in result, "health endpoint'inde status olmalı"
    assert result["status"] == "ok", \
        f"status 'ok' olmalı (şu an: {result['status']})"
    
    # Assert: Modellerin yüklü olduğu bilgisi olmalı
    assert "models_loaded" in result, "health endpoint'inde models_loaded olmalı"
    assert result["models_loaded"] == True, \
        "models_loaded True olmalı"
    
    # Assert: Vectorizer'ın yüklü olduğu bilgisi olmalı
    assert "vectorizer_loaded" in result, "health endpoint'inde vectorizer_loaded olmalı"
    assert result["vectorizer_loaded"] == True, \
        "vectorizer_loaded True olmalı"
    
    # Assert: Model listesi olmalı
    assert "models" in result, "health endpoint'inde models listesi olmalı"
    assert isinstance(result["models"], list), "models bir liste olmalı"
    assert len(result["models"]) == 3, "3 model olmalı"
    
    # Assert: Model isimleri doğru olmalı
    expected_models = ["Logistic Regression", "Naive Bayes", "Random Forest"]
    assert result["models"] == expected_models, \
        f"Model isimleri doğru olmalı (beklenen: {expected_models}, alınan: {result['models']})"
    
    # Test başarılı mesajı
    print(f"\n✅ Test Case 3: PASS - health endpoint başarıyla çalıştı")
    print(f"   Status: {result['status']}")
    print(f"   Models Loaded: {result['models_loaded']}")
    print(f"   Vectorizer Loaded: {result['vectorizer_loaded']}")
    print(f"   Models: {', '.join(result['models'])}")


if __name__ == "__main__":
    # Testleri çalıştır
    print("=" * 60)
    print("White-Box Test Cases - AI or Human Code Detector")
    print("=" * 60)
    print("\nTest Case 1: predict_code Fonksiyonu Testi")
    print("Test Case 2: get_prediction Fonksiyonu Testi")
    print("Test Case 3: health Endpoint Testi")
    print("\n" + "=" * 60)
    pytest.main([__file__, "-v", "--tb=short"])

