"""
Model Eğitimi Scripti
3 farklı ML modeli eğitir: Logistic Regression, Naive Bayes, Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys
import io
from datetime import datetime
from typing import List

# UTF-8 encoding için
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("MODEL EGITIMI BASLIYOR")
print("=" * 70)

# 1. Dataset yükleme
print("\n[1] Dataset yukleniyor...")
print("   [INFO] combined_dataset.csv kullaniliyor (ai_dataset + human_dataset + ai_extra birlesik)")
df = pd.read_csv('combined_dataset.csv')
print(f"   [OK] Toplam ornek: {len(df)}")
print(f"   [OK] Label dagilimi:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    print(f"      {label}: {count} ({percentage:.2f}%)")

# 2. Veri hazırlama
print("\n[2] Veri hazirlaniyor...")
X = df['text'].values
y = df['label'].values

# Train/Test split (stratified - label dağılımını korur)
print("\n[3] Train/Test split yapiliyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Label dağılımını korur
)

print(f"   [OK] Train set: {len(X_train)} ornek")
print(f"   [OK] Test set: {len(X_test)} ornek")
print(f"   [OK] Train label dagilimi:")
train_label_counts = pd.Series(y_train).value_counts()
for label, count in train_label_counts.items():
    percentage = (count / len(y_train)) * 100
    print(f"      {label}: {count} ({percentage:.2f}%)")

# 3. TF-IDF Vectorization
print("\n[4] TF-IDF Vectorization yapiliyor...")
print("   Parametreler:")
print("   - max_features: 20000 (en önemli 20000 kelime - trigram'lar için çok artırıldı)")
print("   - ngram_range: (1, 3) (unigram + bigram + trigram - repeated repeated repeated gibi pattern'leri yakalar)")
print("   - min_df: 1 (en az 1 dokümanda geçmeli - nadir AI pattern'lerini koru)")
print("   - max_df: 0.95 (en fazla %95 dokümanda geçebilir)")
print("   - sublinear_tf: True (log scaling)")

vectorizer = TfidfVectorizer(
    max_features=20000,  # Çok artırıldı - trigram'ların (repeated repeated repeated, memory_memory_memory gibi) vocabulary'de kalması için
    ngram_range=(1, 3),  # Unigram + bigram + trigram (repeated repeated repeated gibi pattern'leri yakalar)
    min_df=1,  # En az 1 dokümanda geçmeli (nadir AI pattern'lerini koru)
    max_df=0.95,  # En fazla %95 dokümanda geçebilir
    sublinear_tf=True,  # Log scaling
    stop_words='english'  # İngilizce stop words'leri kaldır
)

print("   [INFO] Vectorization islemi yapiliyor (bu biraz zaman alabilir)...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   [OK] Train shape: {X_train_tfidf.shape}")
print(f"   [OK] Test shape: {X_test_tfidf.shape}")
print(f"   [OK] Vocabulary size: {len(vectorizer.vocabulary_)}")

# Vectorizer'ı kaydet
print("\n[5] Vectorizer kaydediliyor...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("   [OK] tfidf_vectorizer.pkl kaydedildi")

# 4. Model eğitimi
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'  # Büyük dataset'ler için iyi
    ),
    'Naive Bayes': MultinomialNB(
        alpha=0.15,  # Laplace smoothing - dengeli değer: overconfidence'i azaltır, human kodlarına daha iyi davranır
        fit_prior=True  # Class prior'ları öğren
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Tüm CPU'ları kullan
    )
}

results = {}

print("\n" + "=" * 70)
print("MODEL EĞİTİMİ")
print("=" * 70)

for model_name, model in models.items():
    print(f"\n🤖 {model_name} eğitiliyor...")
    
    # Eğitim
    model.fit(X_train_tfidf, y_train)
    
    # Tahmin
    y_pred = model.predict(X_test_tfidf)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred
    }
    
    print(f"   ✓ Eğitim tamamlandı")
    print(f"   ✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Model kaydet
    filename = f"model_{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"   ✓ {filename} kaydedildi")

# 5. Detaylı değerlendirme
print("\n" + "=" * 70)
print("DETAYLI DEĞERLENDİRME")
print("=" * 70)

for model_name, result in results.items():
    print(f"\n📊 {model_name} - Detaylı Sonuçlar:")
    print("-" * 70)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, result['y_pred'], target_names=['AI', 'Human']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, result['y_pred'])
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              AI    Human")
    print(f"Actual AI    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"      Human  {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Precision, Recall, F1
    tn, fp, fn, tp = cm.ravel()
    precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_human = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_human = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAI - Precision: {precision_ai:.4f}, Recall: {recall_ai:.4f}")
    print(f"Human - Precision: {precision_human:.4f}, Recall: {recall_human:.4f}")

# 6. Özet
print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)
print("\n📊 Model Performansları:")
for model_name, result in results.items():
    print(f"   {model_name:20s}: {result['accuracy']*100:6.2f}%")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"\n🏆 En iyi model: {best_model_name} ({best_accuracy*100:.2f}%)")

# 7. Kaydedilen dosyalar
print("\n💾 Kaydedilen Dosyalar:")
print("   ✓ tfidf_vectorizer.pkl")
for model_name in models.keys():
    filename = f"model_{model_name.lower().replace(' ', '_')}.pkl"
    print(f"   ✓ {filename}")

print("\n" + "=" * 70)
print("✅ MODEL EĞİTİMİ TAMAMLANDI!")
print("=" * 70)
print(f"\n📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Model'ler backend'de kullanıma hazır!")

