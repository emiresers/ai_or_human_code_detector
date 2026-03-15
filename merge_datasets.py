"""
Dataset Birleştirme Scripti
Human ve AI dataset'lerini birleştirip tek bir CSV oluşturur.
"""

import pandas as pd
import os
import sys
import io

# UTF-8 encoding için
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("DATASET BIRLESTIRME")
print("=" * 60)

# Dataset'leri yükle
print("\n[1] Dataset'ler yukleniyor...")
df_human = pd.read_csv('human_dataset.csv')
df_ai = pd.read_csv('ai_dataset.csv')

print(f"   [OK] Human dataset: {len(df_human)} ornek")
print(f"   [OK] AI dataset: {len(df_ai)} ornek")

# Ek AI pattern'leri (ai_extra.csv) varsa ekle
ai_extra_path = 'ai_extra.csv'
if os.path.exists(ai_extra_path):
    print(f"\n[1b] Ek AI pattern'leri ekleniyor: {ai_extra_path}")
    # ai_extra.csv özel formatını oku
    with open(ai_extra_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if raw.startswith("text,label"):
        raw = raw[len("text,label"):]
    chunks = raw.split("\",ai")
    extra_texts = []
    for chunk in chunks:
        cleaned = chunk.lstrip().lstrip(",")
        if cleaned.startswith("\""):
            cleaned = cleaned[1:]
        cleaned = cleaned.lstrip("\n")
        if cleaned.strip():
            extra_texts.append(cleaned)
    if extra_texts:
        df_extra = pd.DataFrame({"text": extra_texts, "label": "ai"})
        print(f"   [OK] Ek AI pattern'leri: {len(df_extra)} ornek")
        df_ai = pd.concat([df_ai, df_extra], ignore_index=True)
        print(f"   [OK] Yeni AI dataset toplam: {len(df_ai)} ornek")

# Kontroller
print("\n[2] Kontroller yapiliyor...")

# 1. Kolon kontrolü
required_columns = ['text', 'label']
if not all(col in df_human.columns for col in required_columns):
    raise ValueError("Human dataset'te gerekli kolonlar yok!")
if not all(col in df_ai.columns for col in required_columns):
    raise ValueError("AI dataset'te gerekli kolonlar yok!")
print("   [OK] Kolonlar dogru")

# 2. Label kontrolü
human_labels = df_human['label'].unique()
ai_labels = df_ai['label'].unique()
print(f"   [OK] Human label'lari: {human_labels}")
print(f"   [OK] AI label'lari: {ai_labels}")

if 'human' not in human_labels:
    print("   [UYARI] Human dataset'te 'human' label'i yok!")
if 'ai' not in ai_labels:
    print("   [UYARI] AI dataset'te 'ai' label'i yok!")

# 3. Boş değer kontrolü
human_empty = df_human['text'].isna().sum()
ai_empty = df_ai['text'].isna().sum()
if human_empty > 0:
    print(f"   [UYARI] Human dataset'te {human_empty} bos deger var!")
    df_human = df_human.dropna(subset=['text'])
if ai_empty > 0:
    print(f"   [UYARI] AI dataset'te {ai_empty} bos deger var!")
    df_ai = df_ai.dropna(subset=['text'])

# 4. Çok kısa kod kontrolü (30 karakterden kısa)
human_short = (df_human['text'].str.len() < 30).sum()
ai_short = (df_ai['text'].str.len() < 30).sum()
if human_short > 0:
    print(f"   [UYARI] Human dataset'te {human_short} cok kisa kod var (<30 karakter)")
if ai_short > 0:
    print(f"   [UYARI] AI dataset'te {ai_short} cok kisa kod var (<30 karakter)")

# Birleştir
print("\n[3] Dataset'ler birlestiriliyor...")
df_combined = pd.concat([df_human, df_ai], ignore_index=True)
print(f"   [OK] Toplam: {len(df_combined)} ornek")

# Duplikat kontrolü (sadece bilgi amaçlı)
print("\n[4] Duplikat kontrolu yapiliyor...")
duplicates = df_combined.duplicated(subset=['text']).sum()
if duplicates > 0:
    print(f"   [INFO] {duplicates} duplikat bulundu (normal - AI dataset'te template tekrarlari var)")
    print(f"   [INFO] Duplikatlar korunuyor (model egitimi icin faydali)")
else:
    print("   [OK] Duplikat yok")

# Karıştır (shuffle)
print("\n[5] Dataset karistiriliyor...")
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print("   [OK] Karistirildi")

# Label dağılımı
print("\n[6] Label dagilimi:")
label_counts = df_combined['label'].value_counts()
for label, count in label_counts.items():
    percentage = (count / len(df_combined)) * 100
    print(f"   {label}: {count} örnek ({percentage:.2f}%)")

# Kaydet
output_file = 'combined_dataset.csv'
print(f"\n[7] Birlestirilmis dataset kaydediliyor: {output_file}")
df_combined.to_csv(output_file, index=False)
print(f"   [OK] Kaydedildi!")

# Dosya boyutu kontrolü
file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
print(f"   [OK] Dosya boyutu: {file_size:.2f} MB")

# Son kontrol
print("\n[TAMAMLANDI] SON KONTROL:")
print(f"   Toplam ornek: {len(df_combined)}")
print(f"   Human: {label_counts.get('human', 0)}")
print(f"   AI: {label_counts.get('ai', 0)}")
print(f"   Kolonlar: {list(df_combined.columns)}")
print(f"   Ilk 3 ornek label: {df_combined['label'].head(3).tolist()}")

print("\n" + "=" * 60)
print("[TAMAMLANDI] BIRLESTIRME TAMAMLANDI!")
print("=" * 60)
print(f"\n[INFO] Dosya: {output_file}")
print(f"[INFO] Toplam: {len(df_combined)} ornek")
print(f"[INFO] Model egitimi icin hazir!")

