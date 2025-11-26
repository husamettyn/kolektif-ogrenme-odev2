"""
Embedding Oluşturma Scripti

Bu script:
1. turkish-e5-large modelini yükler
2. Verisetindeki başlık ve özetler için embedding oluşturur
3. 6 farklı veriseti oluşturur:
   - Train: başlık_embed, özet_embed, concat(başlık_embed, özet_embed)
   - Test: başlık_embed, özet_embed, concat(başlık_embed, özet_embed)
4. Sonuçları dataset/ klasörüne kaydeder
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm

# Cihaz seçimi (GPU varsa kullan)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model bilgisi
MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"

# Giriş/çıkış dizinleri
INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset"

# Batch boyutu (GPU belleğine göre ayarlanabilir)
BATCH_SIZE = 16


def load_model():
    """
    Turkish E5 Large modelini ve tokenizer'ı yükler.
    
    Returns:
        tuple: (tokenizer, model)
    """
    print(f"Model yükleniyor: {MODEL_NAME}")
    print(f"Cihaz: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()  # Değerlendirme moduna al
    
    print("Model yüklendi.")
    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    """
    Token embeddinglerinden cümle embedding'i oluşturur (mean pooling).
    
    Args:
        model_output: Model çıktısı
        attention_mask: Dikkat maskesi
    
    Returns:
        torch.Tensor: Ortalama pooling sonucu
    """
    token_embeddings = model_output[0]  # İlk eleman token embeddingleri
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_embeddings(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """
    Verilen metinler için embedding oluşturur.
    
    Args:
        texts: Embedding oluşturulacak metinler
        tokenizer: Tokenizer
        model: Model
        batch_size: Batch boyutu
    
    Returns:
        np.ndarray: Embedding matrisi (n_samples, embedding_dim)
    """
    all_embeddings = []
    
    # Batch'ler halinde işle
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding oluşturuluyor"):
        batch_texts = texts[i:i + batch_size]
        
        # E5 modeli için "query: " prefix'i ekle
        batch_texts = ["query: " + text for text in batch_texts]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        
        # Embedding oluştur
        with torch.no_grad():
            model_output = model(**encoded)
            embeddings = mean_pooling(model_output, encoded['attention_mask'])
            
            # Normalize et
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def process_dataset(split_name, data, tokenizer, model):
    """
    Bir veriseti split'i için tüm embeddingleri oluşturur.
    
    Args:
        split_name: Split adı (train/test)
        data: Veriseti
        tokenizer: Tokenizer
        model: Model
    
    Returns:
        dict: Embedding verisetleri
    """
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} VERİSETİ İŞLENİYOR")
    print(f"{'='*60}")
    
    # Başlıkları al (Türkçe öncelikli, yoksa İngilizce)
    titles = []
    for item in data:
        title = item['title_tr'] if item['title_tr'] else item['title_en']
        titles.append(title if title else "")
    
    # Özetleri al (Türkçe öncelikli, yoksa İngilizce)
    abstracts = []
    for item in data:
        abstract = item['abstract_tr'] if item['abstract_tr'] else item['abstract_en']
        abstracts.append(abstract if abstract else "")
    
    # Yılları al
    years = [item['year'] for item in data]
    
    print(f"Toplam kayıt: {len(titles)}")
    
    # Başlık embeddinglari
    print("\n1. Başlık embeddinglari oluşturuluyor...")
    title_embeddings = create_embeddings(titles, tokenizer, model)
    print(f"   Boyut: {title_embeddings.shape}")
    
    # Özet embeddinglari
    print("\n2. Özet embeddinglari oluşturuluyor...")
    abstract_embeddings = create_embeddings(abstracts, tokenizer, model)
    print(f"   Boyut: {abstract_embeddings.shape}")
    
    # Birleştirilmiş embeddingler (başlık + özet)
    print("\n3. Birleştirilmiş embeddingler oluşturuluyor...")
    concat_embeddings = np.concatenate([title_embeddings, abstract_embeddings], axis=1)
    print(f"   Boyut: {concat_embeddings.shape}")
    
    return {
        'title': title_embeddings,
        'abstract': abstract_embeddings,
        'concat': concat_embeddings,
        'years': np.array(years)
    }


def save_embeddings(train_embeddings, test_embeddings, output_dir):
    """
    Embedding verisetlerini kaydeder.
    
    Args:
        train_embeddings: Eğitim embedding'leri
        test_embeddings: Test embedding'leri
        output_dir: Çıktı dizini
    """
    print(f"\n{'='*60}")
    print("EMBEDDİNGLER KAYDEDİLİYOR")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 6 farklı veriseti kaydet
    datasets_info = [
        ('title', 'Başlık embedding'),
        ('abstract', 'Özet embedding'),
        ('concat', 'Birleştirilmiş embedding (başlık + özet)')
    ]
    
    for embed_type, description in datasets_info:
        # Train verisi
        train_file = os.path.join(output_dir, f"train_{embed_type}_embeddings.npz")
        np.savez_compressed(
            train_file,
            embeddings=train_embeddings[embed_type],
            years=train_embeddings['years']
        )
        print(f"✓ {train_file}")
        print(f"  - {description}")
        print(f"  - Embedding boyutu: {train_embeddings[embed_type].shape}")
        
        # Test verisi
        test_file = os.path.join(output_dir, f"test_{embed_type}_embeddings.npz")
        np.savez_compressed(
            test_file,
            embeddings=test_embeddings[embed_type],
            years=test_embeddings['years']
        )
        print(f"✓ {test_file}")
        print(f"  - {description}")
        print(f"  - Embedding boyutu: {test_embeddings[embed_type].shape}")
        print()


def generate_embeddings_info(train_embeddings, test_embeddings, output_dir):
    """
    Embedding bilgi dosyası oluşturur.
    
    Args:
        train_embeddings: Eğitim embedding'leri
        test_embeddings: Test embedding'leri
        output_dir: Çıktı dizini
    """
    info_text = f"""# Embedding Bilgileri

## Model Bilgisi

- **Model**: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})
- **Embedding Boyutu**: {train_embeddings['title'].shape[1]}
- **Birleştirilmiş Embedding Boyutu**: {train_embeddings['concat'].shape[1]}

## Oluşturulan Verisetleri

| Dosya | Açıklama | Boyut |
|-------|----------|-------|
| train_title_embeddings.npz | Eğitim - Başlık embeddinglari | {train_embeddings['title'].shape} |
| train_abstract_embeddings.npz | Eğitim - Özet embeddinglari | {train_embeddings['abstract'].shape} |
| train_concat_embeddings.npz | Eğitim - Birleştirilmiş embeddingler | {train_embeddings['concat'].shape} |
| test_title_embeddings.npz | Test - Başlık embeddinglari | {test_embeddings['title'].shape} |
| test_abstract_embeddings.npz | Test - Özet embeddinglari | {test_embeddings['abstract'].shape} |
| test_concat_embeddings.npz | Test - Birleştirilmiş embeddingler | {test_embeddings['concat'].shape} |

## Kullanım

```python
import numpy as np

# Veriyi yükle
data = np.load('dataset/train_title_embeddings.npz')
X_train = data['embeddings']  # Embedding matrisi
y_train = data['years']       # Yıl etiketleri

print(f"X shape: {{X_train.shape}}")
print(f"y shape: {{y_train.shape}}")
```

## Notlar

- Tüm embeddingler L2 normalize edilmiştir.
- E5 modeli için "query: " prefix'i kullanılmıştır.
- Boş başlık/özet durumunda boş string için embedding oluşturulmuştur.
- Türkçe metin yoksa İngilizce metin kullanılmıştır.
"""
    
    with open(os.path.join(output_dir, "EMBEDDINGS_INFO.md"), "w", encoding="utf-8") as f:
        f.write(info_text)
    
    print(f"✓ {output_dir}/EMBEDDINGS_INFO.md")


def main():
    """
    Ana fonksiyon - embedding oluşturma sürecini yönetir.
    """
    print("=" * 60)
    print("EMBEDDİNG OLUŞTURMA")
    print("=" * 60)
    
    # 1. Verisetini yükle
    print("\nVeriseti yükleniyor...")
    dataset = load_from_disk(os.path.join(INPUT_DIR, "thesis_dataset"))
    print(f"Train: {len(dataset['train'])} kayıt")
    print(f"Test: {len(dataset['test'])} kayıt")
    
    # 2. Modeli yükle
    tokenizer, model = load_model()
    
    # 3. Train embeddinglari oluştur
    train_embeddings = process_dataset("train", dataset['train'], tokenizer, model)
    
    # 4. Test embeddinglari oluştur
    test_embeddings = process_dataset("test", dataset['test'], tokenizer, model)
    
    # 5. Kaydet
    save_embeddings(train_embeddings, test_embeddings, OUTPUT_DIR)
    
    # 6. Bilgi dosyası oluştur
    generate_embeddings_info(train_embeddings, test_embeddings, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("EMBEDDİNG OLUŞTURMA TAMAMLANDI")
    print("=" * 60)
    
    total_embeddings = (
        train_embeddings['title'].shape[0] + 
        train_embeddings['abstract'].shape[0] +
        test_embeddings['title'].shape[0] + 
        test_embeddings['abstract'].shape[0]
    )
    print(f"\nÖzet:")
    print(f"  - Toplam embedding sayısı: {total_embeddings}")
    print(f"  - Embedding boyutu: {train_embeddings['title'].shape[1]}")
    print(f"  - Birleştirilmiş embedding boyutu: {train_embeddings['concat'].shape[1]}")


if __name__ == "__main__":
    main()

