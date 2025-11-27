# Kolektif Öğrenme Ödev 2

## Tez Yılı Tahmininde Kolektif Öğrenme

Bu proje, Türkçe akademik tez başlıkları ve özetlerinden yıl tahmini yapmak için kolektif öğrenme yöntemlerini kullanmaktadır.

---

## Kurulum ve Çalıştırma Talimatları

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Verisetini Hazırla

Huggingface'den verisetini indirir, 2001-2025 yılları için her yıldan 500 tez seçer (250 train, 250 test):

```bash
python prepare_dataset.py
```

**Çıktılar:**
- `dataset/thesis_dataset/` - Huggingface Dataset formatında veriseti
- `dataset/train.csv`, `dataset/test.csv` - CSV formatında verisetleri
- `dataset/DATASET_INFO.md` - Veriseti hakkında bilgiler

### 3. Embedding Oluştur

Turkish E5 Large modeli ile başlık ve özet embeddinglerini oluşturur:

```bash
python create_embeddings.py
```

**Not:** Bu adım GPU ile çok daha hızlı çalışır. CPU'da uzun sürebilir.

**Çıktılar:**
- `dataset/train_title_embeddings.npz` - Eğitim başlık embeddinglari
- `dataset/train_abstract_embeddings.npz` - Eğitim özet embeddinglari
- `dataset/train_concat_embeddings.npz` - Eğitim birleştirilmiş embeddingler
- `dataset/test_title_embeddings.npz` - Test başlık embeddinglari
- `dataset/test_abstract_embeddings.npz` - Test özet embeddinglari
- `dataset/test_concat_embeddings.npz` - Test birleştirilmiş embeddingler
- `dataset/EMBEDDINGS_INFO.md` - Embedding bilgileri

### 4. Model Eğitimi ve Değerlendirme

Her algoritma için ayrı ayrı çalıştırın:

```bash
# Random Forest
python run_random_forest.py

# Bagging
python run_bagging.py

# Random Subspace
python run_random_subspace.py
```

**Çıktılar:** Her çalıştırma için `artifacts/TIMESTAMP_ALGORITHM/` klasörü altında:
- `comparison.png` - Veriseti karşılaştırma grafiği
- `title_confusion_matrix.png`, `abstract_confusion_matrix.png`, `concat_confusion_matrix.png`
- `title_year_accuracy.png`, `abstract_year_accuracy.png`, `concat_year_accuracy.png`
- `title_prediction_dist.png`, `abstract_prediction_dist.png`, `concat_prediction_dist.png`
- `title_error_summary.png`, `abstract_error_summary.png`, `concat_error_summary.png` - Yıl bazlı MAE, hata histogramı ve en çok karıştırılan yılların yer aldığı detaylı dashboard
- `dataset_error_overview.png` - Tüm verisetleri için MAE bar grafiği ve mutlak hata violin grafiği
- `metrics_summary.json` - Accuracy/F1/Precision/Recall ile birlikte MAE ve RMSE değerleri
- `results_report.md` - Detaylı sonuç raporu
- Model dosyaları (`.joblib`)

### 5. Hiperparametre Optimizasyonu

Her algoritma için en iyi hiperparametreleri bulmak için:

```bash
python optimize_hyperparameters.py

# Daha hızlı denemeler için:
python optimize_hyperparameters.py --mode quick --datasets title abstract --methods random_forest
```

**Parametreler:**
- `--mode {full,quick}`: `quick` seçilirse daha dar bir grid ve (varsayılan olarak) 2-fold CV kullanılır.
- `--datasets title abstract concat`: Üzerinde arama yapılacak embedding setlerini seçer.
- `--methods random_forest bagging random_subspace`: Çalıştırılacak algoritmaları belirler.
- `--cv-folds N`: Fold sayısını manuel olarak değiştirir (quick modda varsayılan 2'dir).

**Çıktılar:** `artifacts/TIMESTAMP_optimization/` klasörü altında:
- `optimization_results.json` - Optimizasyon sonuçları
- `optimization_report.md` - Detaylı optimizasyon raporu

---

## Proje Yapısı

```
.
├── README.md                      # Bu dosya
├── ODEV.md                        # Ödev gereksinimleri
├── requirements.txt               # Python bağımlılıkları
├── prepare_dataset.py             # Veriseti hazırlama scripti
├── create_embeddings.py           # Embedding oluşturma scripti
├── run_random_forest.py           # Random Forest eğitim scripti
├── run_bagging.py                 # Bagging eğitim scripti
├── run_random_subspace.py         # Random Subspace eğitim scripti
├── optimize_hyperparameters.py    # Hiperparametre optimizasyon scripti
├── dataset/                       # Veriseti klasörü
│   ├── thesis_dataset/            # Huggingface Dataset
│   ├── train.csv, test.csv        # CSV formatında veriler
│   ├── *_embeddings.npz           # Embedding dosyaları
│   └── *.md                       # Bilgi dosyaları
└── artifacts/                     # Sonuç klasörü
    └── TIMESTAMP_ALGORITHM/       # Her deney için ayrı klasör
        ├── *.png                  # Grafikler
        ├── *.json                 # Metrikler
        ├── *.md                   # Raporlar
        └── *.joblib               # Modeller
```

---

## Algoritma Açıklamaları

### Random Forest
- Birden fazla karar ağacının kombinasyonu
- Her ağaç, veri ve özelliklerin rastgele alt kümelerinde eğitilir
- Optimize edilen parametreler: `n_estimators`, `max_depth`

### Bagging (Bootstrap Aggregating)
- Temel öğrenicilerin bootstrap örneklerinde eğitilmesi
- Her öğrenici aynı algoritma (Decision Tree)
- Optimize edilen parametreler: `n_estimators`, `max_samples`

### Random Subspace
- Temel öğrenicilerin rastgele özellik alt kümelerinde eğitilmesi
- Bootstrap örnekleme yapılmaz, sadece özellik seçimi
- Optimize edilen parametreler: `n_estimators`, `max_features`

---

## Veriseti Bilgileri

- **Kaynak**: [turkish-academic-theses-dataset](https://huggingface.co/datasets/umutertugrul/turkish-academic-theses-dataset)
- **Yıl Aralığı**: 2001-2025 (25 yıl)
- **Her yıl için**: 250 eğitim + 250 test = 500 tez
- **Toplam**: 6250 eğitim + 6250 test = 12500 tez
- **Embedding Modeli**: [turkish-e5-large](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large)
- **Embedding Boyutu**: 1024

---

## 6 Veriseti

| # | Veriseti | Açıklama | Boyut |
|---|----------|----------|-------|
| 1 | train_title | Eğitim - Başlık embeddinglari | (6250, 1024) |
| 2 | train_abstract | Eğitim - Özet embeddinglari | (6250, 1024) |
| 3 | train_concat | Eğitim - Birleştirilmiş | (6250, 2048) |
| 4 | test_title | Test - Başlık embeddinglari | (6250, 1024) |
| 5 | test_abstract | Test - Özet embeddinglari | (6250, 1024) |
| 6 | test_concat | Test - Birleştirilmiş | (6250, 2048) |

---

# TODO (Tamamlananlar)
* [x] ODEV.md dosyasını oku
* [x] Dosya yapısı oluşturuldu (dataset, artifacts klasörleri)
* [x] Veriseti indirme ve filtreleme scripti (prepare_dataset.py)
* [x] Embedding oluşturma scripti (create_embeddings.py)
* [x] Random Forest run scripti (run_random_forest.py)
* [x] Bagging run scripti (run_bagging.py)
* [x] Random Subspace run scripti (run_random_subspace.py)
* [x] Hiperparametre optimizasyonu scripti (optimize_hyperparameters.py)
* [x] requirements.txt ve README talimatları
