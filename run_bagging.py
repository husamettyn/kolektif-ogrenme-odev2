"""
Bagging Model Eğitim ve Değerlendirme Scripti

Bu script:
1. 3 farklı embedding verisetini yükler (başlık, özet, birleştirilmiş)
2. Her veriseti için Bagging modeli eğitir
3. Test setinde performans değerlendirmesi yapar
4. Sonuçları ve grafikleri artifacts/ klasörüne kaydeder
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from datetime import datetime
import json
import joblib

from plot_utils import (
    plot_confusion_matrix,
    plot_year_accuracy,
    plot_prediction_distribution,
    plot_error_summary,
    plot_dataset_error_overview,
    plot_comparison,
)

# Sabit parametreler
RANDOM_SEED = 42
# Optimizasyon sonuçlarına göre veri setine özel parametreler
# Tüm veri setleri için: max_samples=1.0, n_estimators=200
N_ESTIMATORS = 200      # Temel öğrenici sayısı (optimizasyon sonucu)
MAX_SAMPLES = 1.0       # Her öğrenici için kullanılacak örnek oranı
MAX_FEATURES = 1.0      # Her öğrenici için kullanılacak özellik oranı
BOOTSTRAP = True        # Bootstrap örnekleme
N_JOBS = -1             # Tüm CPU çekirdeklerini kullan

# Dizinler
DATASET_DIR = "dataset"
ARTIFACTS_DIR = "artifacts"

DATASET_PALETTE = {
    'title': '#2e8b57',
    'abstract': '#1f77b4',
    'concat': '#8e44ad'
}


def load_embedding_data(embed_type):
    """
    Embedding verisetini yükler.
    
    Args:
        embed_type: Embedding türü ('title', 'abstract', 'concat')
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    train_file = os.path.join(DATASET_DIR, f"train_{embed_type}_embeddings.npz")
    test_file = os.path.join(DATASET_DIR, f"test_{embed_type}_embeddings.npz")
    
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    
    return (
        train_data['embeddings'],
        train_data['years'],
        test_data['embeddings'],
        test_data['years']
    )


def train_model(X_train, y_train):
    """
    Bagging modeli eğitir.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
    
    Returns:
        BaggingClassifier: Eğitilmiş model
    """
    # Temel öğrenici olarak Decision Tree kullan
    base_estimator = DecisionTreeClassifier(random_state=RANDOM_SEED)
    
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Model performansını değerlendirir.
    
    Args:
        model: Eğitilmiş model
        X_test: Test özellikleri
        y_test: Test etiketleri
    
    Returns:
        dict: Performans metrikleri
    """
    y_pred = model.predict(X_test)
    errors = y_pred - y_test
    abs_errors = np.abs(errors)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_true': y_test,
        'y_pred': y_pred,
        'errors': errors,
        'abs_errors': abs_errors
    }
    
    return metrics


def save_results(all_results, output_dir):
    """
    Tüm sonuçları kaydeder.
    
    Args:
        all_results: Tüm model sonuçları
        output_dir: Çıktı dizini
    """
    # Özet metrikleri kaydet
    summary = {}
    for embed_type, results in all_results.items():
        summary[embed_type] = {
            'accuracy': results['metrics']['accuracy'],
            'f1_macro': results['metrics']['f1_macro'],
            'f1_weighted': results['metrics']['f1_weighted'],
            'precision_macro': results['metrics']['precision_macro'],
            'recall_macro': results['metrics']['recall_macro'],
            'mae': results['metrics']['mae'],
            'rmse': results['metrics']['rmse']
        }
    
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Karşılaştırma tablosu oluştur
    comparison_text = """# Bagging Sonuçları

## Model Parametreleri

| Parametre | Değer |
|-----------|-------|
| n_estimators | {} |
| max_samples | {} |
| max_features | {} |
| bootstrap | {} |
| base_estimator | DecisionTreeClassifier |
| random_state | {} |

## Performans Karşılaştırması

| Veriseti | Accuracy | F1 (Macro) | F1 (Weighted) | Precision | Recall | MAE | RMSE |
|----------|----------|------------|---------------|-----------|--------|-----|------|
""".format(N_ESTIMATORS, MAX_SAMPLES, MAX_FEATURES, BOOTSTRAP, RANDOM_SEED)
    
    for embed_type, metrics in summary.items():
        comparison_text += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.2f} |\n".format(
            embed_type,
            metrics['accuracy'],
            metrics['f1_macro'],
            metrics['f1_weighted'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['mae'],
            metrics['rmse']
        )
    
    # Her veriseti için detaylı rapor
    for embed_type, results in all_results.items():
        comparison_text += f"\n## {embed_type.capitalize()} Veriseti Detaylı Rapor\n\n```\n"
        comparison_text += results['metrics']['classification_report']
        comparison_text += "\n```\n"
    
    with open(os.path.join(output_dir, "results_report.md"), "w", encoding="utf-8") as f:
        f.write(comparison_text)


def main():
    """
    Ana fonksiyon - Bagging deney sürecini yönetir.
    """
    print("=" * 60)
    print("BAGGING MODEL EĞİTİMİ VE DEĞERLENDİRMESİ")
    print("=" * 60)
    
    # Çıktı dizini oluştur (timestamp ile)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(ARTIFACTS_DIR, f"{timestamp}_bagging")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nÇıktı dizini: {output_dir}")
    
    # Embedding türleri
    embed_types = ['title', 'abstract', 'concat']
    embed_names = {
        'title': 'Başlık',
        'abstract': 'Özet', 
        'concat': 'Birleştirilmiş'
    }
    
    all_results = {}
    
    for embed_type in embed_types:
        print(f"\n{'='*60}")
        print(f"{embed_names[embed_type].upper()} VERİSETİ")
        print(f"{'='*60}")
        
        # Veriyi yükle
        print("\n1. Veri yükleniyor...")
        X_train, y_train, X_test, y_test = load_embedding_data(embed_type)
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Model eğit
        print("\n2. Model eğitiliyor...")
        model = train_model(X_train, y_train)
        
        # Değerlendir
        print("\n3. Model değerlendiriliyor...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 (Macro): {metrics['f1_macro']:.4f}")
        
        # Grafikleri oluştur
        print("\n4. Grafikler oluşturuluyor...")
        
        plot_confusion_matrix(
            metrics, 
            f"Bagging - {embed_names[embed_type]}",
            os.path.join(output_dir, f"{embed_type}_confusion_matrix.png")
        )
        
        plot_year_accuracy(
            metrics,
            f"Bagging - {embed_names[embed_type]}",
            os.path.join(output_dir, f"{embed_type}_year_accuracy.png")
        )
        
        plot_prediction_distribution(
            metrics,
            f"Bagging - {embed_names[embed_type]}",
            os.path.join(output_dir, f"{embed_type}_prediction_dist.png")
        )
        
        plot_error_summary(
            metrics,
            f"Bagging - {embed_names[embed_type]} Detaylı Hata Profili",
            os.path.join(output_dir, f"{embed_type}_error_summary.png"),
            DATASET_PALETTE[embed_type]
        )
        
        # Modeli kaydet
        model_path = os.path.join(output_dir, f"{embed_type}_model.joblib")
        joblib.dump(model, model_path)
        print(f"   Model kaydedildi: {model_path}")
        
        all_results[embed_type] = {
            'metrics': metrics,
            'model_path': model_path
        }
    
    # Sonuçları kaydet
    print(f"\n{'='*60}")
    print("SONUÇLAR KAYDEDİLİYOR")
    print(f"{'='*60}")
    
    save_results(all_results, output_dir)
    plot_comparison(all_results, output_dir, embed_names, "Bagging")
    plot_dataset_error_overview(all_results, output_dir, embed_names, DATASET_PALETTE)
    
    print(f"\n✓ Tüm sonuçlar kaydedildi: {output_dir}/")
    
    # Özet tablo
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    print("\n{:<15} {:>10} {:>10} {:>10} {:>8}".format(
        "Veriseti", "Accuracy", "F1-Macro", "F1-Weight", "MAE"
    ))
    print("-" * 60)
    for embed_type in embed_types:
        m = all_results[embed_type]['metrics']
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.2f}".format(
            embed_names[embed_type],
            m['accuracy'],
            m['f1_macro'],
            m['f1_weighted'],
            m['mae']
        ))


if __name__ == "__main__":
    main()

