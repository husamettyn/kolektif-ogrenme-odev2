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
import seaborn as sns
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

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

# Sabit parametreler
RANDOM_SEED = 42
N_ESTIMATORS = 100      # Temel öğrenici sayısı
MAX_SAMPLES = 1.0       # Her öğrenici için kullanılacak örnek oranı
MAX_FEATURES = 1.0      # Her öğrenici için kullanılacak özellik oranı
BOOTSTRAP = True        # Bootstrap örnekleme
N_JOBS = -1             # Tüm CPU çekirdeklerini kullan

# Dizinler
DATASET_DIR = "dataset"
ARTIFACTS_DIR = "artifacts"


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
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_true': y_test,
        'y_pred': y_pred
    }
    
    return metrics


def plot_confusion_matrix(metrics, title, output_path):
    """
    Confusion matrix grafiği oluşturur.
    
    Args:
        metrics: Performans metrikleri
        title: Grafik başlığı
        output_path: Kayıt yolu
    """
    cm = metrics['confusion_matrix']
    labels = sorted(list(set(metrics['y_true'])))
    
    plt.figure(figsize=(16, 14))
    
    # Normalize edilmiş confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, 
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.title(f'{title}\nNormalize Edilmiş Confusion Matrix', fontsize=14)
    plt.xlabel('Tahmin Edilen Yıl', fontsize=12)
    plt.ylabel('Gerçek Yıl', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_year_accuracy(metrics, title, output_path):
    """
    Yıllara göre doğruluk grafiği oluşturur.
    
    Args:
        metrics: Performans metrikleri
        title: Grafik başlığı
        output_path: Kayıt yolu
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    years = sorted(list(set(y_true)))
    accuracies = []
    
    for year in years:
        mask = y_true == year
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(years, accuracies, color='forestgreen', edgecolor='darkgreen')
    
    # Ortalama çizgisi
    avg_acc = np.mean(accuracies)
    plt.axhline(y=avg_acc, color='red', linestyle='--', label=f'Ortalama: {avg_acc:.3f}')
    
    plt.title(f'{title}\nYıllara Göre Doğruluk', fontsize=14)
    plt.xlabel('Yıl', fontsize=12)
    plt.ylabel('Doğruluk', fontsize=12)
    plt.xticks(years, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_distribution(metrics, title, output_path):
    """
    Tahmin dağılımı grafiği oluşturur.
    
    Args:
        metrics: Performans metrikleri
        title: Grafik başlığı
        output_path: Kayıt yolu
    """
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    # Hata analizi
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hata dağılımı
    axes[0].hist(errors, bins=50, color='forestgreen', edgecolor='darkgreen', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Tahmin Hatası Dağılımı', fontsize=12)
    axes[0].set_xlabel('Hata (Tahmin - Gerçek)', fontsize=10)
    axes[0].set_ylabel('Frekans', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # MAE ve RMSE
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    axes[0].text(0.95, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                 transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Scatter plot: Gerçek vs Tahmin
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10, c='forestgreen')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', linewidth=2, label='İdeal')
    axes[1].set_title('Gerçek vs Tahmin', fontsize=12)
    axes[1].set_xlabel('Gerçek Yıl', fontsize=10)
    axes[1].set_ylabel('Tahmin Edilen Yıl', fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


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
            'recall_macro': results['metrics']['recall_macro']
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

| Veriseti | Accuracy | F1 (Macro) | F1 (Weighted) | Precision | Recall |
|----------|----------|------------|---------------|-----------|--------|
""".format(N_ESTIMATORS, MAX_SAMPLES, MAX_FEATURES, BOOTSTRAP, RANDOM_SEED)
    
    for embed_type, metrics in summary.items():
        comparison_text += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            embed_type,
            metrics['accuracy'],
            metrics['f1_macro'],
            metrics['f1_weighted'],
            metrics['precision_macro'],
            metrics['recall_macro']
        )
    
    # Her veriseti için detaylı rapor
    for embed_type, results in all_results.items():
        comparison_text += f"\n## {embed_type.capitalize()} Veriseti Detaylı Rapor\n\n```\n"
        comparison_text += results['metrics']['classification_report']
        comparison_text += "\n```\n"
    
    with open(os.path.join(output_dir, "results_report.md"), "w", encoding="utf-8") as f:
        f.write(comparison_text)


def plot_comparison(all_results, output_dir):
    """
    Tüm modellerin karşılaştırma grafiğini oluşturur.
    
    Args:
        all_results: Tüm model sonuçları
        output_dir: Çıktı dizini
    """
    embed_types = list(all_results.keys())
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    metrics_labels = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall']
    
    x = np.arange(len(embed_types))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for i, (metric_name, metric_label, color) in enumerate(zip(metrics_names, metrics_labels, colors)):
        values = [all_results[et]['metrics'][metric_name] for et in embed_types]
        bars = ax.bar(x + i * width, values, width, label=metric_label, color=color)
        
        # Değerleri bar'ların üstüne yaz
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Veriseti', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Bagging - Veriseti Karşılaştırması', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(['Başlık', 'Özet', 'Birleştirilmiş'])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


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
    plot_comparison(all_results, output_dir)
    
    print(f"\n✓ Tüm sonuçlar kaydedildi: {output_dir}/")
    
    # Özet tablo
    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    print("\n{:<15} {:>10} {:>10} {:>10}".format("Veriseti", "Accuracy", "F1-Macro", "F1-Weight"))
    print("-" * 50)
    for embed_type in embed_types:
        m = all_results[embed_type]['metrics']
        print("{:<15} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            embed_names[embed_type],
            m['accuracy'],
            m['f1_macro'],
            m['f1_weighted']
        ))


if __name__ == "__main__":
    main()

