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




def plot_error_summary(metrics, title, output_path, color):
    """Her veri seti için hata dağılımı ve karışıklıkları gösterir."""
    y_true = metrics['y_true']
    errors = metrics['errors']
    abs_errors = metrics['abs_errors']
    years = sorted(list(set(y_true)))
    
    year_mae = []
    for year in years:
        mask = y_true == year
        year_mae.append(np.mean(np.abs(errors[mask])) if mask.sum() > 0 else 0)
    
    sorted_abs = np.sort(abs_errors)
    cum_ratio = np.linspace(0, 1, len(sorted_abs))
    top_confusions = get_top_confusions(metrics, top_n=6)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(years, year_mae, color=color, marker='o')
    axes[0, 0].fill_between(years, year_mae, color=color, alpha=0.15)
    axes[0, 0].set_title('Yıllara Göre Ortalama Mutlak Hata (MAE)')
    axes[0, 0].set_xlabel('Yıl')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_xticks(years[::2])
    axes[0, 0].grid(alpha=0.3)
    
    bins = np.arange(0, max(10, int(abs_errors.max()) + 2))
    axes[0, 1].hist(abs_errors, bins=bins, color=color, edgecolor='black', alpha=0.75)
    axes[0, 1].set_title('Mutlak Hata Dağılımı')
    axes[0, 1].set_xlabel('|Tahmin - Gerçek|')
    axes[0, 1].set_ylabel('Frekans')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    axes[1, 0].plot(sorted_abs, cum_ratio, color=color, linewidth=2)
    axes[1, 0].set_title('Kümülatif Mutlak Hata Eğrisi')
    axes[1, 0].set_xlabel('|Hata|')
    axes[1, 0].set_ylabel('Kümülatif Oran')
    axes[1, 0].grid(alpha=0.3)
    
    if top_confusions:
        pairs = [f"{item['actual']} → {item['predicted']}" for item in top_confusions]
        counts = [item['count'] for item in top_confusions]
        axes[1, 1].barh(pairs[::-1], counts[::-1], color=color, alpha=0.85)
        axes[1, 1].set_title('En Çok Karıştırılan Yıl Çiftleri')
        axes[1, 1].set_xlabel('Adet')
        axes[1, 1].grid(axis='x', alpha=0.2)
    else:
        axes[1, 1].text(0.5, 0.5, "Önemli karışıklık yok", ha='center', va='center')
        axes[1, 1].set_axis_off()
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dataset_error_overview(all_results, output_dir, embed_names):
    """Veri setleri arasında hata dağılımını karşılaştırır."""
    datasets = list(all_results.keys())
    mae_values = [all_results[ds]['metrics']['mae'] for ds in datasets]
    abs_error_lists = [all_results[ds]['metrics']['abs_errors'] for ds in datasets]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [DATASET_PALETTE[ds] for ds in datasets]
    
    axes[0].bar(
        [embed_names[ds] for ds in datasets],
        mae_values,
        color=colors,
        edgecolor='black',
        alpha=0.85
    )
    axes[0].set_title('Verisetlerine Göre MAE')
    axes[0].set_ylabel('MAE')
    axes[0].grid(axis='y', alpha=0.3)
    
    parts = axes[1].violinplot(abs_error_lists, showmeans=True, showmedians=True)
    for idx, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[idx])
        body.set_edgecolor('black')
        body.set_alpha(0.6)
    axes[1].set_xticks(np.arange(1, len(datasets)+1))
    axes[1].set_xticklabels([embed_names[ds] for ds in datasets])
    axes[1].set_title('Mutlak Hata Dağılımı (Violin)')
    axes[1].set_ylabel('|Tahmin - Gerçek|')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Verisetleri Hata Özeti', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_error_overview.png"), dpi=150, bbox_inches='tight')
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
    plot_comparison(all_results, output_dir)
    plot_dataset_error_overview(all_results, output_dir, embed_names)
    
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

