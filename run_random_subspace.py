"""
Random Subspace Model Eğitim ve Değerlendirme Scripti

Bu script:
1. 3 farklı embedding verisetini yükler (başlık, özet, birleştirilmiş)
2. Her veriseti için Random Subspace modeli eğitir
3. Test setinde performans değerlendirmesi yapar
4. Sonuçları ve grafikleri artifacts/ klasörüne kaydeder

Not: Random Subspace, Bagging'in max_features < 1.0 ve bootstrap=False versiyonudur.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from datetime import datetime
import json
import joblib

plt.rcParams['font.family'] = 'DejaVu Sans'

# Sabit parametreler - Random Subspace için özellik alt kümesi seçimi
RANDOM_SEED = 42
N_ESTIMATORS = 100
MAX_SAMPLES = 1.0       # Tüm örnekleri kullan
MAX_FEATURES = 0.5      # Özelliklerin %50'sini rastgele seç
BOOTSTRAP = False       # Örnekleme yapma (sadece özellik seçimi)
BOOTSTRAP_FEATURES = True  # Özellik alt kümesi seçimi
N_JOBS = -1

DATASET_DIR = "dataset"
ARTIFACTS_DIR = "artifacts"


def load_embedding_data(embed_type):
    train_file = os.path.join(DATASET_DIR, f"train_{embed_type}_embeddings.npz")
    test_file = os.path.join(DATASET_DIR, f"test_{embed_type}_embeddings.npz")
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    return (train_data['embeddings'], train_data['years'],
            test_data['embeddings'], test_data['years'])


def train_model(X_train, y_train):
    """Random Subspace modeli eğitir (özellik alt kümesi yöntemi)."""
    base_estimator = DecisionTreeClassifier(random_state=RANDOM_SEED)
    
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        bootstrap_features=BOOTSTRAP_FEATURES,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_true': y_test, 'y_pred': y_pred
    }


def plot_confusion_matrix(metrics, title, output_path):
    cm = metrics['confusion_matrix']
    labels = sorted(list(set(metrics['y_true'])))
    plt.figure(figsize=(16, 14))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Purples',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{title}\nNormalize Edilmiş Confusion Matrix', fontsize=14)
    plt.xlabel('Tahmin Edilen Yıl', fontsize=12)
    plt.ylabel('Gerçek Yıl', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_year_accuracy(metrics, title, output_path):
    y_true, y_pred = metrics['y_true'], metrics['y_pred']
    years = sorted(list(set(y_true)))
    accuracies = [accuracy_score(y_true[y_true == y], y_pred[y_true == y]) 
                  if (y_true == y).sum() > 0 else 0 for y in years]
    
    plt.figure(figsize=(14, 6))
    plt.bar(years, accuracies, color='mediumpurple', edgecolor='indigo')
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
    y_true, y_pred = metrics['y_true'], metrics['y_pred']
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(errors, bins=50, color='mediumpurple', edgecolor='indigo', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Tahmin Hatası Dağılımı', fontsize=12)
    axes[0].set_xlabel('Hata (Tahmin - Gerçek)', fontsize=10)
    axes[0].set_ylabel('Frekans', fontsize=10)
    mae, rmse = np.mean(np.abs(errors)), np.sqrt(np.mean(errors**2))
    axes[0].text(0.95, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                 transform=axes[0].transAxes, fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10, c='mediumpurple')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', linewidth=2, label='İdeal')
    axes[1].set_title('Gerçek vs Tahmin', fontsize=12)
    axes[1].set_xlabel('Gerçek Yıl', fontsize=10)
    axes[1].set_ylabel('Tahmin Edilen Yıl', fontsize=10)
    axes[1].legend()
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(all_results, output_dir):
    summary = {et: {'accuracy': r['metrics']['accuracy'], 'f1_macro': r['metrics']['f1_macro'],
                    'f1_weighted': r['metrics']['f1_weighted'], 
                    'precision_macro': r['metrics']['precision_macro'],
                    'recall_macro': r['metrics']['recall_macro']} 
               for et, r in all_results.items()}
    
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    report = f"""# Random Subspace Sonuçları

## Model Parametreleri
| Parametre | Değer |
|-----------|-------|
| n_estimators | {N_ESTIMATORS} |
| max_samples | {MAX_SAMPLES} |
| max_features | {MAX_FEATURES} |
| bootstrap | {BOOTSTRAP} |
| bootstrap_features | {BOOTSTRAP_FEATURES} |
| random_state | {RANDOM_SEED} |

## Performans Karşılaştırması
| Veriseti | Accuracy | F1 (Macro) | F1 (Weighted) | Precision | Recall |
|----------|----------|------------|---------------|-----------|--------|
"""
    for et, m in summary.items():
        report += f"| {et} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {m['f1_weighted']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} |\n"
    
    for et, r in all_results.items():
        report += f"\n## {et.capitalize()} Detaylı Rapor\n```\n{r['metrics']['classification_report']}\n```\n"
    
    with open(os.path.join(output_dir, "results_report.md"), "w", encoding="utf-8") as f:
        f.write(report)


def plot_comparison(all_results, output_dir):
    embed_types = list(all_results.keys())
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    metrics_labels = ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision', 'Recall']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    x = np.arange(len(embed_types))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (mn, ml, c) in enumerate(zip(metrics_names, metrics_labels, colors)):
        values = [all_results[et]['metrics'][mn] for et in embed_types]
        bars = ax.bar(x + i * width, values, width, label=ml, color=c)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Veriseti', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Random Subspace - Veriseti Karşılaştırması', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(['Başlık', 'Özet', 'Birleştirilmiş'])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("RANDOM SUBSPACE MODEL EĞİTİMİ VE DEĞERLENDİRMESİ")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(ARTIFACTS_DIR, f"{timestamp}_random_subspace")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nÇıktı dizini: {output_dir}")
    
    embed_types = ['title', 'abstract', 'concat']
    embed_names = {'title': 'Başlık', 'abstract': 'Özet', 'concat': 'Birleştirilmiş'}
    all_results = {}
    
    for embed_type in embed_types:
        print(f"\n{'='*60}\n{embed_names[embed_type].upper()} VERİSETİ\n{'='*60}")
        
        print("\n1. Veri yükleniyor...")
        X_train, y_train, X_test, y_test = load_embedding_data(embed_type)
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        print("\n2. Model eğitiliyor...")
        model = train_model(X_train, y_train)
        
        print("\n3. Model değerlendiriliyor...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"   Accuracy: {metrics['accuracy']:.4f}, F1 (Macro): {metrics['f1_macro']:.4f}")
        
        print("\n4. Grafikler oluşturuluyor...")
        plot_confusion_matrix(metrics, f"Random Subspace - {embed_names[embed_type]}",
                              os.path.join(output_dir, f"{embed_type}_confusion_matrix.png"))
        plot_year_accuracy(metrics, f"Random Subspace - {embed_names[embed_type]}",
                           os.path.join(output_dir, f"{embed_type}_year_accuracy.png"))
        plot_prediction_distribution(metrics, f"Random Subspace - {embed_names[embed_type]}",
                                     os.path.join(output_dir, f"{embed_type}_prediction_dist.png"))
        
        model_path = os.path.join(output_dir, f"{embed_type}_model.joblib")
        joblib.dump(model, model_path)
        print(f"   Model kaydedildi: {model_path}")
        
        all_results[embed_type] = {'metrics': metrics, 'model_path': model_path}
    
    print(f"\n{'='*60}\nSONUÇLAR KAYDEDİLİYOR\n{'='*60}")
    save_results(all_results, output_dir)
    plot_comparison(all_results, output_dir)
    print(f"\n✓ Tüm sonuçlar kaydedildi: {output_dir}/")
    
    print("\n" + "=" * 60 + "\nÖZET\n" + "=" * 60)
    print("\n{:<15} {:>10} {:>10} {:>10}".format("Veriseti", "Accuracy", "F1-Macro", "F1-Weight"))
    print("-" * 50)
    for et in embed_types:
        m = all_results[et]['metrics']
        print(f"{embed_names[et]:<15} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['f1_weighted']:>10.4f}")


if __name__ == "__main__":
    main()

