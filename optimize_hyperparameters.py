"""
Hiperparametre Optimizasyonu Scripti

Bu script:
1. Bagging, Random Subspace ve Random Forest için hiperparametre optimizasyonu yapar
2. Her yöntem için 2 hiperparametre optimize edilir
3. En iyi parametreleri ve performansı raporlar
"""

import os
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import json

RANDOM_SEED = 42
DATASET_DIR = "dataset"
ARTIFACTS_DIR = "artifacts"
CV_FOLDS = 3  # Cross-validation fold sayısı
N_JOBS = -1


def load_embedding_data(embed_type):
    """Embedding verisetini yükler."""
    train_file = os.path.join(DATASET_DIR, f"train_{embed_type}_embeddings.npz")
    test_file = os.path.join(DATASET_DIR, f"test_{embed_type}_embeddings.npz")
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    return (train_data['embeddings'], train_data['years'],
            test_data['embeddings'], test_data['years'])


def optimize_random_forest(X_train, y_train, X_test, y_test):
    """
    Random Forest hiperparametre optimizasyonu.
    Optimize edilen parametreler: n_estimators, max_depth
    """
    print("\n" + "="*50)
    print("RANDOM FOREST OPTİMİZASYONU")
    print("="*50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None]
    }
    
    model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS)
    
    print(f"Grid Search başlatılıyor... (CV={CV_FOLDS})")
    print(f"Parametre alanı: {param_grid}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring='accuracy',
        verbose=2, n_jobs=1  # Model zaten paralel
    )
    grid_search.fit(X_train, y_train)
    
    # En iyi model ile test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'all_results': []
    }
    
    # Tüm sonuçları kaydet
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        results['all_results'].append({
            'params': params,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score
        })
    
    print(f"\n✓ En İyi Parametreler: {grid_search.best_params_}")
    print(f"✓ En İyi CV Skoru: {grid_search.best_score_:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test F1-Macro: {test_f1:.4f}")
    
    return results


def optimize_bagging(X_train, y_train, X_test, y_test):
    """
    Bagging hiperparametre optimizasyonu.
    Optimize edilen parametreler: n_estimators, max_samples
    """
    print("\n" + "="*50)
    print("BAGGING OPTİMİZASYONU")
    print("="*50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.7, 1.0]
    }
    
    base = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model = BaggingClassifier(estimator=base, random_state=RANDOM_SEED, n_jobs=N_JOBS)
    
    print(f"Grid Search başlatılıyor... (CV={CV_FOLDS})")
    print(f"Parametre alanı: {param_grid}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring='accuracy', verbose=2, n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'all_results': []
    }
    
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        results['all_results'].append({
            'params': {k: v for k, v in params.items()},
            'mean_cv_score': mean_score,
            'std_cv_score': std_score
        })
    
    print(f"\n✓ En İyi Parametreler: {grid_search.best_params_}")
    print(f"✓ En İyi CV Skoru: {grid_search.best_score_:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test F1-Macro: {test_f1:.4f}")
    
    return results


def optimize_random_subspace(X_train, y_train, X_test, y_test):
    """
    Random Subspace hiperparametre optimizasyonu.
    Optimize edilen parametreler: n_estimators, max_features
    """
    print("\n" + "="*50)
    print("RANDOM SUBSPACE OPTİMİZASYONU")
    print("="*50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': [0.3, 0.5, 0.7]
    }
    
    base = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model = BaggingClassifier(
        estimator=base, bootstrap=False, bootstrap_features=True,
        random_state=RANDOM_SEED, n_jobs=N_JOBS
    )
    
    print(f"Grid Search başlatılıyor... (CV={CV_FOLDS})")
    print(f"Parametre alanı: {param_grid}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=CV_FOLDS, scoring='accuracy', verbose=2, n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    results = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'all_results': []
    }
    
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        results['all_results'].append({
            'params': {k: v for k, v in params.items()},
            'mean_cv_score': mean_score,
            'std_cv_score': std_score
        })
    
    print(f"\n✓ En İyi Parametreler: {grid_search.best_params_}")
    print(f"✓ En İyi CV Skoru: {grid_search.best_score_:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test F1-Macro: {test_f1:.4f}")
    
    return results


def save_optimization_results(all_results, output_dir):
    """Optimizasyon sonuçlarını kaydeder."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON kaydet
    json_results = {}
    for method, datasets in all_results.items():
        json_results[method] = {}
        for dataset, results in datasets.items():
            json_results[method][dataset] = {
                'best_params': {k: (v if v is not None else "None") 
                               for k, v in results['best_params'].items()},
                'best_cv_score': float(results['best_cv_score']),
                'test_accuracy': float(results['test_accuracy']),
                'test_f1_macro': float(results['test_f1_macro'])
            }
    
    with open(os.path.join(output_dir, "optimization_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Markdown rapor oluştur
    report = """# Hiperparametre Optimizasyonu Sonuçları

## Özet

Bu rapor, Bagging, Random Subspace ve Random Forest algoritmaları için yapılan 
hiperparametre optimizasyonunun sonuçlarını içermektedir.

### Optimize Edilen Parametreler

| Algoritma | Parametre 1 | Parametre 2 |
|-----------|-------------|-------------|
| Random Forest | n_estimators | max_depth |
| Bagging | n_estimators | max_samples |
| Random Subspace | n_estimators | max_features |

---

"""
    
    for method in ['random_forest', 'bagging', 'random_subspace']:
        method_name = method.replace('_', ' ').title()
        report += f"## {method_name}\n\n"
        report += "| Veriseti | En İyi Parametreler | CV Skoru | Test Acc | Test F1 |\n"
        report += "|----------|---------------------|----------|----------|--------|\n"
        
        for dataset in ['title', 'abstract', 'concat']:
            r = all_results[method][dataset]
            params_str = ", ".join([f"{k}={v}" for k, v in r['best_params'].items()])
            report += f"| {dataset} | {params_str} | {r['best_cv_score']:.4f} | {r['test_accuracy']:.4f} | {r['test_f1_macro']:.4f} |\n"
        
        report += "\n"
    
    report += """---

## Sonuç ve Öneriler

Bu optimizasyon sonuçlarına göre, her algoritma için önerilen hiperparametreler 
ilgili run scriptlerinde kullanılabilir.

**Not**: Grid Search ile {}-fold cross-validation kullanılmıştır.
""".format(CV_FOLDS)
    
    with open(os.path.join(output_dir, "optimization_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✓ Sonuçlar kaydedildi: {output_dir}/")


def main():
    print("=" * 60)
    print("HİPERPARAMETRE OPTİMİZASYONU")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(ARTIFACTS_DIR, f"{timestamp}_optimization")
    
    embed_types = ['title', 'abstract', 'concat']
    embed_names = {'title': 'Başlık', 'abstract': 'Özet', 'concat': 'Birleştirilmiş'}
    
    all_results = {
        'random_forest': {},
        'bagging': {},
        'random_subspace': {}
    }
    
    for embed_type in embed_types:
        print(f"\n{'#'*60}")
        print(f"# VERİSETİ: {embed_names[embed_type].upper()}")
        print(f"{'#'*60}")
        
        print("\nVeri yükleniyor...")
        X_train, y_train, X_test, y_test = load_embedding_data(embed_type)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Random Forest
        all_results['random_forest'][embed_type] = optimize_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        # Bagging
        all_results['bagging'][embed_type] = optimize_bagging(
            X_train, y_train, X_test, y_test
        )
        
        # Random Subspace
        all_results['random_subspace'][embed_type] = optimize_random_subspace(
            X_train, y_train, X_test, y_test
        )
    
    # Sonuçları kaydet
    save_optimization_results(all_results, output_dir)
    
    # Özet yazdır
    print("\n" + "=" * 60)
    print("OPTİMİZASYON SONUÇLARI ÖZETİ")
    print("=" * 60)
    
    for method in ['random_forest', 'bagging', 'random_subspace']:
        print(f"\n{method.upper().replace('_', ' ')}:")
        print("-" * 50)
        for dataset in embed_types:
            r = all_results[method][dataset]
            params = ", ".join([f"{k}={v}" for k, v in r['best_params'].items()])
            print(f"  {embed_names[dataset]:<15}: {params}")
            print(f"                   Test Acc: {r['test_accuracy']:.4f}, F1: {r['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()

