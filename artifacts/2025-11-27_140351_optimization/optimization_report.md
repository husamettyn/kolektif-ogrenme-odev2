# Hiperparametre Optimizasyonu Sonuçları

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

## Random Forest

| Veriseti | En İyi Parametreler | CV Skoru | Test Acc | Test F1 |
|----------|---------------------|----------|----------|--------|
| title | max_depth=20, n_estimators=100 | 0.0486 | 0.0533 | 0.0530 |
| abstract | max_depth=30, n_estimators=200 | 0.0728 | 0.0738 | 0.0711 |
| concat | max_depth=20, n_estimators=200 | 0.0714 | 0.0723 | 0.0700 |

## Bagging

| Veriseti | En İyi Parametreler | CV Skoru | Test Acc | Test F1 |
|----------|---------------------|----------|----------|--------|
| title | max_samples=1.0, n_estimators=200 | 0.0549 | 0.0517 | 0.0515 |
| abstract | max_samples=1.0, n_estimators=200 | 0.0758 | 0.0813 | 0.0788 |
| concat | max_samples=1.0, n_estimators=200 | 0.0688 | 0.0704 | 0.0688 |

## Random Subspace

| Veriseti | En İyi Parametreler | CV Skoru | Test Acc | Test F1 |
|----------|---------------------|----------|----------|--------|
| title | max_features=0.7, n_estimators=100 | 0.0552 | 0.0539 | 0.0532 |
| abstract | max_features=0.3, n_estimators=200 | 0.0776 | 0.0806 | 0.0777 |
| concat | max_features=0.5, n_estimators=200 | 0.0752 | 0.0787 | 0.0773 |

---

## Sonuç ve Öneriler

Bu optimizasyon sonuçlarına göre, her algoritma için önerilen hiperparametreler 
ilgili run scriptlerinde kullanılabilir.

**Not**: Grid Search ile 3-fold cross-validation kullanılmıştır.
