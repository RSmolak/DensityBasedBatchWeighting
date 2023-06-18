from tabulate import tabulate
import numpy as np
from scipy.stats import ttest_rel


scores = np.load("scores_acc.npy")
num_datasets = scores.shape[0]
num_classifiers = scores.shape[1]



# Inicjalizacja tablic do przechowywania statystyki t i wartości p
t_statistics = np.zeros((num_datasets, num_classifiers))
p_values = np.zeros((num_datasets, num_classifiers))

# Obliczanie statystyki t i wartości p dla każdego klasyfikatora w porównaniu do DWC
dwc_idx = 2
for dataset_idx in range(num_datasets):
    for clf_idx in range(num_classifiers):
        if clf_idx == dwc_idx:  # pomijamy porównanie DWC z samym sobą
            t_statistics[dataset_idx, clf_idx] = np.nan
            p_values[dataset_idx, clf_idx] = np.nan
        else:
            result = ttest_rel(scores[dataset_idx, dwc_idx, :], scores[dataset_idx, clf_idx, :])
            t_statistics[dataset_idx, clf_idx] = result.statistic
            p_values[dataset_idx, clf_idx] = result.pvalue

# Tworzenie tabeli z wynikami
# dla uproszczenia, załóżmy, że mamy listę nazw klasyfikatorów
classifier_names = ["NWC", "CWC","DWC", "RC", "SC"]

# Tworzenie tabeli dla każdego zbioru danych
for dataset_idx in range(num_datasets):
    table_t = tabulate([t_statistics[dataset_idx]],
                        tablefmt='grid',
                        headers=classifier_names)
    table_p = tabulate([p_values[dataset_idx]],
                        tablefmt='grid',
                        headers=classifier_names)

    print(f"Zbiór danych {dataset_idx+1}")
    print("Statystyka t:")
    print(table_t)
    print("Wartości p:")
    print(table_p)
