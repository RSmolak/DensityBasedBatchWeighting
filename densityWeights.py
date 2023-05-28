import numpy as np
from scipy.stats import gaussian_kde
def get_kde_weights(data: np.ndarray, transform = None):
    data = data.T
    weights = []
    kde = gaussian_kde(data)
    for sample in data.T:
        weights.append(1/kde.evaluate(sample)[0])
    if transform == 'normalize':
        weights = [float(i)/sum(weights) for i in weights]
    if transform == 'standardize':
        weights = np.array(weights)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        weights = (weights - weight_mean) / weight_std + 1
    if transform == 'normalize-expand':
        weights = [float(i) / sum(weights) for i in weights]
        mean = np.mean(weights)
        multiplicator  = 1/mean
        weights = [weight * multiplicator for weight in weights]
    return weights

def get_weights(labels : np.ndarray):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    class_weights = {}

    for label, count in zip(unique_labels, label_counts):
        class_weights[label] = total_samples / (len(unique_labels) * count)

    weights = [class_weights[label] for label in labels]
    mean = np.mean(weights)
    multiplicator = 1 / mean
    weights = [weight * multiplicator for weight in weights]
    return weights
