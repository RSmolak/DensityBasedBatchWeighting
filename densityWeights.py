import numpy
from scipy.stats import gaussian_kde
def get_kde_weights(data: numpy.ndarray):
    data = data.T
    weights = []
    kde = gaussian_kde(data)
    for sample in data.T:
        weights.append(1/kde.evaluate(sample))
    return weights
