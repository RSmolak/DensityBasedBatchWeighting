from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

from model import MyNNModel, DensityBasedNNClassifier

import pandas as pd

import numpy as np


# Learning hyperparameters
num_epoch = 1000
batch_size = 100
learning_rate = 0.0000005

# Create datasets
X, y = make_classification(n_samples=1000, n_features=15, n_informative=15, n_redundant=0)
generated_dataset = (X,y)

breast_cancer_dataset = load_breast_cancer(return_X_y=True)
#print(breast_cancer_dataset)

data = pd.read_csv('datasets/heart.csv')
label = 'output'
features = data.columns.tolist()
features.remove(label)
heart_dataset = (data[features].values, data[label].values)
#print(heart_dataset)

data = pd.read_csv('datasets/water_potability.csv')
label = 'Potability'
features = data.columns.tolist()
features.remove(label)
water_potability_dataset = (data[features].values, data[label].values)
# print(water_potability_dataset)


dbnn_classifier = DensityBasedNNClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch
)

DATASETS = [
    #generated_dataset,
    #breast_cancer_dataset,
    heart_dataset,
    #water_potability_dataset
]

CLASSIFIERS = [
    dbnn_classifier
]


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

#dbnn_classifier.fit(X, y)
for dataset_idx, (X,y) in enumerate(DATASETS):
    for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(clf_prot)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score = accuracy_score(y[test], y_pred)
            scores[dataset_idx, classifier_idx, fold_idx] = score


print(scores)
np.save("scores_temp", scores)

