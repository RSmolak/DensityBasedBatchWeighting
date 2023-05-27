from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from model import MyNNModel, DensityBasedNNClassifier

import torch.nn as nn
import torch.optim as optim

import numpy as np


# Learning hyperparameters
num_epoch = 1000
batch_size = 100
learning_rate = 0.00005

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0)
generated_dataset = (X,y)


# Define model
model = MyNNModel(len(X[0]), 1)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

dbnn_classifier = DensityBasedNNClassifier(model, optimizer, criterion, batch_size, num_epoch)

DATASETS = [
    generated_dataset,
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
            clf = clone(clf_prot) # jak nie ma clone to się uczy, a jak jest to nie. Pewnie jakiś brak spójności między scikit i pytorch
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score = accuracy_score(y[test], y_pred)
            scores[dataset_idx, classifier_idx, fold_idx] = score


print(scores)
np.save("scores", scores)

