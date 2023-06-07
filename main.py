from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, recall_score
from sklearn.datasets import load_breast_cancer
from ADASYN import CustomNNADASYNClassifier
from OS import CustomNNRandomOversamplingClassifier
from SMOTE import CustomNNSMOTEClassifier

from model import MyNNModel, CustomNNClassifier

import pandas as pd

import numpy as np


# Learning hyperparameters
num_epoch = 100
batch_size = 100
learning_rate = 0.00005

# Create datasets
generated_datasets = []
imbalanced_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
for ratio in imbalanced_ratios:
    X, y = make_classification(n_samples=1000, n_features=15, n_informative=15, n_redundant=0, weights=[ratio,1-ratio])
    generated_datasets.append((X,y))

breast_cancer_dataset = load_breast_cancer(return_X_y=True)
#print(breast_cancer_dataset)

data = pd.read_csv('datasets/heart.csv')
label = 'output'
features = data.columns.tolist()
features.remove(label)
heart_dataset = (data[features].values, data[label].values)
unique_labels, label_counts = np.unique(data[label].values, return_counts=True)
print(unique_labels,label_counts)
#print(heart_dataset)

data = pd.read_csv('datasets/water_potability.csv')
label = 'Potability'
features = data.columns.tolist()
features.remove(label)
water_potability_dataset = (data[features].values, data[label].values)
# print(water_potability_dataset)


density_weighting_classifier = CustomNNClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method='density_weighting'
)

count_weighting_classifier = CustomNNClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method='count_weighting'
)
no_weighting_classifier = CustomNNClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method=None
)
adasyn_classifier = CustomNNADASYNClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method=None
)
ros_classifier = CustomNNRandomOversamplingClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method=None
)
smote_classifier = CustomNNSMOTEClassifier(
    model_class=MyNNModel,
    output_size=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epoch=num_epoch,
    imbalanced_opt_method=None
)

DATASETS = [
    #breast_cancer_dataset,
    #heart_dataset,
    #water_potability_dataset
]
DATASETS.extend(generated_datasets)


CLASSIFIERS = [
    no_weighting_classifier,
    count_weighting_classifier,
    density_weighting_classifier,
    #adasyn_classifier,
    #ros_classifier,
    #smote_classifier,
]


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scores_acc = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))
scores_bal_acc = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))
scores_rec = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))
scores_prec = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))



for dataset_idx, (X,y) in enumerate(DATASETS):
    for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
        print(classifier_idx)
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(clf_prot)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score_acc = accuracy_score(y[test], y_pred)
            scores_acc[dataset_idx, classifier_idx, fold_idx] = score_acc
            score_bal_acc = balanced_accuracy_score(y[test], y_pred)
            scores_bal_acc[dataset_idx, classifier_idx, fold_idx] = score_acc
            score_rec = recall_score(y[test], y_pred)
            scores_rec[dataset_idx, classifier_idx, fold_idx] = score_acc
            score_prec = precision_score(y[test], y_pred)
            scores_prec[dataset_idx, classifier_idx, fold_idx] = score_acc


print('ACC:\n', scores_acc)
print('BAL_ACC:\n', scores_bal_acc)
print('REC:\n', scores_rec)
print('PREC:\n', scores_prec)
#np.save("scores_temp", scores)

