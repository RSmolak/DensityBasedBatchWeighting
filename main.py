import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from densityWeights import get_kde_weights
from model import MyNNModel
from datasets.custom_dataset import WeightedDataset

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler


# Learning hyperparameters
num_epoch = 10
batch_size = 10

# Generate a classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


generated_dataset_weights = get_kde_weights(X_train)
generated_dataset = WeightedDataset(X_train, y_train, generated_dataset_weights)

weights = np.array(get_kde_weights(X))
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


train = DataLoader(dataset=generated_dataset, batch_size=10, sampler=sampler)


# Define model
model = MyNNModel(len(X_train[0]), 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training loop
for epoch in range(num_epoch):
    pass

# Print data, weights and classes
# for i in range(len(X)):
#     print(X[i], weights[i], y[i])


# Create neural network model


# colors = df["Target"].map({0: 'blue', 1: 'red'})
#
# plt.figure(figsize=(10, 7))
# plt.scatter(df["Feature 1"], df["Feature 2"], c=colors, s=weights)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter plot of the dataset')
# plt.show()

