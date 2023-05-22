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
from torch.utils.data import DataLoader


# Learning hyperparameters
num_epoch = 1000
batch_size = 100
learning_rate = 0.0001

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_batches = len(X_train)//batch_size

generated_dataset_weights = get_kde_weights(X_train)
generated_dataset = WeightedDataset(X_train, y_train, generated_dataset_weights)

train_generated_data_loader = DataLoader(generated_dataset, batch_size, shuffle=True)

val_dataset = WeightedDataset(X_test, y_test, get_kde_weights(X_test))
val_data_loader = DataLoader(val_dataset, batch_size)

# Define model
model = MyNNModel(len(X_train[0]), 1)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epoch):
    for i, data in enumerate(train_generated_data_loader):
        batch_data, batch_targets, weights = data
        # print(data)

        outputs = model(batch_data.float()).squeeze()
        raw_loss = criterion(outputs, batch_targets.float())

        weighted_loss = (raw_loss * weights).mean()

        optimizer.zero_grad()

        weighted_loss.backward()

        optimizer.step()

        print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {weighted_loss.item()}')

        # Validation at the end of the epoch
    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        num_batches = 0
        for i, data in enumerate(val_data_loader):
            batch_data, batch_targets, weights = data
            outputs = model(batch_data.float()).squeeze()
            raw_loss = criterion(outputs, batch_targets.float())
            weighted_loss = (raw_loss * weights).mean()
            validation_loss += weighted_loss.item()
            num_batches += 1
        validation_loss /= num_batches
        print(f'Epoch: {epoch + 1}, Validation Loss: {validation_loss}')
    model.train()
# Print data, weights and classes
# for i in range(len(X)):
#     print(X[i], weights[i], y[i])


# colors = df["Target"].map({0: 'blue', 1: 'red'})
#
# plt.figure(figsize=(10, 7))
# plt.scatter(df["Feature 1"], df["Feature 2"], c=colors, s=weights)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter plot of the dataset')
# plt.show()

