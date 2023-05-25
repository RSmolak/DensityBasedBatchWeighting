
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

from model import MyNNModel, DensityBasedNNClassifier

import torch.nn as nn
import torch.optim as optim


# Learning hyperparameters
num_epoch = 1000
batch_size = 100
learning_rate = 0.0001

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=3, n_informative=3, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = MyNNModel(len(X_train[0]), 1)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

classifier = DensityBasedNNClassifier(model, optimizer, criterion, batch_size, num_epoch)
classifier.fit(X_train, y_train)





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

