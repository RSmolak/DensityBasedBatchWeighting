import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from densityWeights import get_kde_weights
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Generate a classification dataset
X, y = make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


# Convert the arrays to a Pandas DataFrame
df = pd.DataFrame(data=np.c_[X, y], columns=["Feature 1", "Feature 2", "Target"])
df_train = pd.DataFrame(data=np.c_[X_train, y_train], columns=["Feature 1", "Feature 2", "Target"])
df_test = pd.DataFrame(data=np.c_[X_test, y_test], columns=["Feature 1", "Feature 2", "Target"])

weights = get_kde_weights(X)
for i in range(len(X)):
    print(X[i], weights[i], y[i])

# Create neural network model
model = MLPClassifier(hidden_layer_sizes=(16,16), activation='relu', solver='adam', random_state=42)

# Perform 5-fold cross-validation
# scores = cross_val_score(model, X, y, cv=5)

# Print the cross-validation scores
# print("Cross-validation scores:", scores)
# print("Mean accuracy:", scores.mean())

colors = df["Target"].map({0: 'blue', 1: 'red'})

plt.figure(figsize=(10, 7))
plt.scatter(df["Feature 1"], df["Feature 2"], c=colors, s=weights)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of the dataset')
plt.show()

