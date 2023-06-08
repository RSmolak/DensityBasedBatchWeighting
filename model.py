import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from densityWeights import get_kde_weights, get_weights
from datasets.custom_dataset import WeightedDataset
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
class MyNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CustomNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method = None):
        self.model_class = model_class
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.imbalanced_opt_method = imbalanced_opt_method

    def fit(self, X, y):
        # Inicjalizacja modelu, optymalizatora i kryterium strat

        self.model = self.model_class(X.shape[1], self.output_size)
        self.model.to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


        if self.imbalanced_opt_method == 'density_weighting':
            generated_dataset_weights = get_kde_weights(X, transform='normalize-expand')
        elif self.imbalanced_opt_method == 'count_weighting':
            generated_dataset_weights = get_weights(y)
        else:
            generated_dataset_weights = np.ones(len(y))
        X = torch.tensor(X, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        dataset = WeightedDataset(X, y, generated_dataset_weights)
        train_generated_data_loader = DataLoader(dataset, self.batch_size, shuffle=True)

        for epoch in range(self.num_epoch):
            self.model.train()
            for i, data in enumerate(train_generated_data_loader):
                batch_data, batch_targets, weights = data
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                weights = weights.to(device)
                outputs = self.model(batch_data.float()).squeeze()
                raw_loss = self.criterion(outputs, batch_targets.float())
                weighted_loss = (raw_loss * weights).mean()
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                if (epoch % 50 == 0 or epoch == self.num_epoch-1) and i == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {weighted_loss.item()}')
        return self

    def predict(self, X):
        self.model.eval()  # ustawiamy model w tryb ewaluacji
        inputs = torch.tensor(X, dtype=torch.float).to(device)
        with torch.no_grad():  # wyłączamy obliczanie gradientów
            outputs = self.model(inputs)
        predicted = torch.round(torch.sigmoid(outputs)).cpu().numpy().flatten()  # zwracamy numpy array z przewidywaniami
        return predicted