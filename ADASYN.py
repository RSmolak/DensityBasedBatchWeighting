from imblearn.over_sampling import ADASYN
from model import CustomNNClassifier

class CustomNNADASYNClassifier(CustomNNClassifier):
    def __init__(self, model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method = None, adasyn_ratio=1):
        super().__init__(model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method)
        self.adasyn = ADASYN(sampling_strategy=adasyn_ratio)
        self.adasyn_ratio = adasyn_ratio

    def fit(self, X, y):
        X_resampled, y_resampled = self.adasyn.fit_resample(X, y)

        # Call the parent class's fit method with resampled data
        super().fit(X_resampled, y_resampled)


