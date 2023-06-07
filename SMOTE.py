from imblearn.over_sampling import SMOTE
from model import CustomNNClassifier

class CustomNNSMOTEClassifier(CustomNNClassifier):
    def __init__(self, model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method = None, smote_ratio=1):
        super().__init__(model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method)
        self.smote = SMOTE(sampling_strategy=smote_ratio)
        self.smote_ratio = smote_ratio

    def fit(self, X, y):
        X_resampled, y_resampled = self.smote.fit_resample(X, y)

        # Call the parent class's fit method with resampled data
        super().fit(X_resampled, y_resampled)

