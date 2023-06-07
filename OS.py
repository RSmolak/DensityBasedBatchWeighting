from imblearn.over_sampling import RandomOverSampler
from model import CustomNNClassifier

class CustomNNRandomOversamplingClassifier(CustomNNClassifier):
    def __init__(self, model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method = None, ros_ratio=1):
        super().__init__(model_class, output_size, learning_rate, batch_size, num_epoch, imbalanced_opt_method)
        self.ros = RandomOverSampler(sampling_strategy=ros_ratio)
        self.ros_ratio = ros_ratio

    def fit(self, X, y):
        X_resampled, y_resampled = self.ros.fit_resample(X, y)

        # Call the parent class's fit method with resampled data
        super().fit(X_resampled, y_resampled)



