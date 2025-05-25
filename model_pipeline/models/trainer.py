from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from .classifiers import AVAILABLE_MODELS

class ModelTrainer:
    def __init__(self):
        self.models = AVAILABLE_MODELS

    def train(self, X_train, y_train, model_name, scoring="accuracy") -> CalibratedClassifierCV:
        """
        Train a specific model with calibration and grid search
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
        Returns:
            Trained and calibrated model with best parameters
        """
        # Get the base model and its parameters
        base_model = self.models[model_name]["model"]()
        params = self.models[model_name]["params"]
        
        # Create a pipeline with the base model and calibration
        calibrated_classifier = CalibratedClassifierCV(
            estimator=base_model,
            cv=5,
            method="sigmoid"
        )
        
        # Modify parameter names to work with calibrated classifier
        calibrated_params = {}
        for param_name, param_values in params.items():
            calibrated_params[f'estimator__{param_name}'] = param_values
        
        # Perform grid search with the calibrated classifier
        grid_search = GridSearchCV(
            calibrated_classifier,
            param_grid=calibrated_params,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_
