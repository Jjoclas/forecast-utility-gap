import os
import pandas as pd
from typing import Dict, Union, List, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from ..models.classifiers import AVAILABLE_MODELS
from ..evaluation.evaluator import ModelEvaluator
from ..models.trainer import ModelTrainer
from utils.features import talib_features, split_dataset
from pathlib import Path

class ModelPipeline:
    def __init__(self, assets: List[str], timeframe: str = "M1", metric: str = "accuracy"):
        """
        Initialize the pipeline with enhanced evaluation capabilities
        """
        self.assets = assets
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        # self.feature_analyzer = FeatureAnalyzer()
        self.results = {}
        self.timeframe = timeframe
        self.scaler = StandardScaler()
        self.metric = metric

    def get_model(self, asset: str, model_name: str):
        """
        Get a model for a specific asset
        """
        current_dir = Path(__file__).parent.resolve()
        project_root = current_dir.parents[1]
        model_path = project_root / "models" / f"{asset}_{model_name}_{self.metric}.joblib"

        if self.check_if_model_exists(model_path):
            return joblib.load(model_path)
        

    def load_data(self, asset: str) -> pd.DataFrame:
        """
        Load data for a specific asset and apply feature engineering
        Args:
            asset: The asset symbol (e.g., "EURGBP")
        Returns:
            pd.DataFrame: The loaded and processed data
        """
        try:
            asset = asset.replace(":", "")
            current_dir = Path(__file__).parent.resolve()
            project_root = current_dir.parents[1]

            # Build full path to the CSV: /app/candles/{asset}_{self.timeframe}.csv
            csv_path = project_root / "candles" / f"{asset}_{self.timeframe}.csv"
            df = pd.read_csv(csv_path)
            return talib_features(df)
        except Exception as e:
            print(f"Error loading data for {asset}: {e}")
            return None

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = 0.2
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing
        Args:
            df: The input DataFrame
            test_size: The proportion of data to use for testing
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        # Remove target and date-related columns
        feature_cols = [
            col
            for col in df.columns
            if col
            not in [
                "at",
                "to",
                "from",
                "open",
                "close",
                "min",
                "max",
                "volume",
                "diff",
                "x",
                "Y",
            ]
        ]

        # Define date cutoffs
        train_cutoff = pd.Timestamp('2024-06-01', tz='UTC')
        val_cutoff = pd.Timestamp('2024-10-01', tz='UTC')

        # Split data based on dates
        train_mask = df.index < train_cutoff
        val_mask = (df.index >= train_cutoff) & (df.index < val_cutoff)
        test_mask = df.index >= val_cutoff

        # Split features and target
        X_train = df[feature_cols][train_mask]
        X_val = df[feature_cols][val_mask]
        X_test = df[feature_cols][test_mask]
        
        y_train = df["Y"][train_mask]
        y_val = df["Y"][val_mask]
        y_test = df["Y"][test_mask]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        # print(len(X_train_scaled), len(X_val_scaled), len(X_test_scaled))
        # raise Exception("Stop here")    
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


    def run(self, test_size: float = 0.2):
        """
        Run the complete pipeline with enhanced metrics
        """
        for asset in self.assets:
            print(f"\nProcessing {asset}...")

            # Load and prepare data
            df = self.load_data(asset)
            if df is None:
                continue

            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
                df, test_size
            )

            asset_results = {"models": {}, "metrics": {}}

            for model_name in self.trainer.models:

                # Train model
                model = self.get_model(asset, model_name)
                if model is None:

                    model = self.trainer.train(X_train, y_train, model_name, scoring=self.metric)

                # Evaluate model with enhanced metrics
                metrics = self.evaluator.evaluate_model(model, X_test, X_val, y_test, y_val, model_name, asset, self.metric)

                # Analyze feature importance
                # importance_df = self.feature_analyzer.analyze_feature_importance(
                #     model, feature_names, model_name, asset
                # )

                # Store results
                asset_results["models"][model_name] = model
                asset_results["metrics"][model_name] = metrics

                # Print detailed metrics
                print(f"\nDetailed metrics for {model_name}:")
                for metric, value in metrics.items():
                    if metric == "Dataset":
                        continue
                    print(f"{metric}: {value:.4f}")

                # Save model
                self.save_model(asset, model_name, model)

            # Plot metrics comparison
            # self.evaluator.plot_metrics(asset_results["metrics"], asset)

            # Store asset results
            self.results[asset] = asset_results



    def check_if_model_exists(self, model_path: str) -> bool:
        """
        Check if a model exists for a specific asset
        """
        return os.path.exists(model_path)

    def save_model(self, asset: str, model_name: str, model: Any):
        """
        Save trained models and their metrics
        """
        model_path = f"models/{asset}_{model_name}_{self.metric}.joblib"
        joblib.dump(model, model_path)

