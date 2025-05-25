from typing import Dict, Any
from .backtesting import BacktestingSimulator
from .metrics import calculate_classification_metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import PAYOUT
class ModelEvaluator:
    def __init__(self):
        self.backtester = BacktestingSimulator()

    def evaluate_model(
        self, model: Any, X_test: np.ndarray, X_val: np.ndarray, y_test: np.ndarray, y_val: np.ndarray, model_name: str, asset: str, scoring: str
    ) -> Dict[str, float]:
        """
        Evaluate a trained model with comprehensive metrics
        Args:
            model: Trained model
            X_test: Test features
            X_val: Validation features
            y_test: Test labels
            y_val: Validation labels
            model_name: Name of the model
            asset: Asset name
        Returns:
            Dictionary containing all evaluation metrics
        """
        for X, y, data_type in zip([X_test, X_val], [y_test, y_val], ["test", "val"]):
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            # Basic classification metrics
            basic_metrics = calculate_classification_metrics(y, y_pred, y_prob)

            # Run backtesting simulation
            backtest_results = self.backtester.simulate(y, y_pred, y_prob)

            # Combine all metrics
            metrics_dict = {
                **basic_metrics,
                "Sharpe Ratio": backtest_results["sharpe_ratio"],
                "Sortino Ratio": backtest_results["sortino_ratio"],
                "Backtesting Final Value": backtest_results["returns"],
                "Return over Risked Capital": backtest_results["Return over Risked Capital"],
                "Backtesting Accuracy": backtest_results["accuracy"],
                "Operations %": backtest_results["operations_pct"],
                "Dataset": data_type
            }

            self.save_metrics(metrics_dict, asset, model_name, scoring, data_type)
        return metrics_dict

    def plot_metrics(self, metrics_dict: Dict[str, Dict[str, float]], asset: str):
        """
        Plot evaluation metrics with enhanced visualization
        """
        metrics_df = pd.DataFrame(metrics_dict).T

        # Create separate plots for different metric categories
        metric_categories = {
            "Classification Metrics": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Risk Metrics": ["Sharpe Ratio", "Sortino Ratio", "Brier Score"],
            "Return Metrics": [
                "Expected Return",
                "Adjusted Expected Return",
                "Backtesting ROI",
            ],
            "Trading Metrics": ["Backtesting Accuracy", "Operations %"],
        }

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"Model Performance Metrics - {asset}", fontsize=16)

        for (category, metrics), ax in zip(metric_categories.items(), axes.flat):
            metrics_df[metrics].plot(kind="bar", ax=ax)
            ax.set_title(category)
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def save_metrics(self, metrics_dict: Dict[str, Dict[str, float]], asset: str, model_name: str, scoring: str, dataset: str):
        """
        Save evaluation metrics to a single CSV file, updating existing entries if they exist
        Args:
            metrics_dict: Dictionary containing metrics
            asset: Asset symbol
            model_name: Name of the model
            dataset: Dataset name
        """
        csv_path = f"metrics/metrics_{scoring}_{PAYOUT}.csv"
        
        # Create a DataFrame from the current metrics
        metrics_df = pd.DataFrame([metrics_dict])
        
        # Add asset and model columns
        metrics_df['asset'] = asset
        metrics_df['model'] = model_name
        metrics_df['dataset'] = dataset
        try:
            # Try to read existing CSV file
            existing_df = pd.read_csv(csv_path)
            
            # Remove existing entry for this asset-model combination if it exists
            mask = ~((existing_df['asset'] == asset) & (existing_df['model'] == model_name) & (existing_df['dataset'] == dataset))
            existing_df = existing_df[mask]
            
            # Append new metrics
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        except FileNotFoundError:
            # If file doesn't exist, create new DataFrame
            updated_df = metrics_df
        
        # Save to CSV
        updated_df.to_csv(csv_path, index=False)


       
