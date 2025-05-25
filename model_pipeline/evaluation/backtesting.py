import numpy as np
from typing import Dict
from utils.config import PAYOUT
import os
import pandas as pd

class BacktestingSimulator:
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance

    @staticmethod
    def calculate_kelly(
        cls, conf: float, payout: float = 0.85, agressiveness: float = 0.1
    ) -> float:
        """Calculate kelly

        Args:
            conf (float): Confidence
            payout (float, optional): Payout. Defaults to 0.85.

        Returns:
            float: Kelly value
        """
        kelly = ((payout * conf) - (1 - conf)) / payout
        return kelly * agressiveness

    def simulate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        payout: float = PAYOUT,
    ) -> Dict[str, float]:
        """
        Simulate trading based on model predictions with realistic business logic

        Args:
            y_true: True labels (-1/1 or 0/1)
            y_pred: Predicted probabilities from model
            y_prob: Raw prediction probabilities
            payout: Payout multiplier for successful trades (default: 0.85)

        Returns:
            Dictionary with backtesting metrics
        """
        balance = self.initial_balance
        results = []
        correct = 0
        incorrect = 0
        n_operations = 0
        sum_value = 0
        min_value = balance
        max_value = balance
        positions = []
        expected_return = []
        kelly_value = 0
        
        # Create a Series with minute frequency, initialized with zeros
        all_returns = pd.Series(0.0, index=y_true.index)
        
        for i, (confidence, predict_class, true_class) in enumerate(zip(y_prob, y_pred, y_true)):
            n_operations += 1
            confidence = confidence if predict_class == 1 else 1 - confidence
            # Calculate expected value
            p_success = confidence
            ev = (p_success * payout) - (1 - p_success)
            expected_return.append(ev)
            # print(f"Trade {i}: \n Confidence: {confidence:.2f} \n True Class {true_class} \n Predicted Class {predict_class} \n Expected Value: {ev:.2f}")
            # Skip trades with negative expected value
            if ev <= 0:
                continue

            kelly = self.calculate_kelly(confidence, payout)

            position_size = kelly * self.initial_balance
            sum_value += position_size

            # Determine if prediction was correct
            is_correct = predict_class == true_class
            # print(f"Trade {i}: \n position_size: {position_size:.2f} \n Balance: {balance:.2f} \n PayOut: {payout:.2f} \n Is Correct: {is_correct}")
            # Calculate trade result
            if is_correct:
                result = position_size * payout
                correct += 1
                kelly_value = kelly_value + (kelly * payout)
                positions.append(result)

            else:
                result = -position_size
                incorrect += 1
                positions.append(result)
                kelly_value = kelly_value - kelly
            # print(f"Trade {i}: {result:.2f} - {balance:.2f}")
            # print("-" * 50)
            # Update balance and track min/max
            balance += result
            min_value = min(min_value, balance)
            max_value = max(max_value, balance)
            
            # Store the result in the time series at the correct timestamp
            all_returns.iloc[i] = result / self.initial_balance  # Convert to percentage return
            
        # Resample to daily returns
        daily_returns = all_returns.resample('D').sum()  # Sum because these are arithmetic returns
        
        # Calculate metrics using daily data
        avg_daily_return = daily_returns.mean()
        daily_std = daily_returns.std()
        daily_downside_returns = daily_returns[daily_returns < 0]
        daily_downside_std = daily_downside_returns.std() if len(daily_downside_returns) > 0 else 0
        
        # Calculate ratios using daily data
        sharpe_ratio = (avg_daily_return / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        sortino_ratio = (avg_daily_return / daily_downside_std) * np.sqrt(252) if daily_downside_std > 0 else 0
        
        # Calculate performance metrics
        positions = np.array(positions) if positions else np.array([0])
        avg_return = np.mean(positions)
        std_return = np.std(positions) if len(positions) > 1 else 0
        downside_returns = positions[positions < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0

        # Calculate accuracy and operation metrics
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        operations_pct = (
            (correct + incorrect) / n_operations * 100 if n_operations > 0 else 0
        )
        rrc = (balance - self.initial_balance) / sum_value if sum_value > 0 else 0
        return {
            "returns": balance - self.initial_balance,
            "accuracy": accuracy,
            "operations_pct": operations_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "total_trades": correct + incorrect,
            "winning_trades": correct,
            "losing_trades": incorrect,
            "total_invested": sum_value,
            "max_balance": max_value,
            "min_balance": min_value,
            "final_balance": balance,
            "kelly_value": kelly_value,
            "avg_return_per_trade": avg_return,
            "return_std": std_return,
            "downside_std": downside_std,
            "Return over Risked Capital": rrc,
        }