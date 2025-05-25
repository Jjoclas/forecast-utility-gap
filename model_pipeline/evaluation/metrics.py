from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    brier_score_loss,
    log_loss,
    roc_auc_score
)
from utils.metrics import (
    expected_return_metric, 
    adjusted_expected_return_metric, 
    expected_growth_rate_metric,
    expected_profit_loss_metrics,
    expected_value_metric
)
from utils.config import PAYOUT

def calculate_classification_metrics(y_true, y_pred, y_prob):
    """Calculate basic classification metrics"""
    # Calculate EPEL metrics
    EP, EL = expected_profit_loss_metrics(y_true, y_prob, payout=PAYOUT)
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Brier Score': brier_score_loss(y_true, y_prob),
        "Expected Return": expected_return_metric(y_true, y_prob, payout=PAYOUT),
        "Adjusted Expected Return": adjusted_expected_return_metric(y_true, y_prob, payout=PAYOUT),
        "Expected Growth Rate": expected_growth_rate_metric(y_true, y_prob, payout=PAYOUT),
        "Negative Log Loss": -log_loss(y_true, y_prob),
        # Add new EPEL and EV metrics
        "Expected Profit": EP,
        "Expected Loss": EL,
        "Expected Value": expected_value_metric(y_true, y_prob, payout=PAYOUT)
    }

def calculate_trading_metrics(returns, predictions):
    """Calculate trading-specific metrics"""
    # Implementation of trading metrics
    pass 