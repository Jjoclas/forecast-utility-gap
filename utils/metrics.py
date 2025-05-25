import numpy as np
from utils.config import PAYOUT

def calculate_kelly(conf: float, payout: float = PAYOUT, agressiveness: float=1.) -> float:
    """Calculate kelly
     
    Args:
        conf (float): Confidence
        payout (float, optional): Payout. Defaults to 0.85.
    
    Returns:
        float: Kelly value
    """
    kelly = ((payout * conf) - (1 - conf)) / payout 
    return kelly * agressiveness

def expected_return_metric(y_true, proba, payout=PAYOUT):
    """
    Custom metric to calculate expected return.

    Parameters:
        y_true: True labels.
        proba: Probability of the 1 (UP)label.
        payout: Payout multiplier.

    Returns:
        float: The expected return metric value.
    """
    P_success = np.where(proba > 0.5, proba, 1 - proba)
    P_failure = 1 - P_success

    amount = 1.0   # Example investment amount

    # Example calculation logic
    expected_returns = P_success * (payout * amount) + P_failure * (-amount)
    sum_returns =  expected_returns.sum()
    return sum_returns / len(y_true)


def adjusted_expected_return_metric(y_true, proba, payout=PAYOUT):
    """
    This Metric calculated the expected return adjusted for useful prediction.
    Useful predictions are those where the model is confident it's higher than 
    the minimal P_success necessary to Expected Return to be positive. given by.
    min_P_success = 1 / (payout + 1)

    Parameters:
        y_true: True labels.
        proba: Probability of the 1 (UP)label.
        payout: Payout multiplier.

    Returns:
        float: The adjusted return metric value.
    """
    P_success = np.where(proba > 0.5, proba, 1 - proba)
    P_failure = 1 - P_success

    amount = 1.0   # Example investment amount

    # Example calculation logic
    expected_returns = P_success * (payout * amount) + P_failure * (-amount)
    sum_returns =  np.where(expected_returns > 0, expected_returns, 0).sum()
    return sum_returns / len(y_true)

def expected_growth_rate_metric(y_true, proba, payout=PAYOUT):
    """
    Calculate the Expected Growth Rate (EGR) metric using Kelly Criterion.
    
    The EGR refines the EV calculation by incorporating the Kelly Criterion to determine 
    the optimal fraction of the bankroll to invest in each trade. Only considers operations 
    where the predicted probability exceeds the minimum threshold for positive Expected Return.
    
    Parameters:
        y_true: True labels
        proba: Probability of the 1 (UP) label
        payout: Payout multiplier
    
    Returns:
        float: The Expected Growth Rate metric value
    """
    # Convert probabilities to P_success (probability of successful prediction)
    P_success = np.where(proba > 0.5, proba, 1 - proba)
    
    # Calculate minimum probability threshold for positive Expected Return
    P_min = 1 / (payout + 1)
    
    # Calculate Kelly fraction for each prediction
    kelly_fraction = (P_success * (payout - 1) - (1 - P_success)) / (payout - 1)
    
    # Calculate Expected Value using Kelly fraction
    EV = P_success * (payout * kelly_fraction) + (1 - P_success) * (-kelly_fraction)
    
    # Only consider operations where P_success > P_min
    valid_operations = P_success > P_min
    
    # Calculate EGR as average of valid EVs
    if np.any(valid_operations):
        return np.mean(np.where(valid_operations, EV, 0))
    else:
        return 0.0

def expected_profit_loss_metrics(y_true, proba, payout=PAYOUT):
    """
    Calculate Expected Profit (EP) and Expected Loss (EL) metrics for binary options trading.
    
    Parameters:
        y_true: True labels
        proba: Probability of the 1 (UP) label
        payout: Payout multiplier
    
    Returns:
        tuple: (Expected Profit, Expected Loss)
    """
    # Convert probabilities to P_success (probability of successful prediction)
    P_success = np.where(proba > 0.5, proba, 1 - proba)
    P_failure = 1 - P_success
    
    amount = 1.0  # Base investment amount
    
    # Calculate Expected Profit and Expected Loss
    EP = P_success * (payout * amount)
    EL = P_failure * amount
    
    # Return average EP and EL across all predictions
    return np.mean(EP), np.mean(EL)

def expected_value_metric(y_true, proba, payout=PAYOUT):
    """
    Calculate Expected Value (EV) metric for binary options trading.
    EV is the difference between Expected Profit (EP) and Expected Loss (EL).
    
    Parameters:
        y_true: True labels
        proba: Probability of the 1 (UP) label
        payout: Payout multiplier
    
    Returns:
        float: The Expected Value metric
    """
    EP, EL = expected_profit_loss_metrics(y_true, proba, payout)
    return EP - EL