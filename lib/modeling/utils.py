import numpy as np

def prob_to_moneyline(prob: float) -> float:
    """
    Converts a win probability (0–1) into an American moneyline.
    Example:
        0.75 → -300
        0.40 → +150
    """
    if prob <= 0 or prob >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive)")

    if prob >= 0.5:
        # favorite (negative moneyline)
        odds = -100 * prob / (1 - prob)
    else:
        # underdog (positive moneyline)
        odds = 100 * (1 - prob) / prob
    return np.round(odds, 0)
