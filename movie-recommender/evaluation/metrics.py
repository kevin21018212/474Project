import numpy as np
import pandas as pd
from typing import List

# Calculate RMSE between predicted and actual ratings
def computeRmse(predictions: List[float], actuals: List[float]) -> float:
    pass

# Evaluate precision at top K recommended items
def computePrecisionAtK(recommended: List[int], relevant: List[int], k: int) -> float:
    pass

# Generate confusion matrix for binary classification of likes/dislikes
def computeConfusionMatrix(predicted: List[int], actual: List[int]) -> pd.DataFrame:
    pass

# Evaluate overall classification accuracy
def computeAccuracy(predicted: List[int], actual: List[int]) -> float:
    pass
