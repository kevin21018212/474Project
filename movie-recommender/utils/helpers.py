import numpy as np
import pandas as pd


# Normalize feature vectors to unit length
def normalizeVectors(featureMatrix: pd.DataFrame) -> pd.DataFrame:
    norms = np.linalg.norm(featureMatrix.values, axis=1, keepdims=True)
    normalized = featureMatrix.values / norms
    return pd.DataFrame(normalized, index=featureMatrix.index, columns=featureMatrix.columns)
#Precision@K = (# of recommended items in top-K that are relevant) / K
def precisionAtK(recommended: list, relevant: list, k: int = 10) -> float:  
    if not recommended:
        return 0.0
    topK = recommended[:k]
    hits = sum(1 for item in topK if item in relevant)
    return hits / k

#Recall@K = (# of recommended items in top-K that are relevant) / (# of relevant items)
def recallAtK(recommended: list, relevant: list, k: int = 10) -> float:  
    if not relevant:
        return 0.0
    topK = recommended[:k]
    hits = sum(1 for item in topK if item in relevant)
    return hits / len(relevant)
