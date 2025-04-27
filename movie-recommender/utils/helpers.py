import numpy as np
import pandas as pd

# Format and print top-N recommended movie titles
def displayRecommendations(movieIds: list, movieMetadata: dict) -> None:
    print("\n🎬 Recommended Movies:")
    for mid in movieIds:
        title = movieMetadata.get(mid, "Unknown Title")
        print(f" - {title}")

# Compute cosine similarity between all vectors in a matrix
def computeCosineSimilarityMatrix(featureMatrix: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(featureMatrix)
    return pd.DataFrame(similarity, index=featureMatrix.index, columns=featureMatrix.index)


# Find top-k similar items to a given vector
def findTopKSimilarItems(targetVector: pd.Series, allVectors: pd.DataFrame, k: int = 10) -> list:
    sims = allVectors.dot(targetVector) / (np.linalg.norm(allVectors, axis=1) * np.linalg.norm(targetVector))
    sims = pd.Series(sims, index=allVectors.index)
    return sims.nlargest(k).index.tolist()

# Normalize feature vectors to unit length
def normalizeVectors(featureMatrix: pd.DataFrame) -> pd.DataFrame:
    norms = np.linalg.norm(featureMatrix.values, axis=1, keepdims=True)
    normalized = featureMatrix.values / norms
    return pd.DataFrame(normalized, index=featureMatrix.index, columns=featureMatrix.columns)



