from utils.dataLoader import IMDbLoader, MovieLensLoader, MetadataPreprocessor, RatingsPreprocessor
import pandas as pd
import numpy as np

class DataTester:
    def __init__(self):
        self.metadata = None
        self.featureMatrix = None
        self.ratings = None
        self.binaryRatings = None

    def run(self):
        print("\n🚀 Running Data Tester: \n")
        
        try:
            self._load_metadata()
            self._build_features()
            self._load_ratings()
            
            return {
                "metadata": self.metadata,
                "featureMatrix": self.featureMatrix,
                "binaryRatings": self.binaryRatings
            }
            
        except Exception as e:
            print(f"\n⚠️ Error in DataTester: {str(e)}")
            return self._get_fallback_output()

    def _load_metadata(self):
        try:
            imdbLoader = IMDbLoader("ml-100k/links.csv", apiKey="766c1b0d")
            raw_metadata = imdbLoader.loadMetadata()
                
            if 'movieId' not in raw_metadata.columns:
                if 'id' in raw_metadata.columns:
                    raw_metadata = raw_metadata.rename(columns={'id': 'movieId'})
                else:
                    raw_metadata['movieId'] = np.arange(1, len(raw_metadata)+1)
            
            self.metadata = imdbLoader.preprocessMetadata()
            print(f"✅ Loaded {len(self.metadata)} movies metadata")
            
        except Exception as e:
            print(f"❌ Metadata loading failed: {str(e)}")
            self.metadata = pd.DataFrame(columns=['movieId', 'title'])
            raise

    def _build_features(self):
        try:
            if self.metadata.empty:
                raise ValueError("No metadata available")
                
            processor = MetadataPreprocessor(self.metadata)
            
            try:
                categorical = processor.encodeCategoricalFeatures()
            except Exception:
                print("⚠️ Could not encode categorical features")
                categorical = pd.DataFrame(index=self.metadata.index)
            
            try:
                votes = processor.normalizeVoteAverage()
            except Exception:
                print("⚠️ Could not normalize vote features")
                votes = pd.DataFrame(index=self.metadata.index)
            
            self.featureMatrix = categorical.join(votes, how='outer')
            self.featureMatrix.index = self.metadata["movieId"]
            print(f"✅ Feature matrix shape: {self.featureMatrix.shape}")
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {str(e)}")
            self.featureMatrix = pd.DataFrame(index=self.metadata.index if not self.metadata.empty else None)
            raise

    def _load_ratings(self):
        try:
            loader = MovieLensLoader("ml-100k/ratings.csv")
            self.ratings = loader.loadRatings()
                
            processor = RatingsPreprocessor(self.ratings)
            
            try:
                self.binaryRatings = processor.binarizeRatings(threshold=3.5)
                print(f"✅ Ratings binarized: {self.binaryRatings['binaryRating'].value_counts().to_dict()}")
            except Exception:
                print("⚠️ Could not binarize ratings, using raw values")
                self.binaryRatings = self.ratings.copy()
                if 'binaryRating' not in self.binaryRatings.columns:
                    self.binaryRatings['binaryRating'] = (self.binaryRatings.get('rating', 0) >= 3.5).astype(int)
            
            print(f"✅ Loaded {self.ratings['userId'].nunique()} users and {self.ratings['movieId'].nunique()} movies")
            
        except Exception as e:
            print(f"❌ Ratings processing failed: {str(e)}")
            self.ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
            self.binaryRatings = pd.DataFrame(columns=['userId', 'movieId', 'binaryRating'])
            raise

    def _get_fallback_output(self):
        return {
            "metadata": self.metadata if self.metadata is not None else pd.DataFrame(columns=['movieId', 'title']),
            "featureMatrix": self.featureMatrix if self.featureMatrix is not None else pd.DataFrame(),
            "binaryRatings": self.binaryRatings if self.binaryRatings is not None else pd.DataFrame(columns=['userId', 'movieId', 'binaryRating'])
        }