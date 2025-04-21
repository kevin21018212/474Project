from flask import Flask, request, jsonify # type: ignore
from user.userProfile import UserProfile
from models.contentFilter import ContentBasedRecommender
from models.collabFilter import CollaborativeRecommender
from models.hybrid import HybridRecommender

app = Flask(__name__)

# Load models and data
def loadModelsAndData():
    pass

# Initialize user session and profile
@app.route("/init", methods=["POST"])
def initializeUser():
    pass

# Submit user feedback (like/dislike)
@app.route("/feedback", methods=["POST"])
def handleFeedback():
    pass

# Get movie recommendations for user
@app.route("/recommend", methods=["GET"])
def getRecommendations():
    pass

# Main Flask entry
if __name__ == "__main__":
    app.run(debug=True)
