from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os  # Required for Vercel

app = Flask(__name__)
CORS(app)

# Load model (updated path for Vercel)
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "🌱 Smart AgriAdvisor backend is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        features = np.array([
            [data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ])
        prediction = model.predict(features)
        return jsonify({
            "crops": [prediction[0]],
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

# Vercel-specific configuration
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))