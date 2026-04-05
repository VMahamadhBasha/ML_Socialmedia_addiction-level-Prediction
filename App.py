"""
app.py
======
Flask web server - loads trained model and serves predictions

Run:  python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────
print("Loading model...")
with open("model.pkl", "rb") as f:
    payload = pickle.load(f)

model    = payload["model"]
scaler   = payload["scaler"]
features = payload["features"]
accuracy = payload.get("accuracy", "N/A")
roc_auc  = payload.get("roc_auc", "N/A")

print(f"✓ Model loaded")
print(f"  Features : {features}")
print(f"  Accuracy : {accuracy}%")
print(f"  ROC-AUC  : {roc_auc}")


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model metadata + features to the frontend."""
    return jsonify({
        "features": features,
        "accuracy": accuracy,
        "roc_auc":  roc_auc
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON input, return prediction."""
    body   = request.get_json()
    inputs = body.get("inputs", {})

    # Build feature vector in correct order
    values = []
    for feat in features:
        try:
            values.append(float(inputs.get(feat, 0)))
        except (ValueError, TypeError):
            values.append(0.0)

    X        = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]   # [not_addicted, addicted]

    # prediction: 1 = Addicted, 0 = Not Addicted
    label      = "Addicted" if prediction == 1 else "Not Addicted"
    confidence = round(float(max(proba)) * 100, 1)
    add_prob   = round(float(proba[1]) * 100, 1)
    not_prob   = round(float(proba[0]) * 100, 1)

    # Total usage for context
    total_mins = sum(values)
    total_hrs  = round(total_mins / 60, 1)

    return jsonify({
        "prediction":       label,
        "confidence":       confidence,
        "addicted_prob":    add_prob,
        "not_addicted_prob": not_prob,
        "total_mins":       round(total_mins, 0),
        "total_hrs":        total_hrs
    })


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("  Server starting at http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, port=5000)