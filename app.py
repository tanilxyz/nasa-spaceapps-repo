# FLASK + XGBOOST + TRANSIT PLOT
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
from threading import Thread
import logging
import sys
import joblib

# --- SUPPRESS FLASK MESSAGES ---
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- GLOBAL VARIABLES ---
ALLOWED_COLUMNS = [
    'kepid', 'kepoi_name', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
    'koi_fpflag_ec', 'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_tce_plnt_num', 'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec',
    'koi_kepmag'
]

# Load saved model
loaded_model = joblib.load("xgb_koi_model.pkl")

# --- HELPER FUNCTIONS ---
def generate_transit_plot(row, predicted_class):
    """Returns Base64 PNG of simulated trapezoid transit plot."""
    depth = float(row["koi_depth"])
    duration = float(row["koi_duration"])
    period = float(row["koi_period"])

    ingress = max(0.1, duration * 0.2)
    total_time = duration * 2
    t = np.linspace(-total_time/2, total_time/2, 500)
    flux = np.ones_like(t)

    ingress_start = -duration/2
    ingress_end = ingress_start + ingress
    egress_end = duration/2
    egress_start = egress_end - ingress

    for i, ti in enumerate(t):
        if ingress_start <= ti < ingress_end:
            flux[i] = 1 - depth * (ti - ingress_start) / ingress
        elif ingress_end <= ti <= egress_start:
            flux[i] = 1 - depth
        elif egress_start < ti <= egress_end:
            flux[i] = 1 - depth * (1 - (ti - egress_start) / ingress)

    plt.figure(figsize=(8,5))
    plt.plot(t, flux, 'b-', lw=2)
    plt.axhline(1, color="gray", linestyle="--", alpha=0.5)
    plt.ylim(1 - depth*1.5, 1 + depth*0.05)
    plt.xlabel("Time (hours)")
    plt.ylabel("Relative Flux")
    plt.title("Transit Shape")

    # Bottom-right info box
    textstr = '\n'.join((
        f"Depth: {depth:.4f}",
        f"Duration: {duration:.2f} h",
        f"Period: {period:.2f} d",
        f"Class: {predicted_class}"
    ))
    plt.gca().text(
        0.95, 0.05, textstr, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8)
    )

    # Save to Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def predict_with_model(df):
    """Predict KOI score and classification for the first row of df."""
    df = df.loc[:, df.columns.intersection(ALLOWED_COLUMNS)]
    X_input = df.drop(columns=['kepid','kepoi_name'])

    sample_instance = X_input.iloc[0].values.reshape(1, -1)
    pred_score = loaded_model.predict(sample_instance)[0]
    predicted_class = "CONFIRMED" if pred_score >= 0.5 else "FALSE POSITIVE"

    plot_b64 = generate_transit_plot(df.iloc[0], predicted_class)

    return {
        "kepid": int(df.iloc[0]["kepid"]),
        "koi_score": float(pred_score),
        "classification": predicted_class,
        "plot_base64": plot_b64
    }

# --- FLASK APP ---
app = Flask(__name__)
CORS(app)

# CSV Upload Endpoint
@app.post("/api/predict_csv")
def predict_csv():
    if 'file' not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400
    try:
        file = request.files['file']
        file.seek(0)
        df = pd.read_csv(io.BytesIO(file.read()))
        result = predict_with_model(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# JSON Input Endpoint
@app.post("/api/predict_json")
def predict_json():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({"error": "Missing 'data' array in JSON"}), 400
        df = pd.DataFrame(data)
        result = predict_with_model(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home route
@app.route("/")
def home():
    return "🚀 Flask server is running!"

# --- RUN FLASK ---
def run_flask():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)

Thread(target=run_flask).start()
