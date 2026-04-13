from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
import pickle

app = Flask(__name__)

# ── Load metadata & chart data ────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model/metadata.json')) as f:
    metadata = json.load(f)

with open(os.path.join(BASE, 'model/chart_data.json')) as f:
    chart_data = json.load(f)

# ── Try loading trained model (if available) ──────────────────
model = None
scaler_X = None
scaler_y = None
label_encoder = None

try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    model_path = os.path.join(BASE, 'model/ann_model.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    with open(os.path.join(BASE, 'model/scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(BASE, 'model/scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(BASE, 'model/label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    print("✅ Model ANN berhasil dimuat.")
except Exception as e:
    print(f"⚠️  Model belum tersedia: {e}")
    print("   Jalankan python train_model.py terlebih dahulu.")


def predict_with_model(provinsi: str, tahun: int) -> float:
    """Predict using trained ANN model."""
    if model is None:
        raise RuntimeError("Model belum dilatih.")
    enc = label_encoder.transform([provinsi])[0]
    x = np.array([[tahun, enc]])
    x_scaled = scaler_X.transform(x)
    pred_scaled = model.predict(x_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()[0]
    return float(pred)


def predict_trend_extrapolation(provinsi: str, tahun: int) -> float:
    """
    Fallback: polynomial regression on historical data for a province.
    Used when ANN model hasn't been trained yet.
    """
    data = chart_data.get(provinsi, {})
    if not data:
        return None
    tahun_hist = np.array(data['tahun'])
    nilai_hist = np.array(data['nilai'])
    # Fit polynomial degree 2
    coeffs = np.polyfit(tahun_hist, nilai_hist, 2)
    pred = np.polyval(coeffs, tahun)
    return float(max(0, pred))


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    provinces = metadata['provinces']
    tahun_min = metadata['tahun_min']
    tahun_max = metadata['tahun_max']
    model_status = "ANN (TensorFlow)" if model else "Regresi Polinomial (Fallback)"
    return render_template(
        'index.html',
        provinces=provinces,
        tahun_min=tahun_min,
        tahun_max=tahun_max + 10,
        total_data=metadata['total_data'],
        model_status=model_status,
        model_ready=(model is not None),
        chart_data=chart_data,
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    provinsi = data.get('provinsi', '').strip().upper()
    tahun = int(data.get('tahun', 2025))

    if provinsi not in metadata['provinces']:
        return jsonify({'error': f'Provinsi "{provinsi}" tidak ditemukan.'}), 400

    try:
        if model:
            hasil = predict_with_model(provinsi, tahun)
            metode = "ANN (TensorFlow/Keras)"
        else:
            hasil = predict_trend_extrapolation(provinsi, tahun)
            metode = "Regresi Polinomial"

        # Also return historical data for chart
        hist = chart_data.get(provinsi, {'tahun': [], 'nilai': []})
        return jsonify({
            'provinsi': provinsi,
            'tahun': tahun,
            'prediksi': round(hasil, 2),
            'metode': metode,
            'historis': hist,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chart_data/<provinsi>')
def get_chart_data(provinsi):
    provinsi = provinsi.upper()
    data = chart_data.get(provinsi)
    if not data:
        return jsonify({'error': 'Provinsi tidak ditemukan'}), 404
    return jsonify(data)


@app.route('/all_data')
def all_data():
    """Return summary statistics for all provinces (latest year)."""
    result = []
    for prov, d in chart_data.items():
        if d['nilai']:
            result.append({
                'provinsi': prov,
                'tahun_terakhir': d['tahun'][-1],
                'nilai_terakhir': d['nilai'][-1],
                'trend': round(d['nilai'][-1] - d['nilai'][0], 2)
            })
    result.sort(key=lambda x: x['nilai_terakhir'], reverse=True)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
