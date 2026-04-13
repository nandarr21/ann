import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('dataset.csv', sep=';')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── Preprocessing ─────────────────────────────────────────────
# Drop rows with zero values (provinces that didn't exist yet)
df = df[df['persentase_penduduk_miskin'] > 0].copy()
print(f"After filtering zeros: {df.shape}")

# Label encode province names
le = LabelEncoder()
df['provinsi_encoded'] = le.fit_transform(df['nama_provinsi'])

# Features: tahun + kode_provinsi encoded
X = df[['tahun', 'provinsi_encoded']].values
y = df['persentase_penduduk_miskin'].values

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ── Build ANN Model ───────────────────────────────────────────
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
print(f"\nMAE: {mae:.4f}%")
print(f"R² Score: {r2:.4f}")

# ── Save everything ───────────────────────────────────────────
model.save("model.keras")

with open('model/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

with open('model/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save province list for dropdown
provinces = sorted(df['nama_provinsi'].unique().tolist())
tahun_list = sorted(df['tahun'].unique().tolist())

metadata = {
    'provinces': provinces,
    'tahun_min': int(df['tahun'].min()),
    'tahun_max': int(df['tahun'].max()),
    'mae': float(mae),
    'r2': float(r2),
    'total_data': len(df),
    'history': {
        'loss': [float(x) for x in history.history['loss'][-50:]],
        'val_loss': [float(x) for x in history.history['val_loss'][-50:]],
    }
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f)

# Save province-year actual data for chart
chart_data = {}
for prov in provinces:
    prov_df = df[df['nama_provinsi'] == prov][['tahun', 'persentase_penduduk_miskin']].sort_values('tahun')
    chart_data[prov] = {
        'tahun': prov_df['tahun'].tolist(),
        'nilai': prov_df['persentase_penduduk_miskin'].tolist()
    }

with open('model/chart_data.json', 'w') as f:
    json.dump(chart_data, f)

print("\n✅ Model dan metadata berhasil disimpan!")
print(f"Provinsi tersedia: {len(provinces)}")
