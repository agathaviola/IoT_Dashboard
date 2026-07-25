import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# =====================
# Load dataset
# =====================
df = pd.read_csv("dataset_lstm.csv")

# =====================
# Load model & scaler suhu
# =====================
model = load_model("model_suhu.keras")
scaler = joblib.load("scaler_suhu.pkl")

window = 30

data = df["Suhu"].values.reshape(-1,1)
data = scaler.transform(data)

last_data = data[-window:]

predictions = []

for i in range(1080):   # 6 jam = 1080 data (interval 20 detik)
    x = last_data.reshape(1,window,1)

    pred = model.predict(x, verbose=0)

    predictions.append(pred[0][0])

    last_data = np.append(last_data[1:], pred)

predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1,1)
)

from datetime import datetime, timedelta

waktu_awal = datetime.now()

waktu = [
    waktu_awal + timedelta(seconds=20*i)
    for i in range(len(predictions))
]

print(predictions[:10])
print("Jumlah prediksi:", len(predictions))

hasil = pd.DataFrame({
    "Waktu": waktu,
    "Prediksi": predictions.flatten()
})

hasil.to_csv("forecast_suhu.csv", index=False)
