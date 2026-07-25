import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("dataset_lstm.csv")

# Load model & scaler
model = load_model("model_kelembaban.keras")
scaler = joblib.load("scaler_kelembaban.pkl")

window = 30

data = df["Kelembaban"].values.reshape(-1,1)
data = scaler.transform(data)

last_data = data[-window:]

predictions = []

for i in range(1080):      # 6 jam, interval 20 detik
    x = last_data.reshape(1, window, 1)

    pred = model.predict(x, verbose=0)

    predictions.append(pred[0][0])

    last_data = np.vstack([last_data[1:], pred])

predictions = scaler.inverse_transform(
    np.array(predictions).reshape(-1,1)
)

hasil = pd.DataFrame({
    "Prediksi Kelembaban": predictions.flatten()
})

hasil.to_csv("forecast_kelembaban.csv", index=False)

print("Forecast kelembaban selesai!")
