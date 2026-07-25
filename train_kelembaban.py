import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# ============================
# Membaca dataset
# ============================
df = pd.read_csv("dataset_lstm.csv")

# Mengambil data kelembaban
data = df["Kelembaban"].values.reshape(-1,1)

# Normalisasi
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Menyimpan scaler
joblib.dump(scaler,"scaler_kelembaban.pkl")

# ============================
# Membuat data sequence
# ============================

X=[]
y=[]

window=30

for i in range(window,len(data)):
    X.append(data[i-window:i])
    y.append(data[i])

X=np.array(X)
y=np.array(y)

# ============================
# Model LSTM
# ============================

model=Sequential()

model.add(
    LSTM(
        50,
        activation="relu",
        input_shape=(window,1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

# ============================
# Training
# ============================

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)

model.save("model_kelembaban.keras")

print("Model kelembaban berhasil disimpan!")
