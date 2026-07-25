import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# ============================
# Membaca dataset
# ============================
# Membaca dataset
df = pd.read_csv("dataset_lstm.csv")

# Cek apakah ada data kosong
print(df.isnull().sum())

# Pastikan kolom berupa angka
df["Suhu"] = pd.to_numeric(df["Suhu"], errors="coerce")
df["Kelembaban"] = pd.to_numeric(df["Kelembaban"], errors="coerce")

# Hapus data yang kosong
df = df.dropna()

print("Jumlah data setelah dibersihkan:", len(df))

# Mengambil data suhu
data = df["Suhu"].values.reshape(-1,1)

print("Jumlah data:", len(df))

print(df.head())

print(df.dtypes)

print(df["Suhu"].describe())

print(df["Suhu"].isnull().sum())

print("Ada nilai inf:", np.isinf(df["Suhu"]).sum())

# Mengambil data suhu
data = df["Suhu"].astype("float32").values.reshape(-1,1)

# Normalisasi
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print("Nilai minimum :", data.min())
print("Nilai maksimum :", data.max())
print("Jumlah NaN :", np.isnan(data).sum())

# Menyimpan scaler
joblib.dump(scaler,"scaler_suhu.pkl")

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

model = Sequential()

model.add(
    LSTM(
        32,
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

print("Shape X :", X.shape)
print("Shape y :", y.shape)

print("Ada NaN di X :", np.isnan(X).sum())
print("Ada NaN di y :", np.isnan(y).sum())

print("Nilai minimum :", X.min())
print("Nilai maksimum :", X.max())

print("NaN pada X :", np.isnan(X).sum())
print("NaN pada y :", np.isnan(y).sum())

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    verbose=1
)

model.save("model_suhu.keras")

print("Model suhu berhasil disimpan!")
