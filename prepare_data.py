import pandas as pd

# Membaca data
df = pd.read_csv("sensor_data_terakhir.csv")

# Mengubah menjadi format lebar (pivot)
df_baru = df.pivot_table(
    index="Waktu",
    columns="Topik",
    values="Nilai",
    aggfunc="mean"
).reset_index()

# Mengganti nama kolom
df_baru.columns = [
    "Waktu",
    "Kelembaban",
    "Suhu"
]

# Mengurutkan kolom
df_baru = df_baru[
    ["Waktu", "Suhu", "Kelembaban"]
]

# Simpan hasil
df_baru.to_csv(
    "dataset_lstm.csv",
    index=False
)

print("Dataset berhasil dibuat!")
