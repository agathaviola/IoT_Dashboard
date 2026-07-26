import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard IoT",
    page_icon="🌡️",
    layout="wide"
)

# CSS kustom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Home",
        "📊 Eksplorasi Data",
        "📈 Monitoring",
        "🔮 Forecasting",
        "👥 Tim Pengembang"
    ]
)

# ======================================
# HOME
# ======================================
if menu == "🏠 Home":
    st.markdown('<div class="main-header">🌡️ Dashboard Monitoring & Forecasting IoT</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Selamat Datang 👋
    
    Dashboard ini digunakan untuk memonitor kondisi sensor IoT serta melakukan forecasting menggunakan model LSTM.
    
    ### 📈 Monitoring
    - Melihat data sensor
    - Grafik suhu
    - Grafik kelembaban
    - Statistik data

     ### 🔍 Eksplorasi Data
    Fitur eksplorasi data digunakan untuk memahami karakteristik dan pola dari data sensor sebelum dilakukan proses analisis lebih lanjut. Pada fitur ini pengguna 
    dapat melihat:
    - Informasi dataset
    - Distribusi data sensor
    - Pola perubahan suhu dan kelembaban
    - Hubungan antar variabel sensor
    
    ### 🔮 Forecasting
    - Prediksi suhu 6 jam
    - Prediksi kelembaban 6 jam
    - Evaluasi model LSTM
    
    Silakan pilih menu di sebelah kiri.
    """)

    # ======================================
# EKSPLORASI DATA
# ======================================
elif menu == "📊 Eksplorasi Data":

    st.markdown(
        '<div class="main-header">📊 Eksplorasi Data Sensor IoT</div>',
        unsafe_allow_html=True
    )

    try:
        df = pd.read_csv("dataset_lstm.csv")
        df["Waktu"] = pd.to_datetime(df["Waktu"])

        st.subheader("📋 Dataset")
        st.dataframe(df, use_container_width=True)

        st.divider()

        st.subheader("📌 Informasi Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Jumlah Data", len(df))

        with col2:
            st.metric("Jumlah Kolom", len(df.columns))

        with col3:
            st.metric("Missing Value", df.isnull().sum().sum())

        st.divider()

        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

        st.divider()

        st.subheader("🌡 Distribusi Suhu")

        fig = px.histogram(
            df,
            x="Suhu",
            nbins=30,
            color_discrete_sequence=["royalblue"]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💧 Distribusi Kelembaban")

        fig = px.histogram(
            df,
            x="Kelembaban",
            nbins=30,
            color_discrete_sequence=["green"]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("📦 Boxplot Suhu")

        fig = px.box(df, y="Suhu")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📦 Boxplot Kelembaban")

        fig = px.box(df, y="Kelembaban")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🔥 Korelasi")

        corr = df[["Suhu","Kelembaban"]].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("📈 Scatter Plot")

        fig = px.scatter(
            df,
            x="Suhu",
            y="Kelembaban",
            trendline="ols"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(e)

# ======================================
# MONITORING
# ======================================
elif menu == "📈 Monitoring":
    st.markdown('<div class="main-header">📈 Monitoring Sensor IoT</div>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("sensor_data_terakhir.csv")
        df["Waktu"] = pd.to_datetime(df["Waktu"])
        
        df_suhu = df[df["Topik"]=="tas_ai_surya_fsm_uksw/suhu"].copy()
        df_kel = df[df["Topik"]=="tas_ai_surya_fsm_uksw/kelembaban"].copy()
        
        df_suhu["Nilai"] = df_suhu["Nilai"].astype(float)
        df_kel["Nilai"] = df_kel["Nilai"].astype(float)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Data", len(df))
        
        with col2:
            st.metric(
                "Suhu Terakhir",
                f"{df_suhu['Nilai'].iloc[-1]:.2f} °C"
            )
        
        with col3:
            st.metric(
                "Kelembaban Terakhir",
                f"{df_kel['Nilai'].iloc[-1]:.2f} %"
            )
        
        st.divider()
        
        st.subheader("🌡 Grafik Suhu")
        fig = px.line(df_suhu, x="Waktu", y="Nilai", title="Grafik Suhu")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💧 Grafik Kelembaban")
        fig2 = px.line(df_kel, x="Waktu", y="Nilai", title="Grafik Kelembaban")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("📊 Statistik")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistik Suhu**")
            st.dataframe(df_suhu["Nilai"].describe(), use_container_width=True)
        
        with col2:
            st.write("**Statistik Kelembaban**")
            st.dataframe(df_kel["Nilai"].describe(), use_container_width=True)
        
        st.subheader("📋 Data Sensor")
        st.dataframe(df, use_container_width=True)
        
    except FileNotFoundError:
        st.error("⚠️ File 'sensor_data_terakhir.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")

# ======================================
# FORECASTING - DIPERBAIKI
# ======================================
elif menu == "🔮 Forecasting":
    st.markdown('<div class="main-header">🔮 Forecasting Sensor IoT</div>', unsafe_allow_html=True)
    
    try:
        # ==========================
        # LOAD DATA
        # ==========================
        df = pd.read_csv("dataset_lstm.csv")
        df["Waktu"] = pd.to_datetime(df["Waktu"])
        
        # Pastikan kolom yang dibutuhkan ada
        if 'Suhu' not in df.columns or 'Kelembaban' not in df.columns:
            st.error("⚠️ File 'dataset_lstm.csv' harus memiliki kolom 'Suhu' dan 'Kelembaban'")
            st.stop()
        
        # ==========================
        # PARAMETER FORECASTING
        # ==========================
        st.sidebar.markdown("### ⚙️ Parameter Forecasting")
        forecast_hours = st.sidebar.slider("Jam Prediksi", 1, 12, 6)
        steps = forecast_hours * 12  # 5 menit per step
        
        if st.sidebar.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Memproses forecasting dengan LSTM..."):
                
                # ==========================
                # PREPARE DATA UNTUK LSTM
                # ==========================
                def create_sequences(data, seq_length=60):
                    X, y = [], []
                    for i in range(seq_length, len(data)):
                        X.append(data[i-seq_length:i])
                        y.append(data[i])
                    return np.array(X), np.array(y)
                
                # Scale data
                scaler_temp = MinMaxScaler()
                scaler_hum = MinMaxScaler()
                
                temp_scaled = scaler_temp.fit_transform(df['Suhu'].values.reshape(-1, 1))
                hum_scaled = scaler_hum.fit_transform(df['Kelembaban'].values.reshape(-1, 1))
                
                # Buat sequences
                seq_length = 60
                X_temp, y_temp = create_sequences(temp_scaled.flatten(), seq_length)
                X_hum, y_hum = create_sequences(hum_scaled.flatten(), seq_length)
                
                # Reshape untuk LSTM [samples, timesteps, features]
                X_temp = X_temp.reshape((X_temp.shape[0], X_temp.shape[1], 1))
                X_hum = X_hum.reshape((X_hum.shape[0], X_hum.shape[1], 1))
                
                # ==========================
                # BUILD & TRAIN MODEL LSTM
                # ==========================
                # Model untuk Suhu
                model_temp = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])
                model_temp.compile(optimizer='adam', loss='mse')
                model_temp.fit(X_temp, y_temp, epochs=30, batch_size=32, verbose=0)
                
                # Model untuk Kelembaban
                model_hum = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])
                model_hum.compile(optimizer='adam', loss='mse')
                model_hum.fit(X_hum, y_hum, epochs=30, batch_size=32, verbose=0)
                
                # ==========================
                # GENERATE FORECAST
                # ==========================
                def forecast_future(model, scaler, last_sequence, steps):
                    predictions = []
                    current_seq = last_sequence.copy()
                    
                    for _ in range(steps):
                        input_data = current_seq.reshape(1, seq_length, 1)
                        pred = model.predict(input_data, verbose=0)[0, 0]
                        predictions.append(pred)
                        current_seq = np.roll(current_seq, -1)
                        current_seq[-1] = pred
                    
                    predictions = np.array(predictions).reshape(-1, 1)
                    return scaler.inverse_transform(predictions).flatten()
                
                # Ambil sequence terakhir
                last_seq_temp = temp_scaled[-seq_length:].flatten()
                last_seq_hum = hum_scaled[-seq_length:].flatten()
                
                # Generate forecast
                forecast_temp = forecast_future(model_temp, scaler_temp, last_seq_temp, steps)
                forecast_hum = forecast_future(model_hum, scaler_hum, last_seq_hum, steps)
                
                # ==========================
                # BUAT DATAFRAME HASIL
                # ==========================
                last_timestamp = df['Waktu'].iloc[-1]
                forecast_timestamps = pd.date_range(
                    start=last_timestamp + pd.Timedelta(minutes=5),
                    periods=steps,
                    freq='5min'
                )
                
                hasil = pd.DataFrame({
                    'Waktu': forecast_timestamps,
                    'Prediksi Suhu': forecast_temp,
                    'Prediksi Kelembaban': forecast_hum
                })
                
                # Simpan ke session state
                st.session_state.forecast_hasil = hasil
                st.session_state.forecast_df = df
                st.session_state.forecast_steps = steps
                
                st.success("✅ Forecasting berhasil dihasilkan!")
                st.rerun()
        
        # ==========================
        # TAMPILKAN HASIL FORECAST
        # ==========================
        if 'forecast_hasil' in st.session_state:
            hasil = st.session_state.forecast_hasil
            df = st.session_state.forecast_df
            
            # Metric
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Prediksi Suhu Terakhir",
                    f"{hasil['Prediksi Suhu'].iloc[-1]:.2f} °C"
                )
            
            with col2:
                st.metric(
                    "Prediksi Kelembaban Terakhir",
                    f"{hasil['Prediksi Kelembaban'].iloc[-1]:.2f} %"
                )
            
            st.divider()
            
            # ==========================
            # GRAFIK SUHU
            # ==========================
            st.subheader("🌡️ Aktual vs Prediksi Suhu")
            
            # Ambil data aktual untuk periode yang sama
            aktual_suhu = df.tail(len(hasil))[['Waktu', 'Suhu']].copy()
            aktual_suhu.columns = ['Waktu', 'Nilai']
            aktual_suhu['Jenis'] = 'Aktual'
            
            pred_suhu = hasil[['Waktu', 'Prediksi Suhu']].copy()
            pred_suhu.columns = ['Waktu', 'Nilai']
            pred_suhu['Jenis'] = 'Prediksi'
            
            gabung_suhu = pd.concat([aktual_suhu, pred_suhu])
            
            fig_suhu = px.line(
                gabung_suhu,
                x='Waktu',
                y='Nilai',
                color='Jenis',
                title='Perbandingan Suhu Aktual vs Prediksi',
                markers=True,
                color_discrete_map={'Aktual': 'blue', 'Prediksi': 'red'}
            )
            fig_suhu.update_layout(
                xaxis_title='Waktu',
                yaxis_title='Suhu (°C)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_suhu, use_container_width=True)
            
            # ==========================
            # GRAFIK KELEMBABAN
            # ==========================
            st.subheader("💧 Aktual vs Prediksi Kelembaban")
            
            aktual_kel = df.tail(len(hasil))[['Waktu', 'Kelembaban']].copy()
            aktual_kel.columns = ['Waktu', 'Nilai']
            aktual_kel['Jenis'] = 'Aktual'
            
            pred_kel = hasil[['Waktu', 'Prediksi Kelembaban']].copy()
            pred_kel.columns = ['Waktu', 'Nilai']
            pred_kel['Jenis'] = 'Prediksi'
            
            gabung_kel = pd.concat([aktual_kel, pred_kel])
            
            fig_kel = px.line(
                gabung_kel,
                x='Waktu',
                y='Nilai',
                color='Jenis',
                title='Perbandingan Kelembaban Aktual vs Prediksi',
                markers=True,
                color_discrete_map={'Aktual': 'green', 'Prediksi': 'orange'}
            )
            fig_kel.update_layout(
                xaxis_title='Waktu',
                yaxis_title='Kelembaban (%)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_kel, use_container_width=True)
            
            st.divider()
            
            # ==========================
            # GRAFIK GABUNGAN
            # ==========================
            st.subheader("📊 Grafik Gabungan Suhu & Kelembaban")
            
            fig_gabungan = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Suhu', 'Kelembaban'),
                shared_xaxes=True,
                vertical_spacing=0.15
            )
            
            # Suhu
            fig_gabungan.add_trace(
                go.Scatter(
                    x=aktual_suhu['Waktu'], y=aktual_suhu['Nilai'],
                    name='Aktual Suhu', line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            fig_gabungan.add_trace(
                go.Scatter(
                    x=pred_suhu['Waktu'], y=pred_suhu['Nilai'],
                    name='Prediksi Suhu', line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Kelembaban
            fig_gabungan.add_trace(
                go.Scatter(
                    x=aktual_kel['Waktu'], y=aktual_kel['Nilai'],
                    name='Aktual Kelembaban', line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            fig_gabungan.add_trace(
                go.Scatter(
                    x=pred_kel['Waktu'], y=pred_kel['Nilai'],
                    name='Prediksi Kelembaban', line=dict(color='orange', width=2, dash='dash')
                ),
                row=2, col=1
            )
            
            fig_gabungan.update_layout(height=600, hovermode='x unified')
            fig_gabungan.update_xaxes(title_text='Waktu', row=2, col=1)
            fig_gabungan.update_yaxes(title_text='Suhu (°C)', row=1, col=1)
            fig_gabungan.update_yaxes(title_text='Kelembaban (%)', row=2, col=1)
            
            st.plotly_chart(fig_gabungan, use_container_width=True)
            
            st.divider()
            
            # ==========================
            # TABEL FORECAST
            # ==========================
            st.subheader("📋 Tabel Hasil Forecast")
            st.dataframe(hasil, use_container_width=True, hide_index=True)
            
            # ==========================
            # EVALUASI MODEL
            # ==========================
            st.divider()
            st.subheader("📊 Evaluasi Model")
            
            # Ambil data aktual sesuai dengan periode forecast
            aktual_suhu_nilai = df['Suhu'].tail(len(hasil)).values
            prediksi_suhu_nilai = hasil['Prediksi Suhu'].values
            
            aktual_kel_nilai = df['Kelembaban'].tail(len(hasil)).values
            prediksi_kel_nilai = hasil['Prediksi Kelembaban'].values
            
            # Hitung metrik
            rmse_suhu = np.sqrt(mean_squared_error(aktual_suhu_nilai, prediksi_suhu_nilai))
            mae_suhu = mean_absolute_error(aktual_suhu_nilai, prediksi_suhu_nilai)
            mape_suhu = np.mean(np.abs((aktual_suhu_nilai - prediksi_suhu_nilai) / 
                                       (aktual_suhu_nilai + 1e-10))) * 100
            
            rmse_kel = np.sqrt(mean_squared_error(aktual_kel_nilai, prediksi_kel_nilai))
            mae_kel = mean_absolute_error(aktual_kel_nilai, prediksi_kel_nilai)
            mape_kel = np.mean(np.abs((aktual_kel_nilai - prediksi_kel_nilai) / 
                                      (aktual_kel_nilai + 1e-10))) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RMSE Suhu", f"{rmse_suhu:.3f}")
            with col2:
                st.metric("MAE Suhu", f"{mae_suhu:.3f}")
            with col3:
                st.metric("MAPE Suhu", f"{mape_suhu:.2f}%")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("RMSE Kelembaban", f"{rmse_kel:.3f}")
            with col5:
                st.metric("MAE Kelembaban", f"{mae_kel:.3f}")
            with col6:
                st.metric("MAPE Kelembaban", f"{mape_kel:.2f}%")
            
            st.divider()
            
            # ==========================
            # DOWNLOAD
            # ==========================
            csv = hasil.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Hasil Forecast (CSV)",
                data=csv,
                file_name="hasil_forecast.csv",
                mime="text/csv"
            )
        
        else:
            st.info("💡 Klik tombol 'Generate Forecast' di sidebar untuk memulai prediksi")
            
            # Tampilkan data historis terakhir
            st.subheader("📊 Data Historis Terakhir")
            st.dataframe(df.tail(20), use_container_width=True)
            
            # Plot data historis
            st.subheader("📈 Data Historis Suhu & Kelembaban")
            
            fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_hist.add_trace(
                go.Scatter(x=df['Waktu'], y=df['Suhu'], 
                          name='Suhu', line=dict(color='blue', width=2)),
                secondary_y=False
            )
            fig_hist.add_trace(
                go.Scatter(x=df['Waktu'], y=df['Kelembaban'], 
                          name='Kelembaban', line=dict(color='green', width=2)),
                secondary_y=True
            )
            
            fig_hist.update_layout(
                title='Data Historis Sensor',
                xaxis_title='Waktu',
                hovermode='x unified',
                height=400
            )
            fig_hist.update_yaxes(title_text='Suhu (°C)', secondary_y=False)
            fig_hist.update_yaxes(title_text='Kelembaban (%)', secondary_y=True)
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {e}")
        st.info("Pastikan file 'dataset_lstm.csv' ada di direktori yang sama.")
    except Exception as e:
        st.error(f"⚠️ Terjadi error: {e}")
        st.info("Pastikan data memiliki format yang benar.")

# ======================================
# TIM PENGEMBANG
# ======================================
elif menu == "👥 Tim Pengembang":
    st.markdown('<div class="main-header">👥 Tim Pengembang</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Dashboard Monitoring & Forecasting IoT
    
    Dashboard ini dikembangkan sebagai tugas mata kuliah Artificial Intelligence.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            st.image("images/hana.jpg.png", width=180)
        except:
            st.info("📷 Foto Hana tidak ditemukan")
        st.write("### Hana Rahmawati")
        st.write("**NIM:** 662023006")
        st.write("**Program Studi:** Matematika")
    
    with col2:
        try:
            st.image("images/vivi.jpg.png", width=180)
        except:
            st.info("📷 Foto Vivi tidak ditemukan")
        st.write("### Agatha Viola Amanda Febyani")
        st.write("**NIM:** 662023007")
        st.write("**Program Studi:** Matematika")
    
    st.divider()
    
    st.subheader("📝 Deskripsi")
    st.write("""
    Dashboard ini digunakan untuk memonitor data sensor suhu dan kelembaban secara real-time
    serta melakukan prediksi kondisi sensor menggunakan metode **Long Short-Term Memory (LSTM)**.

    """)
