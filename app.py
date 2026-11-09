import streamlit as st
import pandas as pd
import plotly.express as px
from src.etl_pipeline import PlantDataETL
from src.ml_engine import PredictiveMaintenanceModel

st.set_page_config(page_title="Enterprise Plant Performance & AI Analytics", layout="wide", page_icon="🏭")


@st.cache_resource(show_spinner=False)
def train_model(data):
    """Dilatih sekali per dataset (bukan setiap interaksi widget) dan memakai data penuh, bukan data terfilter."""
    model = PredictiveMaintenanceModel()
    metrics, msg = model.train(data)
    return model, metrics, msg


st.title("🏭 Plant Performance Intelligence & AI Maintenance System")
st.caption("Arsitektur Portofolio Advance: Gemma Dwi Prasetya | S1 Teknik Informatika (Data Engineering & ML)")

# Sidebar Configuration
st.sidebar.header("📁 Data Source & Ingestion Engine")
data_option = st.sidebar.radio("Pilih Sumber Data:", ["Dataset Default (Kaggle Industrial Log)", "Upload Custom File (Excel/CSV)"])

uploaded_file = None
if data_option == "Upload Custom File (Excel/CSV)":
    uploaded_file = st.sidebar.file_uploader("Unggah File Data Pabrik (.xlsx / .csv)", type=["csv", "xlsx"])

# Load Data Process
try:
    if uploaded_file is not None:
        etl = PlantDataETL(uploaded_file)
        df = etl.load_and_transform()
        st.sidebar.success("File Custom Berhasil Di-ingest & Ditransformasi!")
    else:
        etl = PlantDataETL("data/plant_performance_advance_kaggle.csv")
        df = etl.load_and_transform()
        st.sidebar.info("Menggunakan Dataset Kaggle Default.")
except Exception as e:
    st.error(f"Error Processing Data: {e}")
    st.stop()

for w in etl.warnings:
    st.sidebar.warning(w)

df_all = df  # data penuh (sebelum filter) untuk pelatihan model

# Sidebar Filters
st.sidebar.header("🔍 Dynamic Filter Operations")
n_anomaly = int(df["data_anomaly"].sum()) if "data_anomaly" in df.columns else 0
if n_anomaly:
    if st.sidebar.checkbox(
        f"Kecualikan {n_anomaly} baris anomali dari KPI, alert & grafik",
        value=False,
        help="Baris anomali: downtime melebihi jadwal produksi, atau defect melebihi jumlah produksi.",
    ):
        df = PlantDataETL.exclude_anomalies(df)
if "production_line" in df.columns:
    lines = st.sidebar.multiselect("Pilih Lini Produksi:", options=df["production_line"].unique(), default=df["production_line"].unique())
    df = df[df["production_line"].isin(lines)]

if "shift" in df.columns:
    shifts = st.sidebar.multiselect("Pilih Shift Operasional:", options=df["shift"].unique(), default=df["shift"].unique())
    df = df[df["shift"].isin(shifts)]

if df.empty:
    st.warning("Tidak ada data yang tersisa setelah filter diterapkan.")
    st.stop()

# FITUR TAMBAHAN 1: AUTOMATED ALERT & NOTIFICATION SYSTEM
alerts = PlantDataETL.generate_alerts(df, oee_threshold=80.0)
with st.expander(f"🔔 Notification Center & Anomaly Alert ({len(alerts)} Peringatan)", expanded=bool(alerts)):
    if alerts:
        for alert in alerts[:5]: # Tampilkan 5 alert pertama
            st.warning(alert)
        if len(alerts) > 5:
            st.info(f"Dan {len(alerts) - 5} alert performa rendah lainnya tersimpan di log system.")
    else:
        st.success("✅ Seluruh lini produksi berjalan optimal tanpa anomaly perataan OEE (<80%).")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive OEE & Performance KPI", "🤖 AI Predictive Maintenance (ML)", "📄 Executive Report & Data Explorer"])

# TAB 1: EXECUTIVE DASHBOARD
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall OEE Average", f"{df['oee_pct'].mean():.2f}%", delta=f"{df['oee_pct'].mean() - 85.0:.2f}% vs Target World-Class")
    m2.metric("Total Downtime Accumulation", f"{df['downtime_hrs'].sum():.1f} Jam")
    m3.metric("Availability Index", f"{df['availability_pct'].mean():.2f}%")
    m4.metric("Quality Rate Average", f"{df['quality_pct'].mean():.2f}%")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Tren OEE Harian vs Threshold World Class (85%)")
        fig_oee = px.line(df, x="date" if "date" in df.columns else df.index, y="oee_pct", color="production_line" if "production_line" in df.columns else None, markers=True)
        fig_oee.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Target 85%")
        st.plotly_chart(fig_oee, use_container_width=True)

    with col_chart2:
        st.subheader("📊 Downtime Breakdown per Lini Produksi (Jam)")
        if "production_line" in df.columns:
            fig_down = px.bar(df, x="production_line", y="downtime_hrs", color="shift" if "shift" in df.columns else None, barmode="group")
            st.plotly_chart(fig_down, use_container_width=True)

# TAB 2: MACHINE LEARNING PREDICTIVE MAINTENANCE
with tab2:
    st.header("🤖 Machine Learning Model: Machine Failure Prediction")
    ml, metrics, msg = train_model(df_all)

    if metrics is not None:
        st.success(f"Status Model: {msg}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall (kerusakan tertangkap)", f"{metrics['recall']}%")
        c2.metric("Precision", f"{metrics['precision']}%")
        c3.metric("F1-Score", f"{metrics['f1']}%")
        c4.metric("Akurasi", f"{metrics['accuracy']}%", help=f"Bila model selalu menjawab 'normal', akurasi = {metrics['baseline_accuracy']}%. Gunakan Recall/F1 sebagai ukuran utama.")
        st.caption(
            f"Evaluasi: stratified {metrics['folds']}-fold cross-validation diulang {metrics['repeats']}x (rata-rata) pada "
            f"{metrics['n']} baris ({metrics['n_pos']} kasus risiko tinggi), threshold keputusan {metrics['threshold']}. "
            f"Rata-rata terdeteksi {metrics['tp']:.1f} dari {metrics['n_pos']} kasus; alarm palsu {metrics['fp']:.1f}."
        )
        if metrics['n_pos'] < 20:
            st.info("Jumlah kasus risiko tinggi sangat sedikit, sehingga metrik di atas masih berupa perkiraan kasar.")

        st.subheader("⚡ Simulator Prediksi Risiko Mesin Real-time")
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        vibr = col_sim1.slider("Sensor Getaran (Vibration mm/s)", 0.0, 10.0, 2.5)
        temp = col_sim2.slider("Sensor Suhu (Temperature °C)", 40.0, 120.0, 75.0)
        downt = col_sim3.slider("Histori Downtime (Jam)", 0.0, 8.0, 1.0)

        if st.button("Jalankan Prediksi Risiko Mesin"):
            res = ml.predict_risk(vibr, temp, downt)
            prob = ml.predict_proba(vibr, temp, downt)
            if "TINGGI" in res:
                st.error(f"Hasil Diagnosa ML: {res} (probabilitas {prob * 100:.0f}%)")
            else:
                st.success(f"Hasil Diagnosa ML: {res} (probabilitas {prob * 100:.0f}%)")
    else:
        st.warning(msg)

# TAB 3: DATA EXPLORER & FITUR TAMBAHAN 2 (EXPORT EXCEL REPORT)
with tab3:
    st.subheader("📄 Export Executive Performance Report")
    
    excel_data = PlantDataETL.export_to_excel_report(df)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📊 Download Executive Report (Multi-Sheet Excel)",
            data=excel_data,
            file_name="Plant_Performance_Executive_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col_dl2:
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Raw ETL Data (CSV)", data=csv_download, file_name="transformed_plant_data.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("📋 Transformed Data Explorer")
    st.dataframe(df, use_container_width=True)