# 🏭 Plant Performance Intelligence & AI Maintenance System

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Deplolyed on Biznet Gio](https://img.shields.io/badge/Cloud-Biznet_Gio_VPS-orange.svg)](https://biznetgio.com)

**Live Demo Dashboard:** [http://IP_PUBLIC_VPS_KAMU](http://IP_PUBLIC_VPS_KAMU)

---

## 📌 Executive Summary
Sistem ini dirancang untuk mengotomatisasi pemrosesan data kinerja operasional pabrik (*Data Pipeline/ETL*), memvisualisasikan *Overall Equipment Effectiveness* (OEE), serta memprediksi risiko kerusakan mesin berbasis sensor (*Predictive Maintenance*) menggunakan Machine Learning.

Project ini dikembangkan sebagai portofolio kualifikasi **Plant Performance Administrator - PT Semen Indonesia (Persero) Tbk.**

---

## 🏗️ System Architecture Flow

```text
[ Raw Data Logs ] ────────┐
 (CSV / Excel)            │
                          ▼
             ┌─────────────────────────┐
             │   Multi-Format ETL Engine│ (Python & Pandas)
             └────────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌────────────────────┐
│ AI / ML Engine   │           │ Interactive Streamlit│
│ (Random Forest)  │           │ Analytics Dashboard│
└────────┬─────────┘           └─────────┬──────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
             ┌─────────────────────────┐
             │ Docker Container & VPS  │ (Biznet Gio + Nginx)
             └─────────────────────────┘

✨ Key Features & Capabilities

    Robust Multi-Source Ingestion: Mendukung pembacaan log sensor otomatis dari CSV & Dynamic Custom Excel.

    Automated KPI & OEE Calculation: Mengkalkulasi nilai Availability, Performance, Quality Rate, dan Overall OEE secara real-time.

    Machine Learning Predictive Maintenance: Model Random Forest Classifier untuk klasifikasi anomali getaran & suhu mesin.

    Automated Alert & Notification System: Mendeteksi penurunan performa OEE harian (< 80%) secara otomatis.

    Executive Excel Reporting: Ekspor laporan terstruktur dengan format multi-sheet spreadsheet.

    🛠️ Tech Stack & Dependencies

    Language & Core: Python 3.10, Pandas, NumPy, Scikit-Learn

    Visualization & UI: Streamlit, Plotly Express

    Infrastructure & DevOps: Docker, Docker Compose, Nginx, Biznet Gio NEO Virtual Compute

🚀 Quick Start (Local Setup)
Bash

# 1. Clone Repository
git clone [https://github.com/username_kamu/plant-intelligence-system.git](https://github.com/username_kamu/plant-intelligence-system.git)
cd plant-intelligence-system

# 2. Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate  # atau .\venv\Scripts\activate di Windows
pip install -r requirements.txt

# 3. Run Streamlit Dashboard
streamlit run app.py

Developer: Gemma Dwi Prasetya

Education: S1 Teknik Informatika - Universitas Negeri Surabaya (IPK 3.77)

Role: Backend Developer & Data Engineer