import io

import numpy as np
import pandas as pd

REQUIRED_COLS = ['planned_time_hrs', 'downtime_hrs', 'target_units', 'produced_units', 'defect_units']


class PlantDataETL:
    def __init__(self, data_source):
        self.data_source = data_source
        self.warnings = []  # catatan kualitas data dari proses load_and_transform terakhir

    def _read(self):
        # Multi-format ingestion (Excel / CSV), path string atau objek upload
        src = self.data_source
        name = src if isinstance(src, str) else getattr(src, 'name', '')
        if str(name).lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(src)
        return pd.read_csv(src)

    def load_and_transform(self):
        df = self._read()
        self.warnings = []

        for col in REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"Kolom wajib '{col}' tidak ditemukan pada file.")

        # Validasi: paksa numerik, lalu buang baris yang tidak bisa dihitung secara sah
        for col in REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        invalid = (
            df[REQUIRED_COLS].isna().any(axis=1)
            | (df['planned_time_hrs'] <= 0)
            | (df['target_units'] <= 0)
            | (df['downtime_hrs'] < 0)
            | (df['produced_units'] < 0)
            | (df['defect_units'] < 0)
        )
        n_invalid = int(invalid.sum())
        if n_invalid:
            self.warnings.append(
                f"{n_invalid} baris dibuang karena tidak valid (nilai kosong/non-numerik, "
                f"planned_time atau target <= 0, atau nilai negatif)."
            )
            df = df[~invalid].reset_index(drop=True)
        if df.empty:
            raise ValueError("Tidak ada baris data yang valid setelah proses validasi.")

        # Baris yang lolos validasi tetapi datanya tidak konsisten secara logika. Tidak dibuang (agar tetap
        # terlihat), melainkan ditandai di kolom 'data_anomaly' sehingga bisa dikecualikan dari KPI.
        over_downtime = df['downtime_hrs'] > df['planned_time_hrs']
        over_defect = df['defect_units'] > df['produced_units']
        df['data_anomaly'] = over_downtime | over_defect
        if over_downtime.any():
            self.warnings.append(f"{int(over_downtime.sum())} baris memiliki downtime_hrs > planned_time_hrs (availability dibatasi ke 0; ditandai data_anomaly).")
        if over_defect.any():
            self.warnings.append(f"{int(over_defect.sum())} baris memiliki defect_units > produced_units (good_units dibatasi ke 0; ditandai data_anomaly).")

        # Sensor tidak mungkin bernilai fisik negatif: tandai sebagai kosong (NaN), bukan dibuang
        if 'vibration_mm_s' in df.columns:
            df['vibration_mm_s'] = pd.to_numeric(df['vibration_mm_s'], errors='coerce')
            bad = int((df['vibration_mm_s'] < 0).sum())
            if bad:
                df.loc[df['vibration_mm_s'] < 0, 'vibration_mm_s'] = np.nan
                self.warnings.append(f"{bad} nilai vibration_mm_s negatif dianggap error sensor (dikosongkan).")
        if 'temperature_celsius' in df.columns:
            df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'], errors='coerce')
            bad = int((df['temperature_celsius'] <= 0).sum())
            if bad:
                df.loc[df['temperature_celsius'] <= 0, 'temperature_celsius'] = np.nan
                self.warnings.append(f"{bad} nilai temperature_celsius <= 0 dianggap error sensor (dikosongkan).")

        # Data Cleansing & KPI Engineering
        df['operating_time_hrs'] = np.maximum(0, df['planned_time_hrs'] - df['downtime_hrs'])
        df['good_units'] = np.maximum(0, df['produced_units'] - df['defect_units'])

        # OEE Calculations
        df['availability_pct'] = np.round((df['operating_time_hrs'] / df['planned_time_hrs']) * 100, 2)
        df['performance_pct'] = np.round((df['produced_units'] / df['target_units']) * 100, 2)
        df['quality_pct'] = np.where(df['produced_units'] > 0, np.round((df['good_units'] / df['produced_units']) * 100, 2), 0)
        df['oee_pct'] = np.round((df['availability_pct'] * df['performance_pct'] * df['quality_pct']) / 10000, 2)

        n_over = int((df['oee_pct'] > 100).sum())
        if n_over:
            self.warnings.append(f"{n_over} baris memiliki OEE > 100% (produksi melebihi target); periksa nilai target_units.")

        return df

    @staticmethod
    def exclude_anomalies(df):
        """Membuang baris yang ditandai data_anomaly (data tidak konsisten) dari DataFrame hasil transformasi."""
        if 'data_anomaly' not in df.columns:
            return df
        return df[~df['data_anomaly']]

    @staticmethod
    def generate_alerts(df, oee_threshold=80.0):
        """Membuat notifikasi alert otomatis jika OEE di bawah threshold atau downtime tinggi"""
        alerts = []
        low_oee_rows = df[df['oee_pct'] < oee_threshold]

        for idx, row in low_oee_rows.iterrows():
            line = row.get('production_line', f'Row {idx}')
            shift = row.get('shift', 'N/A')
            date_val = row.get('date', f'Index {idx}')
            alerts.append(f"⚠️ [ALERT PERFORMA] Tanggal {date_val} | {line} | {shift} -> OEE Kritis: {row['oee_pct']}% (Downtime: {row['downtime_hrs']} Jam)")

        return alerts

    @staticmethod
    def export_to_excel_report(df):
        """Menghasilkan file Excel Laporan Eksekutif dengan Multiple Sheets"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Data Transformed
            df.to_excel(writer, sheet_name='Transformed_Data', index=False)

            # Sheet 2: Summary KPI
            summary = pd.DataFrame({
                'Metric': ['Average OEE (%)', 'Total Downtime (Hrs)', 'Avg Availability (%)', 'Avg Quality (%)'],
                'Value': [df['oee_pct'].mean(), df['downtime_hrs'].sum(), df['availability_pct'].mean(), df['quality_pct'].mean()]
            })
            summary.to_excel(writer, sheet_name='KPI_Executive_Summary', index=False)

        output.seek(0)
        return output
