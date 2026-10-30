import pandas as pd
import numpy as np
import io

class PlantDataETL:
    def __init__(self, data_source):
        self.data_source = data_source

    def load_and_transform(self):
        # Multi-format ingestion (Excel / CSV)
        if isinstance(self.data_source, str):
            if self.data_source.endswith('.xlsx') or self.data_source.endswith('.xls'):
                df = pd.read_excel(self.data_source)
            else:
                df = pd.read_csv(self.data_source)
        else:
            if self.data_source.name.endswith('.xlsx') or self.data_source.name.endswith('.xls'):
                df = pd.read_excel(self.data_source)
            else:
                df = pd.read_csv(self.data_source)

        # Standardizing Columns
        required_cols = ['planned_time_hrs', 'downtime_hrs', 'target_units', 'produced_units', 'defect_units']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Kolom wajib '{col}' tidak ditemukan pada file.")

        # Data Cleansing & KPI Engineering
        df['operating_time_hrs'] = np.maximum(0, df['planned_time_hrs'] - df['downtime_hrs'])
        df['good_units'] = np.maximum(0, df['produced_units'] - df['defect_units'])

        # OEE Calculations
        df['availability_pct'] = np.round((df['operating_time_hrs'] / df['planned_time_hrs']) * 100, 2)
        df['performance_pct'] = np.round((df['produced_units'] / df['target_units']) * 100, 2)
        df['quality_pct'] = np.where(df['produced_units'] > 0, np.round((df['good_units'] / df['produced_units']) * 100, 2), 0)
        df['oee_pct'] = np.round((df['availability_pct'] * df['performance_pct'] * df['quality_pct']) / 10000, 2)

        return df

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