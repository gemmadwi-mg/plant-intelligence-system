import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURES = ['vibration_mm_s', 'temperature_celsius', 'downtime_hrs']

# Threshold keputusan 0.3 (bukan 0.5): pada predictive maintenance, kerusakan yang terlewat
# jauh lebih mahal daripada alarm palsu, jadi model dibuat lebih sensitif.
DEFAULT_THRESHOLD = 0.3

# Cross-validation diulang dengan beberapa seed dan dirata-rata, supaya angka tidak bergantung
# pada satu pembagian data yang kebetulan bagus atau buruk.
REPEATS = 5


def _new_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)


class PredictiveMaintenanceModel:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.model = _new_forest()
        self.threshold = threshold
        self.is_trained = False

    def train(self, df):
        """Evaluasi dengan repeated stratified k-fold cross-validation, lalu latih model final pada seluruh data.

        Kelas risiko sangat jarang (sekitar 3%), sehingga satu split train/test akan berisi hanya
        beberapa kasus positif dan hasilnya sangat tidak stabil. Cross-validation menilai setiap baris
        data tepat satu kali sebagai data uji.

        Return: (metrics: dict | None, pesan: str)
        """
        missing_sensors = [c for c in FEATURES if c not in df.columns]
        if missing_sensors:
            return None, ("Kolom sensor tidak ditemukan: " + ", ".join(missing_sensors) +
                          ". Tab ini membutuhkan kolom vibration_mm_s, temperature_celsius, dan failure_risk.")
        if 'failure_risk' not in df.columns:
            return None, ("Kolom label 'failure_risk' (0 = normal, 1 = risiko tinggi) tidak ditemukan, sehingga model "
                          "tidak bisa dilatih. Kolom sensor sudah tersedia; tambahkan label dari riwayat kerusakan mesin.")
        needed = FEATURES + ['failure_risk']

        data = df[needed].dropna()
        X = data[FEATURES]
        y = (data['failure_risk'] > 0).astype(int)

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if min(n_pos, n_neg) < 2:
            return None, (f"Data tidak cukup untuk melatih model: ditemukan {n_pos} baris risiko tinggi dan "
                          f"{n_neg} baris normal (minimal 2 per kelas).")

        k = min(5, n_pos, n_neg)
        runs = []
        for rep in range(REPEATS):
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
            proba = cross_val_predict(_new_forest(), X, y, cv=cv, method='predict_proba')[:, 1]
            pred = (proba >= self.threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            runs.append([precision, recall, f1, (pred == y).mean(), tp, fp, fn, tn])
        precision, recall, f1, acc, tp, fp, fn, tn = np.mean(runs, axis=0)

        self.model.fit(X, y)
        self.is_trained = True

        metrics = {
            'precision': round(float(precision) * 100, 2),
            'recall': round(float(recall) * 100, 2),
            'f1': round(float(f1) * 100, 2),
            'accuracy': round(float(acc) * 100, 2),
            'baseline_accuracy': round(float(1 - y.mean()) * 100, 2),  # akurasi bila selalu menjawab "normal"
            'tp': round(float(tp), 1), 'fp': round(float(fp), 1), 'fn': round(float(fn), 1), 'tn': round(float(tn), 1),
            'n': int(len(y)), 'n_pos': n_pos, 'folds': k, 'repeats': REPEATS, 'threshold': self.threshold,
        }
        return metrics, "Model Berhasil Dilatih."

    def predict_proba(self, vibration, temp, downtime):
        """Probabilitas risiko breakdown (0.0 - 1.0)."""
        if not self.is_trained:
            raise RuntimeError("Model belum dilatih.")
        row = pd.DataFrame([[vibration, temp, downtime]], columns=FEATURES)
        return float(self.model.predict_proba(row)[0, 1])

    def predict_risk(self, vibration, temp, downtime):
        if not self.is_trained:
            return "Model Belum Dilatih"
        p = self.predict_proba(vibration, temp, downtime)
        return "⚠️ RISIKO BREAKDOWN TINGGI" if p >= self.threshold else "✅ KONDISI MESIN NORMAL"
