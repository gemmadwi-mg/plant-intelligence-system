from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class PredictiveMaintenanceModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, df):
        if 'vibration_mm_s' not in df.columns or 'temperature_celsius' not in df.columns or 'failure_risk' not in df.columns:
            return None, "Dataset tidak memiliki parameter sensor yang dibutuhkan untuk ML."

        X = df[['vibration_mm_s', 'temperature_celsius', 'downtime_hrs']]
        y = df['failure_risk']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.is_trained = True
        return round(acc * 100, 2), "Model Berhasil Dilatih."

    def predict_risk(self, vibration, temp, downtime):
        if not self.is_trained:
            return "Model Belum Dilatih"
        input_data = pd.DataFrame([[vibration, temp, downtime]], columns=['vibration_mm_s', 'temperature_celsius', 'downtime_hrs'])
        prediction = self.model.predict(input_data)[0]
        return "⚠️ RISIKO BREAKDOWN TINGGI" if prediction == 1 else "✅ KONDISI MESIN NORMAL"