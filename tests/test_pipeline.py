import io
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.etl_pipeline import PlantDataETL  # noqa: E402
from src.ml_engine import PredictiveMaintenanceModel  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'plant_performance_advance_kaggle.csv')
HEAD = "planned_time_hrs,downtime_hrs,target_units,produced_units,defect_units\n"


def _csv(text, name='t.csv'):
    f = io.StringIO(text)
    f.name = name
    return f


class ETLTests(unittest.TestCase):
    def test_oee_formula(self):
        r = PlantDataETL(_csv(HEAD + "8,0.5,400,380,8\n")).load_and_transform().iloc[0]
        self.assertEqual((r.availability_pct, r.performance_pct, r.quality_pct), (93.75, 95.0, 97.89))
        self.assertAlmostEqual(r.oee_pct, 87.18, places=2)

    def test_missing_required_column(self):
        with self.assertRaisesRegex(ValueError, "target_units"):
            PlantDataETL(_csv("planned_time_hrs,downtime_hrs\n8,1\n")).load_and_transform()

    def test_zero_planned_time_is_dropped_not_nan(self):
        etl = PlantDataETL(_csv(HEAD + "0,0,100,50,1\n8,1,400,380,8\n"))
        df = etl.load_and_transform()
        self.assertEqual(len(df), 1)
        self.assertFalse(df['oee_pct'].isna().any())
        self.assertTrue(any("dibuang" in w for w in etl.warnings))

    def test_all_rows_invalid_raises(self):
        with self.assertRaisesRegex(ValueError, "valid"):
            PlantDataETL(_csv(HEAD + "0,0,100,50,1\n")).load_and_transform()

    def test_non_numeric_value_dropped(self):
        df = PlantDataETL(_csv(HEAD + "8,abc,400,380,8\n8,1,400,380,8\n")).load_and_transform()
        self.assertEqual(len(df), 1)

    def test_negative_sensor_blanked(self):
        etl = PlantDataETL(_csv(HEAD.strip() + ",vibration_mm_s,temperature_celsius\n8,1,400,380,8,-0.3,70\n"))
        df = etl.load_and_transform()
        self.assertTrue(np.isnan(df.loc[0, 'vibration_mm_s']))
        self.assertEqual(df.loc[0, 'temperature_celsius'], 70)

    def test_alerts_threshold(self):
        df = PlantDataETL(_csv(HEAD + "8,0.5,400,380,8\n8,3,400,250,30\n")).load_and_transform()
        self.assertEqual(len(PlantDataETL.generate_alerts(df, 80)), 1)
        self.assertEqual(len(PlantDataETL.generate_alerts(df, 20)), 0)

    def test_anomaly_flag_and_exclusion(self):
        etl = PlantDataETL(_csv(HEAD + "8,0.5,400,380,8\n8,9,400,0,0\n8,2,400,300,350\n"))
        df = etl.load_and_transform()
        self.assertEqual(df['data_anomaly'].tolist(), [False, True, True])
        self.assertEqual(len(etl.warnings), 2)
        clean = PlantDataETL.exclude_anomalies(df)
        self.assertEqual(len(clean), 1)
        self.assertAlmostEqual(clean['oee_pct'].mean(), 87.18, places=2)

    def test_default_dataset_has_no_anomaly(self):
        df = PlantDataETL(DATA).load_and_transform()
        self.assertEqual(int(df['data_anomaly'].sum()), 0)
        self.assertEqual(len(PlantDataETL.exclude_anomalies(df)), 270)

    def test_excel_export_has_two_sheets(self):
        df = PlantDataETL(_csv(HEAD + "8,0.5,400,380,8\n")).load_and_transform()
        sheets = pd.read_excel(PlantDataETL.export_to_excel_report(df), sheet_name=None)
        self.assertEqual(set(sheets), {'Transformed_Data', 'KPI_Executive_Summary'})

    def test_default_dataset_matches_readme(self):
        df = PlantDataETL(DATA).load_and_transform()
        self.assertEqual(len(df), 270)
        self.assertAlmostEqual(df['oee_pct'].mean(), 73.51, places=2)
        self.assertEqual(len(PlantDataETL.generate_alerts(df)), 155)


class MLTests(unittest.TestCase):
    def test_requires_sensor_columns(self):
        df = PlantDataETL(_csv(HEAD + "8,1,400,380,8\n")).load_and_transform()
        metrics, msg = PredictiveMaintenanceModel().train(df)
        self.assertIsNone(metrics)
        self.assertIn("sensor", msg)

    def test_missing_label_has_specific_message(self):
        csv = HEAD.strip() + ",vibration_mm_s,temperature_celsius\n8,1,400,380,8,2.0,70\n"
        df = PlantDataETL(_csv(csv)).load_and_transform()
        metrics, msg = PredictiveMaintenanceModel().train(df)
        self.assertIsNone(metrics)
        self.assertIn("failure_risk", msg)
        self.assertNotIn("Kolom sensor tidak ditemukan", msg)

    def test_needs_enough_positives(self):
        df = PlantDataETL(DATA).load_and_transform()
        df['failure_risk'] = 0
        df.loc[0, 'failure_risk'] = 1
        metrics, msg = PredictiveMaintenanceModel().train(df)
        self.assertIsNone(metrics)
        self.assertIn("tidak cukup", msg)

    def test_catches_most_failures(self):
        df = PlantDataETL(DATA).load_and_transform()
        m = PredictiveMaintenanceModel()
        metrics, _ = m.train(df)
        self.assertGreaterEqual(metrics['recall'], 80)
        self.assertGreater(metrics['f1'], 60)
        self.assertAlmostEqual(metrics['tp'] + metrics['fn'], metrics['n_pos'], places=1)
        self.assertEqual(metrics['n_pos'], 9)
        self.assertGreater(m.predict_proba(8.5, 110, 6.5), m.predict_proba(1.5, 70, 0.5))
        self.assertIn("TINGGI", m.predict_risk(8.5, 110, 6.5))
        self.assertIn("NORMAL", m.predict_risk(1.5, 70, 0.5))


if __name__ == '__main__':
    unittest.main()
