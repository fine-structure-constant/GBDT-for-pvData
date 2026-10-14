"""Integration checks on persisted data and validation outputs."""
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'pvdelta_v1'


@unittest.skipUnless((OUT/'oof_predictions.csv').exists(),'Run project first')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(OUT/'cleaned_records.csv')
        cls.pred=pd.read_csv(OUT/'oof_predictions.csv')
        cls.features=pd.read_csv(OUT/'feature_dictionary.csv')

    def test_source_unchanged_and_row_trace(self):
        provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(Path(provenance['source_path']).read_bytes()).hexdigest(),provenance['sha256'])
        self.assertEqual(self.data.source_excel_row.tolist(),list(range(2,len(self.data)+2)))
        self.assertTrue(self.data.record_id.is_unique)

    def test_no_result_leakage_in_features(self):
        forbidden={'pce_after','pce_delta_abs','pce_delta_relative_pct','delta_recomputed_pp','source_excel_row','record_id','pubchem_id','cas_number'}
        self.assertFalse(forbidden & set(self.features.feature))
        self.assertIn('pce_before',set(self.features.feature))

    def test_each_group_in_one_outer_fold(self):
        self.assertEqual(self.pred.groupby('material_pair_group').outer_fold.nunique().max(),1)
        self.assertEqual(set(self.pred.outer_fold),{1,2,3,4,5})
        self.assertEqual(set(self.pred.record_id),set(self.data.loc[self.data.strict_eligible,'record_id']))
        self.assertTrue(self.pred.record_id.is_unique)
        self.assertTrue(np.isfinite(self.pred.prediction_GBDT).all())

    def test_strict_exclusions_and_negative_results(self):
        strict=self.data.loc[self.data.strict_eligible]
        self.assertTrue(strict.exclusion_reasons.isna().all())
        self.assertTrue(strict.canonical_duplicate.eq(0).all())
        self.assertTrue(strict.architecture.isin(['n_i_p','p_i_n']).all())
        self.assertTrue((strict.pce_delta_abs<0).any())
        self.assertFalse(strict.issue_codes.fillna('').str.contains('ABBREVIATION_FORMULA_MISMATCH').any())

    def test_numeric_site_fractions_reconcile(self):
        for site in 'ABX':
            cols=[c for c in self.data if c.startswith(site+'_fraction_')]
            known=self.data[site+'_ratio_known'].eq(1)
            np.testing.assert_allclose(self.data.loc[known,cols].sum(axis=1),1,atol=1e-8)
            self.assertTrue(self.data.loc[~known,cols].isna().all().all())

    def test_target_and_metrics_reconcile(self):
        np.testing.assert_allclose(self.pred.pce_after-self.pred.pce_before,self.pred.pce_delta_abs,atol=1e-6)
        mae=float(np.mean(abs(self.pred.pce_delta_abs-self.pred.prediction_GBDT)))
        summary=json.loads((OUT/'model_summary.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(mae,summary['primary_metrics']['MAE_pp'],places=7)


if __name__=='__main__':
    unittest.main()
