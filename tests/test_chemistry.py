import math
import unittest
from pvdelta.chemistry import parse_composition, parse_formula, smiles_fragments, smiles_heavy_counts


class ChemistryTests(unittest.TestCase):
    def test_alias_and_compact_formula(self):
        a=parse_composition('CH3NH3_Pb_I3')
        b=parse_composition('MAPbI3')
        self.assertEqual(a['A_coefficients_json'],b['A_coefficients_json'])
        self.assertEqual(a['composition_topology'],'3D_ABX3')

    def test_mixed_numeric(self):
        a=parse_composition('Cs0.05_FA0.8_MA0.15_Pb_I2.55_Br0.45')
        self.assertAlmostEqual(a['A_fraction_FA'],0.8)
        self.assertAlmostEqual(a['X_fraction_Br'],0.15)
        self.assertEqual(a['composition_topology'],'3D_ABX3')

    def test_unknown_not_half(self):
        a=parse_composition('FA_MA_Pb_I3-x_Clx')
        self.assertEqual(a['A_has_FA'],1)
        self.assertTrue(math.isnan(a['A_fraction_FA']))
        self.assertTrue(math.isnan(a['X_fraction_Cl']))
        self.assertEqual(a['X_has_Cl'],1)

    def test_wrong_stoichiometry_not_repaired(self):
        a=parse_composition('FA0.75_MA0.15_Pb_I3')
        self.assertEqual(a['composition_topology'],'stoichiometry_review')
        self.assertAlmostEqual(a['A_sum'],0.9)

    def test_layered_and_double(self):
        self.assertEqual(parse_composition('PEA2_MA4_Pb5_I16')['composition_topology'],'layered_RP_candidate')
        self.assertEqual(parse_composition('PEA2_MA4_Pb_I16')['composition_topology'],'stoichiometry_review')
        self.assertEqual(parse_composition('Cs2_Ag_Bi_Br6')['composition_topology'],'double_perovskite')

    def test_multiphase_not_single_host(self):
        self.assertEqual(parse_composition('MA_Pb_I3 | FA_Sn_I3')['composition_parse_status'],'multiphase')

    def test_ions(self):
        f=smiles_fragments('C[NH3+].[I-]')
        self.assertEqual([x['fragment_role'] for x in f],['cation','anion'])
        self.assertEqual(smiles_fragments('[Sn+2].[F-].[F-]')[0]['formal_charge'],2)
        self.assertEqual(smiles_fragments('CN.I')[1]['fragment_role'],'neutral')

    def test_formula_charge_and_atoms(self):
        self.assertEqual(parse_formula('Ca+2'),({'Ca':1.0},'parsed',2))
        self.assertEqual(parse_formula('C2H6OS')[0],{'C':2.0,'H':6.0,'O':1.0,'S':1.0})
        self.assertEqual(smiles_heavy_counts('C[NH3+].[I-]'),{'C':1,'N':1,'I':1})


if __name__=='__main__':
    unittest.main()
