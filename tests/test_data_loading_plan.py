import unittest
from unittest.mock import MagicMock
from fedbiomed.common.data.data_loading_plan import DataLoadingPlan, DataLoadingPlanMixin, MapperDP
from tests.testsupport.testing_data_pipeline import PipelineForTesting


class TestDataPipeline(unittest.TestCase):
    changed_data = {'my': 'different-data'}

    def test_data_pipeline_01_equality(self):
        dp1 = PipelineForTesting()
        dp2 = PipelineForTesting()
        dp3 = MapperDP()
        dp3.type_id = 'testing-mapper'
        self.assertEqual(dp1, dp2)
        self.assertEqual(dp1, 'pipeline_for_testing')
        self.assertEqual(dp3, 'testing-mapper')
        self.assertNotEqual(dp1, dp3)
        self.assertNotEqual(dp2, dp3)
        dp2.data = self.changed_data
        self.assertEqual(dp1, dp2)  # Equality is based only on type_id, not data!
        self.assertFalse(dp1.data == dp2.data)

    def test_data_pipeline_02_serialize_and_load(self):
        dp1 = PipelineForTesting()
        dp2 = PipelineForTesting()
        dp2.data = self.changed_data
        self.assertFalse(dp1.data == dp2.data)
        serialized = dp1.serialize()
        dp2.load(serialized)
        self.assertDictEqual(dp1.data, dp2.data)

        exec(f"import {serialized['pipeline_module']}")
        dp3 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}()")
        dp3.load(serialized)
        self.assertEqual(dp1.type_id, dp3.type_id)
        self.assertDictEqual(dp1.data, dp3.data)

        dp4 = MapperDP()
        dp4.type_id = 'testing-mapper'
        self.assertEqual(dp4, 'testing-mapper')
        serialized = dp4.serialize()
        exec(f"import {serialized['pipeline_module']}")
        dp5 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}()")
        dp5.load(serialized)
        self.assertEqual('testing-mapper', dp5)
        dp5 = dp5.load(serialized)
        self.assertEqual('testing-mapper', dp5)

    def test_data_pipeline_03_apply(self):
        dp1 = PipelineForTesting()
        dp2 = PipelineForTesting()
        dp2.data = self.changed_data
        dp3 = MapperDP()
        dp3.type_id = 'testing-mapper'
        dp3.map = self.changed_data

        apply_1 = dp1.apply()
        self.assertEqual(len(apply_1), 1)
        self.assertIn('data', apply_1)
        apply_2 = dp2.apply()
        self.assertEqual(len(apply_2), 1)
        self.assertIn('different-data', apply_2)
        apply_3 = dp3.apply('my')
        self.assertEqual(apply_3, 'different-data')


class TestDataLoadingPlan(unittest.TestCase):
    def setUp(self):
        self.dp1 = PipelineForTesting()
        self.dp2 = PipelineForTesting()
        self.assertDictEqual(self.dp1.data, self.dp2.data)
        self.dp2.type_id = 'other_testing_pipeline'
        self.dp2.data = TestDataPipeline.changed_data

    def test_data_loading_plan_01_list_and_str_interface(self):
        dlp = DataLoadingPlan()
        dlp.append(self.dp1)
        dlp.append(self.dp2)
        self.assertIn(self.dp1, dlp)
        self.assertIn('pipeline_for_testing', dlp)
        self.assertIn(self.dp2, dlp)
        self.assertIn('other_testing_pipeline', dlp)
        self.assertEqual(self.dp1, dlp['pipeline_for_testing'])
        self.assertEqual(self.dp2, dlp['other_testing_pipeline'])
        self.assertEqual(self.dp1, dlp[0])
        self.assertEqual(self.dp2, dlp[1])

        it = dlp.__iter__()
        first = next(it)
        self.assertEqual(first, self.dp1)
        second = next(it)
        self.assertEqual(second, self.dp2)

        str_repr = str(dlp)
        self.assertIn(dlp.dlp_id, str_repr)
        self.assertIn(self.dp1.type_id, str_repr)
        self.assertIn(self.dp2.type_id, str_repr)

    def test_data_loading_plan_02_serialize_and_load(self):
        dlp = DataLoadingPlan()
        dlp.append(self.dp1)
        dlp.append(self.dp2)
        serialized = dlp.serialize()
        dlp2 = DataLoadingPlan()
        self.assertNotEqual(dlp.dlp_id, dlp2.dlp_id)
        self.assertNotIn(self.dp1, dlp2)
        self.assertNotIn(self.dp2, dlp2)
        dlp2.load(serialized)
        self.assertIn(self.dp1, dlp2)
        self.assertIn(self.dp2, dlp2)
        self.assertEqual(dlp.dlp_id, dlp2.dlp_id)
        dlp_values = dlp['pipeline_for_testing'].apply()
        dlp2_values = dlp2['pipeline_for_testing'].apply()
        for v1, v2 in zip(dlp_values, dlp2_values):
            self.assertEqual(v1, v2)

    def test_data_loading_plan_03_mixin_functionality(self):
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

        tp = MyDataset()
        self.assertTrue(hasattr(tp, '_dlp'))
        self.assertIsNone(tp._dlp)
        dlp = DataLoadingPlan()
        dlp.append(self.dp1)
        dlp.append(self.dp2)
        serialized = dlp.serialize()
        tp.set_dlp(DataLoadingPlan().load(serialized))
        self.assertIn(self.dp1, tp._dlp)
        self.assertIn('pipeline_for_testing', tp._dlp)
        self.assertIn(self.dp2, tp._dlp)
        self.assertIn('other_testing_pipeline', tp._dlp)

    def test_data_loading_plan_04_apply(self):
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

            def test_mapper(self):
                orig_key = 'orig-key'
                return self.apply_dp(orig_key, 'testing-mapper', orig_key)

        tp = MyDataset()
        dlp = DataLoadingPlan()
        dp = MapperDP()
        dp.type_id = 'testing-mapper'
        dp.map = {'orig-key': 'new-key'}
        dlp.append(dp)

        self.assertEqual(tp.test_mapper(), 'orig-key')
        tp.set_dlp(dlp)
        self.assertEqual(tp.test_mapper(), 'new-key')


if __name__ == '__main__':
    unittest.main()
