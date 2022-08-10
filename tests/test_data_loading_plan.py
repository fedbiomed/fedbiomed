import unittest
from fedbiomed.common.data import DataLoadingPlan, DataLoadingPlanMixin, MapperDP
from tests.testsupport.testing_data_pipeline import PipelineForTesting


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.changed_data = {'my': 'different-data'}
        self.dp1 = PipelineForTesting()
        self.dp2 = PipelineForTesting()

    def test_data_pipeline_01_serialize_and_load(self):
        """Tests that DataPipeline is serialized and loaded correctly"""
        self.dp1.data = self.changed_data
        self.assertFalse(self.dp1.data == self.dp2.data)
        serialized = self.dp1.serialize()
        self.assertIn('pipeline_class', serialized)
        self.assertIn('pipeline_module', serialized)
        self.assertIn('pipeline_serialization_id', serialized)

        self.dp2.deserialize(serialized)
        self.assertDictEqual(self.dp1.data, self.dp2.data)

        exec(f"import {serialized['pipeline_module']}")
        dp3 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}()")
        dp3.deserialize(serialized)
        self.assertDictEqual(self.dp1.data, dp3.data)

        dp4 = MapperDP()
        dp4.map = {'test': 1, 1: 'test'}
        serialized = dp4.serialize()
        exec(f"import {serialized['pipeline_module']}")
        dp5 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}()")
        dp5.deserialize(serialized)
        self.assertEqual(dp4.serialization_id, dp5.serialization_id)
        self.assertDictEqual(dp4.map, dp5.map)

    def test_data_pipeline_02_apply(self):
        """Tests that the apply function of DataPipeline works as intended"""
        self.dp2.data = self.changed_data
        dp3 = MapperDP()
        dp3.map = self.changed_data

        apply_1 = self.dp1.apply()
        self.assertEqual(len(apply_1), 1)
        self.assertIn('data', apply_1)
        apply_2 = self.dp2.apply()
        self.assertEqual(len(apply_2), 1)
        self.assertIn('different-data', apply_2)
        apply_3 = dp3.apply('my')
        self.assertEqual(apply_3, 'different-data')


class TestDataLoadingPlan(unittest.TestCase):
    def setUp(self):
        self.dp1 = PipelineForTesting()
        self.dp2 = PipelineForTesting()
        self.assertDictEqual(self.dp1.data, self.dp2.data)
        self.dp2.data = {'my': 'different-data'}

    def test_data_loading_plan_01_interface(self):
        """Tests that DataLoadingPlan exposes the correct interface to the developer"""
        dlp = DataLoadingPlan()
        dlp['pipeline_for_testing'] = self.dp1
        dlp['other_testing_pipeline'] = self.dp2
        self.assertIn('pipeline_for_testing', dlp)
        self.assertIn('other_testing_pipeline', dlp)
        self.assertDictEqual(self.dp1.data, dlp['pipeline_for_testing'].data)
        self.assertDictEqual(self.dp2.data, dlp['other_testing_pipeline'].data)

        it = iter(dlp.items())
        first_key, first_dp = next(it)
        self.assertEqual(first_key, 'pipeline_for_testing')
        self.assertDictEqual(self.dp1.data, first_dp.data)
        second_key, second_dp = next(it)
        self.assertEqual(second_key, 'other_testing_pipeline')
        self.assertDictEqual(self.dp2.data, second_dp.data)

        str_repr = str(dlp)
        self.assertIn(dlp.dlp_id, str_repr)
        self.assertIn('pipeline_for_testing', str_repr)
        self.assertIn('other_testing_pipeline', str_repr)

    def test_data_loading_plan_02_serialize_and_load(self):
        """Tests that a DataLoadingPlan can be serialized and loaded correctly"""
        dlp = DataLoadingPlan()
        dlp['pipeline_for_testing'] = self.dp1
        dlp['other_testing_pipeline'] = self.dp2
        dlp2 = DataLoadingPlan()
        self.assertNotEqual(dlp.dlp_id, dlp2.dlp_id)
        self.assertNotIn('pipeline_for_testing', dlp2)
        self.assertNotIn('other_testing_pipeline', dlp2)
        dlp2.deserialize(*dlp.serialize())
        self.assertIn('pipeline_for_testing', dlp2)
        self.assertIn('other_testing_pipeline', dlp2)
        self.assertEqual(dlp.dlp_id, dlp2.dlp_id)

        dlp_values = dlp['pipeline_for_testing'].apply()
        dlp2_values = dlp2['pipeline_for_testing'].apply()
        for v1, v2 in zip(dlp_values, dlp2_values):
            self.assertEqual(v1, v2)

    def test_data_loading_plan_03_mixin_functionality(self):
        """Tests that the DataLoadingPlanMixin class provides the intended functionality"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

        tp = MyDataset()
        self.assertTrue(hasattr(tp, '_dlp'))
        self.assertIsNone(tp._dlp)
        dlp = DataLoadingPlan()
        dlp['pipeline_for_testing'] = self.dp1
        dlp['other_testing_pipeline'] = self.dp2
        tp.set_dlp(DataLoadingPlan().deserialize(*dlp.serialize()))
        self.assertIn('pipeline_for_testing', tp._dlp)
        self.assertIn('other_testing_pipeline', tp._dlp)

    def test_data_loading_plan_04_apply(self):
        """Tests application of a DataLoadingPlan's DataPipeline"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

            def test_mapper(self):
                orig_key = 'orig-key'
                return self.apply_dp(orig_key, 'testing-mapper', orig_key)

        dp = MapperDP()
        dp.map = {'orig-key': 'new-key'}
        dlp = DataLoadingPlan()
        dlp['testing-mapper'] = dp

        tp = MyDataset()
        self.assertEqual(tp.test_mapper(), 'orig-key')
        tp.set_dlp(dlp)
        self.assertEqual(tp.test_mapper(), 'new-key')


if __name__ == '__main__':
    unittest.main()
