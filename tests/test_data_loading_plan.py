import unittest
from fedbiomed.common.data import DataLoadingPlan, DataLoadingPlanMixin, MapperDP
from tests.testsupport.testing_data_pipeline import PipelineForTesting


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.changed_data = {'my': 'different-data'}
        self.dp1 = PipelineForTesting('pipeline_for_testing')
        self.dp2 = PipelineForTesting('pipeline_for_testing')

    def test_data_pipeline_01_equality(self):
        """Tests checking equality of DataPipeline"""
        dp3 = MapperDP('testing-mapper')
        self.assertEqual(self.dp1, self.dp2)
        self.assertEqual(self.dp1, 'pipeline_for_testing')
        self.assertEqual(dp3, 'testing-mapper')
        self.assertNotEqual(self.dp1, dp3)
        self.assertNotEqual(self.dp2, dp3)
        self.dp2.data = self.changed_data
        self.assertEqual(self.dp1, self.dp2)  # Equality is based only on type_id, not data!
        self.assertFalse(self.dp1.data == self.dp2.data)

    def test_data_pipeline_02_serialize_and_load(self):
        """Tests that DataPipeline is serialized and loaded correctly"""
        self.dp1.data = self.changed_data
        self.assertFalse(self.dp1.data == self.dp2.data)
        serialized = self.dp1.serialize()
        self.assertIn('pipeline_class', serialized)
        self.assertIn('pipeline_module', serialized)
        self.assertIn('type_id', serialized)
        self.assertIn('pipeline_serialization_id', serialized)

        self.dp2.load(serialized)
        self.assertDictEqual(self.dp1.data, self.dp2.data)

        exec(f"import {serialized['pipeline_module']}")
        dp3 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}('{serialized['type_id']}')")
        dp3.load(serialized)
        self.assertEqual(self.dp1, dp3)
        self.assertDictEqual(self.dp1.data, dp3.data)

        dp4 = MapperDP('testing-mapper')
        self.assertEqual(dp4, 'testing-mapper')
        serialized = dp4.serialize()
        exec(f"import {serialized['pipeline_module']}")
        dp5 = eval(f"{serialized['pipeline_module']}.{serialized['pipeline_class']}('{serialized['type_id']}')")
        dp5.load(serialized)
        self.assertEqual('testing-mapper', dp5)
        dp5 = dp5.load(serialized)
        self.assertEqual('testing-mapper', dp5)

    def test_data_pipeline_03_apply(self):
        """Tests that the apply function of DataPipeline works as intended"""
        self.dp2.data = self.changed_data
        dp3 = MapperDP('testing-mapper')
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
        self.dp1 = PipelineForTesting('pipeline_for_testing')
        self.dp2 = PipelineForTesting('other_testing_pipeline')
        self.assertDictEqual(self.dp1.data, self.dp2.data)
        self.dp2.data = {'my': 'different-data'}

    def test_data_loading_plan_01_interface(self):
        """Tests that DataLoadingPlan exposes the correct interface to the developer"""
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
        self.assertIn('pipeline_for_testing', str_repr)
        self.assertIn('other_testing_pipeline', str_repr)

    def test_data_loading_plan_02_serialize_and_load(self):
        """Tests that a DataLoadingPlan can be serialized and loaded correctly"""
        dlp = DataLoadingPlan()
        dlp.append(self.dp1)
        dlp.append(self.dp2)
        dlp2 = DataLoadingPlan()
        self.assertNotEqual(dlp.dlp_id, dlp2.dlp_id)
        self.assertNotIn(self.dp1, dlp2)
        self.assertNotIn(self.dp2, dlp2)
        aggregated_serialized = DataLoadingPlan.aggregate_serialized_metadata(dlp.serialize(),
                                                                              dlp.serialize_pipelines())
        dlp2.load_from_aggregated_serialized(aggregated_serialized)
        self.assertIn(self.dp1, dlp2)
        self.assertIn(self.dp2, dlp2)
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
        dlp.append(self.dp1)
        dlp.append(self.dp2)
        aggregated_serialized = DataLoadingPlan.aggregate_serialized_metadata(dlp.serialize(),
                                                                              dlp.serialize_pipelines())
        tp.set_dlp(DataLoadingPlan().load_from_aggregated_serialized(aggregated_serialized))
        self.assertIn(self.dp1, tp._dlp)
        self.assertIn('pipeline_for_testing', tp._dlp)
        self.assertIn(self.dp2, tp._dlp)
        self.assertIn('other_testing_pipeline', tp._dlp)

    def test_data_loading_plan_04_apply(self):
        """Tests application of a DataLoadingPlan's DataPipeline"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

            def test_mapper(self):
                orig_key = 'orig-key'
                return self.apply_dp(orig_key, 'testing-mapper', orig_key)

        tp = MyDataset()
        dlp = DataLoadingPlan()
        dp = MapperDP('testing-mapper')
        dp.map = {'orig-key': 'new-key'}
        dlp.append(dp)

        self.assertEqual(tp.test_mapper(), 'orig-key')
        tp.set_dlp(dlp)
        self.assertEqual(tp.test_mapper(), 'new-key')


if __name__ == '__main__':
    unittest.main()
