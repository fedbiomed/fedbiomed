import unittest
from unittest.mock import MagicMock
from fedbiomed.common.data.data_loading_plan import DataPipeline, DataLoadingPlan, DataLoadingPlanMixin


class PipelineForTesting(DataPipeline):
    def __init__(self):
        self.type_id = 'pipeline_for_testing'
        self.data = {'my': 'data'}

    def serialize(self):
        ret = super(PipelineForTesting, self).serialize()
        ret.update({'data': self.data})
        return ret

    def load(self, load_from):
        super(PipelineForTesting, self).load(load_from)
        self.data = load_from['data']
        return self


class TestDataPipeline(unittest.TestCase):
    changed_data = {'my': 'different-data'}

    def test_data_pipeline_01_equality(self):
        dp1 = PipelineForTesting()
        dp2 = PipelineForTesting()
        self.assertEqual(dp1, dp2)
        self.assertEqual(dp1, 'pipeline_for_testing')
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


class TestDataLoadingPlan(unittest.TestCase):
    def setUp(self):
        self.dp1 = PipelineForTesting()
        self.dp2 = PipelineForTesting()
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


if __name__ == '__main__':
    unittest.main()
