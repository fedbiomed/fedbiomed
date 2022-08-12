import unittest
from fedbiomed.common.data import DataLoadingPlan, DataLoadingPlanMixin, MapperBlock
from testsupport.testing_data_loading_block import LoadingBlockForTesting, LoadingBlockTypesForTesting


class TestDataLoadingBlock(unittest.TestCase):
    def setUp(self):
        self.changed_data = {'my': 'different-data'}
        self.dp1 = LoadingBlockForTesting()
        self.dp2 = LoadingBlockForTesting()

    def test_data_loading_block_01_serialize_and_load(self):
        """Tests that DataLoadingBlock is serialized and loaded correctly"""
        self.dp1.data = self.changed_data
        self.assertFalse(self.dp1.data == self.dp2.data)
        serialized = self.dp1.serialize()
        self.assertIn('loading_block_class', serialized)
        self.assertIn('loading_block_module', serialized)
        self.assertIn('loading_block_serialization_id', serialized)

        self.dp2.deserialize(serialized)
        self.assertDictEqual(self.dp1.data, self.dp2.data)

        exec(f"import {serialized['loading_block_module']}")
        dp3 = eval(f"{serialized['loading_block_module']}.{serialized['loading_block_class']}()")
        dp3.deserialize(serialized)
        self.assertDictEqual(self.dp1.data, dp3.data)

        dp4 = MapperBlock()
        dp4.map = {'test': 1, 1: 'test'}
        serialized = dp4.serialize()
        exec(f"import {serialized['loading_block_module']}")
        dp5 = eval(f"{serialized['loading_block_module']}.{serialized['loading_block_class']}()")
        dp5.deserialize(serialized)
        self.assertEqual(dp4.get_serialization_id(), dp5.get_serialization_id())
        self.assertDictEqual(dp4.map, dp5.map)

    def test_data_loading_block_02_apply(self):
        """Tests that the apply function of DataLoadingBlock works as intended"""
        self.dp2.data = self.changed_data
        dp3 = MapperBlock()
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
        self.dp1 = LoadingBlockForTesting()
        self.dp2 = LoadingBlockForTesting()
        self.assertDictEqual(self.dp1.data, self.dp2.data)
        self.dp2.data = {'my': 'different-data'}

    def test_data_loading_plan_01_interface(self):
        """Tests that DataLoadingPlan exposes the correct interface to the developer"""
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dp1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dp2
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, dlp)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, dlp)
        self.assertDictEqual(self.dp1.data, dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING].data)
        self.assertDictEqual(self.dp2.data, dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING].data)

        it = iter(dlp.items())
        first_key, first_dp = next(it)
        self.assertEqual(first_key, LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING)
        self.assertDictEqual(self.dp1.data, first_dp.data)
        second_key, second_dp = next(it)
        self.assertEqual(second_key, LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING)
        self.assertDictEqual(self.dp2.data, second_dp.data)

        str_repr = str(dlp)
        self.assertIn(dlp.dlp_id, str_repr)
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING.value, str_repr)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING.value, str_repr)

    def test_data_loading_plan_02_serialize_and_load(self):
        """Tests that a DataLoadingPlan can be serialized and loaded correctly"""
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dp1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dp2
        dlp2 = DataLoadingPlan()
        self.assertNotEqual(dlp.dlp_id, dlp2.dlp_id)
        self.assertNotIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, dlp2)
        self.assertNotIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, dlp2)

        serialized_dlp, serialized_loading_blocks = dlp.serialize()
        self.assertIn('dlp_id', serialized_dlp)
        self.assertIsInstance(serialized_dlp['dlp_id'], str)
        self.assertIn('dlp_name', serialized_dlp)
        self.assertIsInstance(serialized_dlp['dlp_name'], str)
        self.assertIn('loading_blocks', serialized_dlp)
        self.assertIsInstance(serialized_dlp['loading_blocks'], dict)
        self.assertEqual(len(serialized_dlp['loading_blocks']), 2)
        self.assertIn('key_paths', serialized_dlp)
        self.assertIsInstance(serialized_dlp['key_paths'], dict)

        self.assertIsInstance(serialized_loading_blocks, list)
        self.assertEqual(len(serialized_loading_blocks), 2)
        self.assertIn('loading_block_serialization_id', serialized_loading_blocks[0])
        self.assertIn('loading_block_serialization_id', serialized_loading_blocks[1])

        self.assertIn(serialized_loading_blocks[0]['loading_block_serialization_id'],
                      serialized_dlp['loading_blocks'].values())
        self.assertIn(serialized_loading_blocks[1]['loading_block_serialization_id'],
                      serialized_dlp['loading_blocks'].values())

        dlp2.deserialize(*dlp.serialize())
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, dlp2)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, dlp2)
        self.assertEqual(dlp.dlp_id, dlp2.dlp_id)

        dlp_values = dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING].apply()
        dlp2_values = dlp2[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING].apply()
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
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dp1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dp2
        tp.set_dlp(DataLoadingPlan().deserialize(*dlp.serialize()))
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, tp._dlp)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, tp._dlp)

    def test_data_loading_plan_04_apply(self):
        """Tests application of a DataLoadingPlan's DataLoadingBlock"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

            def test_mapper(self):
                orig_key = 'orig-key'
                return self.apply_dp(orig_key, LoadingBlockTypesForTesting.TESTING_MAPPER, orig_key)

        dp = MapperBlock()
        dp.map = {'orig-key': 'new-key'}
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.TESTING_MAPPER] = dp

        tp = MyDataset()
        self.assertEqual(tp.test_mapper(), 'orig-key')
        tp.set_dlp(dlp)
        self.assertEqual(tp.test_mapper(), 'new-key')


if __name__ == '__main__':
    unittest.main()
