import unittest
from fedbiomed.common.data import DataLoadingPlan, DataLoadingPlanMixin, MapperBlock
from testsupport.testing_data_loading_block import LoadingBlockForTesting, LoadingBlockTypesForTesting, \
    TestAbstractsBlock
from fedbiomed.common.exceptions import FedbiomedLoadingBlockError

class TestDataLoadingBlock(unittest.TestCase):
    def setUp(self):
        self.changed_data = {'my': 'different-data'}
        self.dlb1 = LoadingBlockForTesting()
        self.dlb2 = LoadingBlockForTesting()

    def test_data_loading_block_01_serialize_and_load(self):
        """Tests that DataLoadingBlock is serialized and loaded correctly"""
        self.dlb1.data = self.changed_data
        self.assertFalse(self.dlb1.data == self.dlb2.data)
        serialized = self.dlb1.serialize()
        self.assertIn('loading_block_class', serialized)
        self.assertIn('loading_block_module', serialized)
        self.assertIn('loading_block_serialization_id', serialized)

        self.dlb2.deserialize(serialized)
        self.assertDictEqual(self.dlb1.data, self.dlb2.data)

        exec(f"import {serialized['loading_block_module']}")
        dlb3 = eval(f"{serialized['loading_block_module']}.{serialized['loading_block_class']}()")
        dlb3.deserialize(serialized)
        self.assertDictEqual(self.dlb1.data, dlb3.data)

        dlb4 = MapperBlock()
        dlb4.map = {'test': 1, 1: 'test'}
        serialized = dlb4.serialize()
        exec(f"import {serialized['loading_block_module']}")
        dlb5 = eval(f"{serialized['loading_block_module']}.{serialized['loading_block_class']}()")
        dlb5.deserialize(serialized)
        self.assertEqual(dlb4.get_serialization_id(), dlb5.get_serialization_id())
        self.assertDictEqual(dlb4.map, dlb5.map)

    def test_data_loading_block_02_apply(self):
        """Tests that the apply function of DataLoadingBlock works as intended"""
        self.dlb2.data = self.changed_data
        dlb3 = MapperBlock()
        dlb3.map = self.changed_data

        apply_1 = self.dlb1.apply()
        self.assertEqual(len(apply_1), 1)
        self.assertIn('data', apply_1)
        apply_2 = self.dlb2.apply()
        self.assertEqual(len(apply_2), 1)
        self.assertIn('different-data', apply_2)
        apply_3 = dlb3.apply('my')
        self.assertEqual(apply_3, 'different-data')
        with self.assertRaises(FedbiomedLoadingBlockError):
            dlb3.apply('not-my')

    def test_data_loading_block_03_abstract(self):
        """Tests for abstract method(s) of DataLoadingBlock"""

        # block class to cheat ABC into running abstract method(s)
        dlb = TestAbstractsBlock()
        apply = dlb.apply("some", ["arbitrary", 3], "arguments", {}, 8)

        self.assertEqual(apply, None)


class TestDataLoadingPlan(unittest.TestCase):
    def setUp(self):
        self.dlb1 = LoadingBlockForTesting()
        self.dlb2 = LoadingBlockForTesting()
        self.assertDictEqual(self.dlb1.data, self.dlb2.data)
        self.dlb2.data = {'my': 'different-data'}

    def test_data_loading_plan_01_interface(self):
        """Tests that DataLoadingPlan exposes the correct interface to the developer"""
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dlb1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dlb2
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, dlp)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, dlp)
        self.assertDictEqual(self.dlb1.data, dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING].data)
        self.assertDictEqual(self.dlb2.data, dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING].data)

        it = iter(dlp.items())
        first_key, first_dlb = next(it)
        self.assertEqual(first_key, LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING)
        self.assertDictEqual(self.dlb1.data, first_dlb.data)
        second_key, second_dlb = next(it)
        self.assertEqual(second_key, LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING)
        self.assertDictEqual(self.dlb2.data, second_dlb.data)

        str_repr = str(dlp)
        self.assertIn(dlp.dlp_id, str_repr)
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING.value, str_repr)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING.value, str_repr)

    def test_data_loading_plan_02_serialize_and_load(self):
        """Tests that a DataLoadingPlan can be serialized and loaded correctly"""
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dlb1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dlb2
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
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = self.dlb1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = self.dlb2

        # heuristic test that no DLP exist for dataset
        apply_1 = tp.apply_dlb("my default", LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING)
        self.assertEqual(apply_1, "my default")
        apply_2 = tp.apply_dlb("other default", LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING)
        self.assertEqual(apply_2, "other default")

        # test that DLP is properly set for dataset
        tp.set_dlp(DataLoadingPlan().deserialize(*dlp.serialize()))
        apply_1 = list(tp.apply_dlb("my default", LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING))
        self.assertEqual(apply_1, ['data'])
        apply_2 = list(tp.apply_dlb("other default", LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING))
        self.assertEqual(apply_2, ['different-data'])        

        # test DLP was properly cleared
        tp.clear_dlp()
        apply_1 = tp.apply_dlb("my default", LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING)
        self.assertEqual(apply_1, "my default")
        apply_2 = tp.apply_dlb("other default", LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING)
        self.assertEqual(apply_2, "other default")

    def test_data_loading_plan_04_apply(self):
        """Tests application of a DataLoadingPlan's DataLoadingBlock"""
        class MyDataset(DataLoadingPlanMixin):
            def __init__(self):
                super(MyDataset, self).__init__()

            def test_mapper(self):
                orig_key = 'orig-key'
                return self.apply_dlb(orig_key, LoadingBlockTypesForTesting.TESTING_MAPPER, orig_key)

        dlb = MapperBlock()
        dlb.map = {'orig-key': 'new-key'}
        dlp = DataLoadingPlan()
        dlp[LoadingBlockTypesForTesting.TESTING_MAPPER] = dlb

        tp = MyDataset()
        self.assertEqual(tp.test_mapper(), 'orig-key')
        tp.set_dlp(dlp)
        self.assertEqual(tp.test_mapper(), 'new-key')


if __name__ == '__main__':
    unittest.main()
