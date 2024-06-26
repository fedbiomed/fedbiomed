import unittest

from fedbiomed.common.utils import quantize

class TestSecaggUtils(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_01_quantize(self):
        w = [.1, .2, .3]
        clipping_range = 2
        target_range = 2**16

        val = quantize(w, clipping_range, target_range)
        
        for v in val:
            self.assertLess(v, target_range)
            self.assertGreater(v, 0)


    def test_02_multiply(self):
        pass