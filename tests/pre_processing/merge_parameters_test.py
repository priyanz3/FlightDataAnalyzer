# -*- coding: utf-8 -*-

import numpy as np
import unittest

from analysis_engine.node import P

from analysis_engine.pre_processing.merge_parameters import (
    Groundspeed,
)
from numpy.ma.testutils import assert_array_equal


class TestGroundspeed(unittest.TestCase):

    def setUp(self):
        self.node_class = Groundspeed

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 3)  # 2**2-1

    def test_basic(self):
        one = P('Groundspeed (1)', np.ma.array([100, 200, 300]), frequency=0.5, offset=0.0)
        two = P('Groundspeed (2)', np.ma.array([150, 250, 350]), frequency=0.5, offset=1.0)
        gs = Groundspeed()
        gs.derive(one, two)
        # Note: end samples are not 100 & 350 due to method of merging.
        assert_array_equal(gs.array[1:-1], np.array([150, 200, 250, 300]))
        self.assertEqual(gs.frequency, 1.0)
        self.assertEqual(gs.offset, 0.0)
