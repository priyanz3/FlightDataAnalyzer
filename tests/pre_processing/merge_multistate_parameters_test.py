# -*- coding: utf-8 -*-

import numpy as np
import unittest

from analysis_engine.node import M

from analysis_engine.pre_processing.merge_multistate_parameters import (
    GearDown,
    GearDownInTransit,
    GearInTransit,
    GearUp,
    GearUpInTransit,
)


class TestGearDown(unittest.TestCase):

    def setUp(self):
        self.node_class = GearDown

        self.values_mapping = {
            0: 'Up',
            1: 'Down',
        }
        self.expected = M('Gear Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 15)  # 2**4-1

    def test_derive(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) Down', array=np.ma.array([1]*5 + [0]*50 + [1]*5),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) Down', array=np.ma.array([1]*6 + [0]*48 + [1]*6),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearInTransit(unittest.TestCase):

    def setUp(self):
        self.node_class = GearInTransit

        self.values_mapping = {
            0: '-',
            1: 'In Transit',
        }
        self.expected = M('Gear In Transition', array=np.ma.array([0]*5 + [1]*10 + [0]*30 + [1]*10 + [0]*5),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 15)  # 2**4-1

    def test_derive(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) In Transit', array=np.ma.array([0]*6 + [1]*9 + [0]*30 + [1]*9 + [0]*6),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) In Transit', array=np.ma.array([0]*5 + [1]*9 + [0]*32 + [1]*9 + [0]*5),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearUp(unittest.TestCase):

    def setUp(self):
        self.node_class = GearUp

        self.values_mapping = {
            0: 'Down',
            1: 'Up',
        }
        self.expected = M('Gear Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 15)  # 2**4-1

    def test_derive(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) Up', array=np.ma.array([0]*15 + [1]*30 + [0]*15),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) Up', array=np.ma.array([0]*14 + [1]*31 + [0]*15),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)
        
    def test_derive_spike(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) Up', array=np.ma.array([0]*3 + [1]*1 + [0]*11 + [1]*30 + [0]*15),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) Up', array=np.ma.array([0]*3 + [1]*1 + [0]*10 + [1]*31 + [0]*15),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearDownInTransit(unittest.TestCase):

    def setUp(self):
        self.node_class = GearDownInTransit

        self.values_mapping = {
            0: '-',
            1: 'Extending'  # 'Retracting' for Up In Transit
        }
        self.expected = M('Gear Down In Transit', array=np.ma.array([0]*45 + [1]*10 + [0]*5),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 15)  # 2**4-1

    def test_derive__combine(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) Down In Transit', array=np.ma.array([0]*46 + [1]*9 + [0]*5),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) Down In Transit', array=np.ma.array([0]*45 + [1]*9 + [0]*6),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)


class TestGearUpInTransit(unittest.TestCase):

    def setUp(self):
        self.node_class = GearUpInTransit

        self.values_mapping = {
            0: '-',
            1: 'Retracting'  # 'Extending' for Down In Transit
        }
        self.expected = M('Gear Up In Transit', array=np.ma.array([0]*5 + [1]*10 + [0]*45),
                          values_mapping=self.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertEqual(len(combinations), 15)  # 2**4-1

    def test_derive__combine(self):
        # combine individal (L/R/C/N) params
        left = M('Gear (L) Up In Transit', array=np.ma.array([0]*5 + [1]*9 + [0]*46),
                 values_mapping=self.values_mapping)
        right = M('Gear (R) Up In Transit', array=np.ma.array([0]*6 + [1]*9 + [0]*45),
                  values_mapping=self.values_mapping)
        node = self.node_class()
        node.derive(left, None, right, None)

        np.testing.assert_array_equal(node.array, self.expected.array)
        self.assertEqual(node.values_mapping, self.values_mapping)
