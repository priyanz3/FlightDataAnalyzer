import unittest
import numpy as np

from analysis_engine.node import (
    P, S, 
    KeyPointValue,
)

from analysis_engine.helicopter.key_point_values import (
    HeadingDuringLanding,
)

from ..flight_phase_test import (
    buildsection,
)

class TestHeadingDuringLanding(unittest.TestCase):
    def setUp(self):
        self.node_class = HeadingDuringLanding

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        expected_combinations = [('Heading Continuous', 'Transition Flight To Hover'),]
        self.assertEqual(combinations, expected_combinations)

    def test_derive_basic(self):
        head = P('Heading Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Transition Flight To Hover',11,15)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, landing,)
        expected = [KeyPointValue(index=11, value=359.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)
