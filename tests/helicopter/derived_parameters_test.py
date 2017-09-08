import unittest
import numpy as np

from analysis_engine.node import (
    Attribute, A, App, ApproachItem, KeyPointValue, KPV,
    KeyTimeInstance, KTI, M, Parameter, P, Section, S)

from analysis_engine.helicopter.derived_parameters import (
    ApproachRange,
)


class TestApproachRange(unittest.TestCase):
    def test_can_operate(self):
        operational_combinations = ApproachRange.get_operational_combinations()
        self.assertTrue(('Altitude AAL', 'Latitude Smoothed',
                         'Longitude Smoothed', 'Touchdown') in operational_combinations,
                        msg="Missing 'helicopter' combination")

    def test_derive(self):
        d = 1.0/60.0
        lat = P('Latitude', array=[0.0, d/2.0, d])
        lon = P('Longitude', array=[0.0, 0.0, 0.0])
        alt = P('Altitude AAL', array=[200, 100, 0.0])
        tdn = KTI('Touchdown', items=[KeyTimeInstance(2, 'Touchdown'),])
        ar = ApproachRange()
        ar.derive(alt, lat, lon, tdn)
        result = ar.array
        # Strictly, 1nm is 1852m, but this error arises from the haversine function.
        self.assertEqual(int(result[0]), 1853)
