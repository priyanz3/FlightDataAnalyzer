import os
import unittest
import numpy as np

from analysis_engine.node import (
    Attribute,
    A,
    App,
    load,
    M,
    P,
    Section,
    S,
)
from analysis_engine.helicopter.multistate_parameters import (
    GearOnGround,
)

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, 'test_data')

class TestGearOnGround(unittest.TestCase):
    def setUp(self):
        self.node_class = GearOnGround

    def test_can_operate(self):
        helicopter_expected = ('Vertical Speed', 'Eng (*) Torque Avg')
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(helicopter_expected in opts)

    def test_derive__columbia234(self):
        vert_spd = load(os.path.join(test_data_path, "gear_on_ground__columbia234_vert_spd.nod"))
        torque = load(os.path.join(test_data_path, "gear_on_ground__columbia234_torque.nod"))
        collective = load(os.path.join(test_data_path,"gear_on_ground__columbia234_collective.nod"))
        ac_series = A("Series", value="Columbia 234")
        wow = GearOnGround()
        wow.derive(vert_spd, torque, ac_series, collective)
        self.assertTrue(np.ma.all(wow.array[:252] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[254:540] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1040:1200] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1420:1440] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1533:1550] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[1615:1622] == 'Air'))
        #self.assertTrue(np.ma.all(wow.array[1696:1730] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[1900:2150] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2350:2385] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2550:2750] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2900:3020] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[3366:3376] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[3425:] == 'Ground'))

    def test_derive__columbia234_collective(self):
        vert_spd = load(os.path.join(test_data_path, "gear_on_ground__columbia234_vert_spd_flight2.nod"))
        torque = load(os.path.join(test_data_path, "gear_on_ground__columbia234_torque_flight2.nod"))
        collective = load(os.path.join(test_data_path, "gear_on_ground__columbia234_collective_flight2.nod"))
        ac_series = A("Series", value="Columbia 234")
        wow = GearOnGround()
        wow.derive(vert_spd, torque, ac_series, collective)
        self.assertTrue(all(wow.array[:277] == 'Ground'))
        self.assertTrue(all(wow.array[300:1272] == 'Air'))
        self.assertTrue(all(wow.array[1275:1470] == 'Ground'))
        self.assertTrue(all(wow.array[1474:1772] == 'Air'))
        self.assertTrue(all(wow.array[1775:1803] == 'Ground'))
        self.assertTrue(all(wow.array[1806:2107] == 'Air'))
        self.assertTrue(all(wow.array[2109:2200] == 'Ground'))
        self.assertTrue(all(wow.array[2203:3894] == 'Air'))
        self.assertTrue(all(wow.array[3896:] == 'Ground'))
