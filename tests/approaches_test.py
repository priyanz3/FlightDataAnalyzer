import numpy as np
import unittest

from mock import call, Mock, patch

from analysis_engine.approaches import ApproachInformation
from analysis_engine.flight_phase import ApproachAndLanding
from analysis_engine.node import (
    A, ApproachItem, KPV, KeyPointValue, P, S, Section)


class TestApproachInformation(unittest.TestCase):

    def setUp(self):
        self.node_class = ApproachInformation
        self.alt_aal = P(name='Altitude AAL', array=np.ma.array([
            10, 5, 0, 0, 5, 10, 20, 30, 40, 50,      # Touch & Go
            50, 45, 30, 35, 30, 30, 35, 40, 40, 40,  # Go Around
            30, 20, 10, 0, 0, 0, 0, 0, 0, 0,         # Landing
        ]))
        self.app = ApproachAndLanding()
        self.fast = S(name='Fast', items=[
            Section(name='Fast', slice=slice(0, 22), start_edge=0,
                    stop_edge=22.5),
        ])

        self.land_hdg = KPV(name='Heading During Landing', items=[
            KeyPointValue(index=22, value=60),
        ])
        self.land_lat = KPV(name='Latitude At Touchdown', items=[
            KeyPointValue(index=22, value=10),
        ])
        self.land_lon = KPV(name='Longitude At Touchdown', items=[
            KeyPointValue(index=22, value=-2),
        ])
        self.appr_hdg = KPV(name='Heading At Lowest Altitude During Approach', items=[
            KeyPointValue(index=5, value=25),
            KeyPointValue(index=12, value=35),
        ])
        self.appr_lat = KPV(name='Latitude At Lowest Altitude During Approach', items=[
            KeyPointValue(index=5, value=8),
        ])
        self.appr_lon = KPV(name='Longitude At Lowest Altitude During Approach', items=[
            KeyPointValue(index=5, value=4),
        ])
        self.land_afr_apt_none = A(name='AFR Landing Airport', value=None)
        self.land_afr_rwy_none = A(name='AFR Landing Runway', value=None)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertTrue(('Approach And Landing', 'Altitude AAL', 'Fast')
                        in combinations)

    @patch('analysis_engine.api_handler.FileHandler.get_nearest_runway')
    @patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    def test_derive(self, get_nearest_airport, get_nearest_runway):
        approaches = self.node_class()
        approaches._lookup_airport_and_runway = Mock()
        approaches._lookup_airport_and_runway.return_value = [None, None]

        # No approaches if no approach sections in the flight:
        approaches.derive(self.app, self.alt_aal, self.fast)
        self.assertEqual(approaches, [])
        # Test the different approach types:
        slices = [slice(0, 5), slice(10, 15), slice(20, 25)]
        self.app.create_phases(slices)

        approaches.derive(self.app, self.alt_aal, self.fast,
                          land_afr_apt=self.land_afr_apt_none,
                          land_afr_rwy=self.land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15)),
                          ApproachItem('LANDING', slice(20, 25))])
        #approaches.set_flight_attr.assert_called_once_with()
        #approaches.set_flight_attr.reset_mock()
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[1], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None),
            call(_slice=slices[2], appr_ils_freq=None, precise=False,
                 lowest_lat=None, lowest_lon=None, lowest_hdg=None,
                 land_afr_apt=self.land_afr_apt_none,
                 land_afr_rwy=self.land_afr_rwy_none, hint='landing'),
        ])
        del approaches[:]
        approaches._lookup_airport_and_runway.reset_mock()
        # Test that landing lat/lon/hdg used for landing only, else use approach
        # lat/lon/hdg:
        approaches.derive(self.app, self.alt_aal, self.fast, self.land_hdg, self.land_lat,
                          self.land_lon, self.appr_hdg, self.appr_lat, self.appr_lon,
                          land_afr_apt=self.land_afr_apt_none,
                          land_afr_rwy=self.land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('TOUCH_AND_GO', slice(0, 5)),
                          ApproachItem('GO_AROUND', slice(10, 15),
                                       lowest_hdg=self.appr_hdg[1].value),
                          ApproachItem('LANDING', slice(20, 25),
                                       lowest_lat=self.land_lat[0].value,
                                       lowest_lon=self.land_lon[0].value,
                                       lowest_hdg=self.land_hdg[0].value)])
        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], lowest_hdg=None, lowest_lat=None,
                 lowest_lon=None, appr_ils_freq=None, precise=False),
            call(_slice=slices[1], lowest_hdg=self.appr_hdg[1].value,
                 lowest_lat=None, lowest_lon=None, appr_ils_freq=None,
                 precise=False),
            call(_slice=slices[2], lowest_hdg=self.land_hdg[0].value,
                 lowest_lat=self.land_lat[0].value, lowest_lon=self.land_lon[0].value,
                 appr_ils_freq=None, precise=False,
                 land_afr_apt=self.land_afr_apt_none, land_afr_rwy=self.land_afr_rwy_none,
                 hint='landing'),
        ])
        approaches._lookup_airport_and_runway.reset_mock()

        # FIXME: Finish implementing these tests to check that using the API
        #        works correctly and any fall back values are used as
        #        appropriate.

    @patch('analysis_engine.api_handler.FileHandler.get_nearest_runway')
    @patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    def test_derive__ils_sidestep(self, get_nearest_airport, get_nearest_runway):
        approaches = self.node_class()
        approaches._lookup_airport_and_runway = Mock()
        approaches._lookup_airport_and_runway.return_value = [None, None]

        slices = [slice(15, 25)]
        self.app.create_phases(slices)

        appr_ils_freq = KPV(name='ILS Frequency During Approach', items=[
            KeyPointValue(index=15, value=109.5),
            KeyPointValue(index=24, value=110.9),
        ])

        approaches.derive(self.app, self.alt_aal, self.fast, self.land_hdg, self.land_lat,
                          self.land_lon, self.appr_hdg, self.appr_lat, self.appr_lon,
                          appr_ils_freq=appr_ils_freq,
                          land_afr_apt=self.land_afr_apt_none,
                          land_afr_rwy=self.land_afr_rwy_none)
        self.assertEqual(approaches,
                         [ApproachItem('LANDING', slice(15, 25),
                                       lowest_lat=self.land_lat[0].value,
                                       lowest_lon=self.land_lon[0].value,
                                       lowest_hdg=self.land_hdg[0].value,
                                       ils_freq=110.9)])

        approaches._lookup_airport_and_runway.assert_has_calls([
            call(_slice=slices[0], lowest_hdg=self.land_hdg[0].value,
                 lowest_lat=self.land_lat[0].value, lowest_lon=self.land_lon[0].value,
                 appr_ils_freq=110.9, precise=False,
                 land_afr_apt=self.land_afr_apt_none, land_afr_rwy=self.land_afr_rwy_none,
                 hint='landing'),
        ])

    @unittest.skip('Test Not Implemented')
    def test_derive_afr_fallback(self):
        self.assertTrue(False, msg='Test not implemented.')
