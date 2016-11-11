import numpy as np
import os
import unittest

from mock import call, Mock, patch

from flightdatautilities import api

from analysis_engine.approaches import ApproachInformation, is_heliport
from analysis_engine.flight_phase import ApproachAndLanding
from analysis_engine.node import (
    A, ApproachItem, aeroplane, helicopter, KPV, KeyPointValue, P, S, Section,
    load)

from . import airports

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'test_data', 'approaches')

class TestIsHeliport(unittest.TestCase):

    def test_is_heliport(self):
        self.assertFalse(is_heliport(aeroplane, airports.gatwick, airports.gatwick['runways'][0]))
        self.assertTrue(is_heliport(helicopter, None, None))
        helipad = {'identifier':'H', 'strip': {'length': 0}}
        heliport = {'name':'Vangard helipad', 'runways':[helipad]}
        self.assertTrue(is_heliport(helicopter, heliport, helipad))
        self.assertTrue(is_heliport(helicopter, {}, helipad))
        self.assertFalse(is_heliport(helicopter, airports.gatwick, airports.gatwick['runways'][0]))

class TestApproachInformation(unittest.TestCase):

    def setUp(self):
        self.node_class = ApproachInformation
        self.gatwick = airports.gatwick

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL'), ac_type=aeroplane))
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AGL'), ac_type=helicopter))
        self.assertTrue(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL',
             'Latitude Prepared', 'Longitude Prepared'), ac_type=aeroplane))
        self.assertFalse(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Latitude Prepared'), ac_type=aeroplane))
        self.assertFalse(ApproachInformation.can_operate(
            ('Approach And Landing', 'Altitude AAL', 'Longitude Prepared'), ac_type=aeroplane))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_basic(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.ma.arange(-2.5, 0, 0.05), [-0.15]*50))),
                          P('ILS Glideslope',np.ma.array([0.0]*100)),
                          P('ILS Frequency', np.ma.array([110.90]*100)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(41, 100, None))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_frequency_masked(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.ma.arange(-2.5, 0, 0.05), [-0.15]*50))),
                          P('ILS Glideslope',np.ma.array([0.0]*20)),
                          P('ILS Frequency', np.ma.array(data=[110.90]*20, mask=[True]*20)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_frequency_wrong_frequency(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer', np.ma.concatenate((np.ma.arange(-2.5, 0, 0.05), [-0.15]*50))),
                          P('ILS Glideslope',np.ma.array([0.0]*100)),
                          P('ILS Frequency', np.ma.array(data=[110.95]*100)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_masked_preamble(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        ils_array = np.ma.concatenate((np.ma.arange(-2.5, 0, 0.05), [-0.15]*50))
        ils_array.mask = np.ma.getmaskarray(ils_array)
        ils_array.mask[0:45] = True
        ils_array.mask[70:] = True
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer', ils_array),
                          P('ILS Glideslope',np.ma.array([0.0]*100)),
                          P('ILS Frequency', np.ma.array(data=[110.90]*100)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(45, 70))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_never_on_loc(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0, -10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(0, 100), 0, 100)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer', np.ma.array([3.0]*20)),
                          P('ILS Glideslope',np.ma.array([0.0]*20)),
                          P('ILS Frequency', np.ma.array(data=[110.90]*20)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].gs_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_only_last_segment(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(20, 90), 20, 90)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.repeat([0,0,0,1,3,3,2,1,0,0], 10), frequency = 0.5),
                          P('ILS Glideslope',np.ma.array([0.0]*100)),
                          P('ILS Frequency', np.ma.array(data=[110.90]*100)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=80, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=80, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        # Slice changed from original test to reflect new way of determining localizer established phase.
        # Not looking at loc signal, but approach phase and established startpoint, and runway turnoff endpoint.
        self.assertEqual(approaches[0].loc_est, slice(20, 70))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_stays_established_with_large_visible_deviations(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array(np.repeat([0,0,0,1,2.3,2.3,2,1,0,0], 10))),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(10, 70)) # 70 reflects the 2 dot endpoint

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_insensitive_to_few_masked_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        ils_array = np.ma.array(np.repeat([0,0,0,1,2.3,2.3,2,1,0,0], 10))
        ils_array.mask = np.ma.getmaskarray(ils_array)
        ils_array.mask[60:62] = True   
        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer',ils_array),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(10, 70)) # 70 reflects the 2 dot endpoint

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_skips_too_many_masked_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-10)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 90), 10, 90)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array(data=[0.0]*20, mask=[0,1]*10)),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=19, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=19, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_skips_too_few_values(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(4, 18), 4, 18)]),
                          P('Heading Continuous', np.ma.array([260.0]*100)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array(data=[0.0]*20, mask=[1]*10+[0]*5+[1]*5)),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_all_masked(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Alttiude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.array([260.0]*20)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array(data=[0.0]*20, mask=[1]*20)),
                          None,
                          None,
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, None)

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_always_on_loc(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(1000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.array([260.0]*20)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array([-0.2]*20)),
                          P('ILS Glideslope',np.ma.array([0.0]*20)),
                          P('ILS Frequency', np.ma.array([110.90]*20)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(2,19,None))

    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_not_above_1500ft(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(2000, 0,-100)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 19), 2, 19)]),
                          P('Heading Continuous', np.ma.array([260.0]*20)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array([0.0]*20)),
                          P('ILS Glideslope',np.ma.array([0.0]*20)),
                          P('ILS Frequency', np.ma.array([110.90]*20)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(5,19,None))


    @patch('analysis_engine.approaches.api')
    def test_ils_localizer_established_not_below_1000ft(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.arange(2000, 0,-50)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(2, 39), 2, 39)]),
                          P('Heading Continuous', np.ma.array([260.0]*40)),
                          None,
                          None,
                          P('ILS Localizer',np.ma.array([2.0]*21+[0.0]*12+[-2.0]*7)),
                          P('ILS Glideslope',np.ma.array([0.0]*40)),
                          P('ILS Frequency', np.ma.array([110.90]*40)),
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=17, value=51.145, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=17, value=-0.19, name='Longitude At Touchdown')]),
                          A('Precise Positioning', True),
                          )
        get_handler.get_nearest_airport.assert_called_with(latitude=51.145, longitude=-0.19)
        self.assertEqual(approaches[0].loc_est, slice(20,38.5,None))


    #@patch('analysis_engine.api_handler.FileHandler.get_nearest_airport')
    @unittest.skip('superceded')
    def test_derive(self, get_nearest_airport):
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


    @unittest.skip('Covered by TestHerat')
    def test_derive_afr_fallback(self):
        self.assertTrue(False, msg='Test not implemented.')

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_basic(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.array([0]*30)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(10, 27), 10, 26)]),
                          P('Heading Continuous', np.ma.array([260]*30)),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True),
                          )
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 26)

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_curved(self, api):
        
        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.array(range(50, 0, -1)+[0]*40)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.concatenate(([260]*70, np.arange(20)+260))),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True),
                          )
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 70)

    @patch('analysis_engine.approaches.api')
    def test_landing_turn_off_runway_curved_left(self, api):
        
        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = self.gatwick
        api.get_handler.return_value = get_handler

        approaches = ApproachInformation()
        approaches.derive(P('Altitude AAL For Flight Phases', np.ma.array(range(50, 0, -1)+[0]*40)),
                          None,
                          A('Aircraft Type', 'aeroplane'),
                          S(items=[Section('Approach', slice(30, 80), 30, 80)]),
                          P('Heading Continuous', np.ma.concatenate(([260]*70, np.arange(0, -20, -1)+260))),
                          None,
                          None,
                          None,
                          None,
                          None,
                          A(name='AFR Landing Airport', value={'id': 2379}),
                          A(name='AFR Landing Runway', value=None) ,
                          None,
                          None,
                          A('Precise Positioning', True),
                          )
        get_handler.get_airport.assert_called_with(2379)
        self.assertEqual(approaches[0].turnoff, 70)


class TestAlicante(unittest.TestCase):
    
    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_no_ils(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.alicante
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10091047_')
        app_start = 13261*2 # 13261 taken from CSV output file, but AppInfo runs at 2Hz.
        app_end = 13535*2        

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Alicante')
        self.assertEqual(approaches[0][3]['identifier'], '28')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        

class TestBardufoss(unittest.TestCase):

    @patch('analysis_engine.approaches.api')
    def test_slightly_offset_localizer(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.bardufoss
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None 
        root = os.path.join(test_data_path, 'ILS_test_9928419_')
        app_start = 11352
        app_end = 11800  

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Bardufoss')
        self.assertEqual(approaches[0][3]['identifier'], '10')
        # The aircraft was never established on the glidepath 
        # (started OK, but went outside 0.5 dots within 10 seconds of acquiring the localizer).
        self.assertEqual(approaches[0].gs_est, None)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 11541)
        self.assertEqual(int(approaches[0].loc_est.stop), 11684)


class TestBodo(unittest.TestCase):

    @patch('analysis_engine.approaches.api')
    def test_offset_localizer(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.bodo
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None 
        root = os.path.join(test_data_path, 'ILS_test_10076024_')
        app_start = 8780
        app_end = 8861        

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Bodo')
        self.assertEqual(approaches[0][3]['identifier'], '25')
        # The aircraft was never established on the glidepath 
        # (started OK, but went outside 0.5 dots within 10 seconds of acquiring the localizer).
        self.assertEqual(approaches[0].gs_est, None)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), app_start)
        self.assertEqual(int(approaches[0].loc_est.stop), 8861)


class TestChania(unittest.TestCase):
    
    # There is no ILS from this direction.
    @patch('analysis_engine.approaches.api')
    def test_no_ils(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.chania
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_9936188_')
        app_start = 12029*2 # 13261 taken from CSV output file, but AppInfo runs at 2Hz.
        app_end = 12324*2        

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Chania')
        self.assertEqual(approaches[0][3]['identifier'], '29')
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None) 


class TestDallasFortWorth(unittest.TestCase):
    
    # A change in runways late on the approach.
    @patch('analysis_engine.approaches.api')
    def test_runway_change_at_DFW(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.dallas_fort_worth
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_9821340_')
        app_start = 3187 * 2
        app_end = 3542 * 2

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Dallas Fort Worth Intl')
        # We approached runway 17C...
        self.assertEqual(approaches[0][4]['identifier'], '17C')
        # ...but landed on 17R...
        self.assertEqual(approaches[0][3]['identifier'], '17R')
        # The aircraft was established on the glidepath
        self.assertEqual(int(approaches[0].gs_est.start), 6523)
        self.assertEqual(int(approaches[0].gs_est.stop), 6590)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 6523)
        self.assertEqual(int(approaches[0].loc_est.stop), 6590)



class TestFortWorth(unittest.TestCase):
    
    # This is the Fort Worth that it NOT at Dallas !!!
    @patch('analysis_engine.approaches.api')
    def test_ils_corrected_data(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.fort_worth
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_9094175_')
        app_start = 17834
        app_end = 18218     

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Fort Worth Alliance')
        self.assertEqual(approaches[0][3]['identifier'], '16L')
        # The aircraft was established on the glidepath slightly after the approach start
        self.assertEqual(int(approaches[0].gs_est.start), 17870)
        self.assertEqual(int(approaches[0].gs_est.stop), 18076)
        # ...but was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), app_start)
        self.assertEqual(int(approaches[0].loc_est.stop), app_end)


class TestHerat(unittest.TestCase):
    
    # This aircraft does not record latitude and longitude
    @patch('analysis_engine.approaches.api')
    def test_no_ils_but_signal_looks_ok(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.side_effect = api.NotFoundError()
        get_handler.get_airport.return_value = airports.herat
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10411379_')
        app_start = 10792.6
        app_end = 11329.0    

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value={'id': 3275}),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown',  items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Herat')
        self.assertEqual(approaches[0][3]['identifier'], '36')
        # There is no ILS on this runway.
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)


class TestKirkenes(unittest.TestCase):
    
    # There is an ILS on the reverse runway, not on this approach.
    @patch('analysis_engine.approaches.api')
    def test_ils_on_reciprocal_runway(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.kirkenes
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10066360_')
        app_start = 13803.5
        app_end = 14246.0    

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Kirkenes')
        self.assertEqual(approaches[0][3]['identifier'], '06')
        # There is no ILS on this runway.
        self.assertEqual(approaches[0].gs_est, None)
        self.assertEqual(approaches[0].loc_est, None)
        self.assertEqual(approaches[0].ils_freq, None)
        
        
class TestScatsta(unittest.TestCase):
    
    # This airport has a localizer but no glideslope antenna
    @patch('analysis_engine.approaches.api')
    def test_no_ils_glidepath(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.scatsta
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10174453_')
        app_start = 6996.0
        app_end = 7327.0  

        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None),
                          KPV('Latitude At Touchdown', items=[KeyPointValue(index=7060, value=60.433, name='Latitude At Touchdown')]),
                          KPV('Longitude At Touchdown', items=[KeyPointValue(index=7060, value=-1.292, name='Longitude At Touchdown')]),                          
                          A('Precise Positioning', False),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Scatsta')
        self.assertEqual(approaches[0][3]['identifier'], '24')
        # We should be established on the (offset) localizer:        
        self.assertEqual(int(approaches[0].loc_est.start), 7055)
        self.assertEqual(int(approaches[0].loc_est.stop), 7086)
        # ...but there is no glideslope on this runway.
        self.assertEqual(approaches[0].gs_est, None)
        
        
class TestSirSeretseKhama(unittest.TestCase):
    
    # Note: This aircraft does not have a recorded ILS Frequency
    @patch('analysis_engine.approaches.api')
    def test_late_turn_and_glideslope_established(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.sir_seretse_khama
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10089954_')
        app_start = 8256
        app_end = 8942
        
        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Sir Seretse Khama Intl')
        self.assertEqual(approaches[0][3]['identifier'], '08')
        # The aircraft did get established on the glidepath
        self.assertEqual(int(approaches[0].gs_est.start), 8679)
        self.assertEqual(int(approaches[0].gs_est.stop), 8882)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 8679)
        self.assertEqual(int(approaches[0].loc_est.stop), app_end)


class TestWashingtonNational(unittest.TestCase):
    
    # This runway has a huge offset to the ILS - 40deg!
    @patch('analysis_engine.approaches.api')
    def test_huge_offset(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.washington_national
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10260422_')
        app_start = 8284
        app_end = 8832
        
        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(app_start, app_end),
                                           start_edge=app_start, 
                                           stop_edge=app_end)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', True),
                          )
        self.assertEqual(approaches[0][2]['name'], 'Washington National')
        self.assertEqual(approaches[0][3]['identifier'], '19')
        # The aircraft did get established on the glidepath
        self.assertEqual(approaches[0].gs_est, None)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 8534)
        self.assertEqual(int(approaches[0].loc_est.stop), 8703)


class TestZaventem(unittest.TestCase):
    
    # This was a go-around and landing.
    @patch('analysis_engine.approaches.api')
    def test_go_around_and_landing(self, api):

        get_handler = Mock()
        get_handler.get_nearest_airport.return_value = airports.zaventem
        api.get_handler.return_value = get_handler

        def fetch(par_name):
            try:
                return load(root + par_name + '.nod')
            except:
                return None
        root = os.path.join(test_data_path, 'ILS_test_10180313_')
        app_start = 11754
        app_end = 12346
        
        approaches = ApproachInformation()
        approaches.derive(fetch('Altitude AAL'),
                          fetch('Altitude AGL'),
                          A('Aircraft Type', 'aeroplane'),
                          S(name='Approach And Landing', 
                            items=[Section(name='Approach And Landing', 
                                           slice=slice(11754, 12346),
                                           start_edge=11754, stop_edge=12346),
                                   Section(name='Approach And Landing', 
                                           slice=slice(13500, 13898),
                                           start_edge=13500, stop_edge=13898)]),
                          fetch('Heading Continuous'),
                          fetch('Latitude Prepared'),
                          fetch('Longitude Prepared'),
                          fetch('ILS Localizer'),
                          fetch('ILS Glideslope'),
                          fetch('ILS Frequency'),
                          A(name='AFR Landing Airport', value=None),
                          A(name='AFR Landing Runway', value=None) ,
                          KPV('Latitude At Touchdown', items=[]),
                          KPV('Longitude At Touchdown', items=[]),
                          A('Precise Positioning', False),
                          )
        self.assertEqual(len(approaches), 2)
        self.assertEqual(approaches[0][2]['name'], 'Brussels Airport')
        self.assertEqual(approaches[0][3]['identifier'], '01')
        self.assertEqual(approaches[0].type, 'GO_AROUND')
        self.assertEqual(approaches[1].type, 'LANDING')
        self.assertEqual(int(approaches[0].gs_est.start), 12106)
        self.assertEqual(int(approaches[1].gs_est.start), 13554)
        # ...and was on the localizer
        self.assertEqual(int(approaches[0].loc_est.start), 12106)
        self.assertEqual(int(approaches[1].loc_est.start), 13554)
