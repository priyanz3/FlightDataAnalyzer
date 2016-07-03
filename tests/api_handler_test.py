# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################

'''
Flight Data Analyzer: API Handler: Tests
'''

##############################################################################
# Imports


import unittest
import yaml

from flightdatautilities import api

from analysis_engine import settings


##############################################################################
# Test Cases


class FileHandlerTest(unittest.TestCase):

    def setUp(self):
        self.handler = api.get_handler(settings.API_FILE_HANDLER)
        with open(settings.API_FILE_PATHS['airports'], 'rb') as f:
            self.airports = yaml.load(f)

    @unittest.skip('Not implemented yet.')
    def test_get_aircraft(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_analyser_profiles(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_data_exports(self):
        pass

    def test_get_airport(self):
        self.assertEqual(self.handler.get_airport(2456), self.airports[0])
        self.assertEqual(self.handler.get_airport('KRS'), self.airports[0])
        self.assertEqual(self.handler.get_airport('ENCN'), self.airports[0])
        self.assertEqual(self.handler.get_airport(2461), self.airports[1])
        self.assertEqual(self.handler.get_airport('OSL'), self.airports[1])
        self.assertEqual(self.handler.get_airport('ENGM'), self.airports[1])

    def test_get_nearest_airport(self):
        airport = self.handler.get_nearest_airport(58, 8)
        self.assertEqual(airport['distance'], 23253.447237062534)
        del airport['distance']
        self.assertEqual(airport, self.airports[0])
        airport = self.handler.get_nearest_airport(60, 11)
        self.assertEqual(airport['distance'], 22267.45203750386)
        del airport['distance']
        self.assertEqual(airport, self.airports[1])


class HTTPHandlerTest(unittest.TestCase):

    def setUp(self):
        self.handler = api.get_handler(settings.API_HTTP_HANDLER)

    @unittest.skip('Not implemented yet.')
    def test_get_aircraft(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_analyser_profiles(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_data_exports(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_airport(self):
        pass

    @unittest.skip('Not implemented yet.')
    def test_get_nearest_airport(self):
        pass
