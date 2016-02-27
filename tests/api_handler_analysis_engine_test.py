import responses
import simplejson
import socket
import unittest

from mock import Mock, patch

from analysis_engine.api_handler import (
    APIError,
    APIHandlerHTTP,
    NotFoundError,
)
from analysis_engine.api_handler_analysis_engine import (
    AnalysisEngineAPIHandlerLocal,
)


class APIHandlerHTTPTest(unittest.TestCase):

    @responses.activate
    def test_request(self):
        responses.add(responses.GET, 'http://example.com/api/1/valid', json={'data': 'ok'})
        responses.add(responses.GET, 'http://example.com/api/1/missing', status=404)
        responses.add(responses.GET, 'http://example.com/api/1/unauthorized', status=401)
        responses.add(responses.GET, 'http://example.com/api/1/server_error', status=500)
        responses.add(responses.GET, 'http://example.com/api/1/decode_error', status=200,
                      body='{"invalid"}', content_type='application/json')

        handler = APIHandlerHTTP()

        self.assertEqual(handler.request('http://example.com/api/1/valid'), {'data': 'ok'})
        self.assertRaises(NotFoundError, handler.request, 'http://example.com/api/1/missing')
        self.assertRaises(APIError, handler.request, 'http://example.com/api/1/unauthorized')
        self.assertRaises(APIError, handler.request, 'http://example.com/api/1/server_error')
        self.assertRaises(APIError, handler.request, 'http://example.com/api/1/decode_error')


class AnalysisEngineAPIHandlerLocalTest(unittest.TestCase):

    def setUp(self):
        self.handler = AnalysisEngineAPIHandlerLocal()

    def test_get_airport(self):
        self.assertEqual(self.handler.get_airport(2456),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport('KRS'),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport('ENCN'),
                         self.handler.airports[0])
        self.assertEqual(self.handler.get_airport(2461),
                         self.handler.airports[1])
        self.assertEqual(self.handler.get_airport('OSL'),
                         self.handler.airports[1])
        self.assertEqual(self.handler.get_airport('ENGM'),
                         self.handler.airports[1])

    def test_get_nearest_airport(self):
        airport = self.handler.get_nearest_airport(58, 8)
        self.assertEqual(airport['distance'], 23253.447237062534)
        del airport['distance']
        self.assertEqual(airport, self.handler.airports[0])
        airport = self.handler.get_nearest_airport(60, 11)
        self.assertEqual(airport['distance'], 22267.45203750386)
        del airport['distance']
        self.assertEqual(airport, self.handler.airports[1])

    def test_get_nearest_runway(self):
        runway = self.handler.get_nearest_runway(None, None, latitude=58, longitude=8)
        self.assertAlmostEqual(runway['distance'], 22316.691624918927)
        del runway['distance']
        self.assertEqual(runway, self.handler.runways[0])
        runway = self.handler.get_nearest_runway(None, None, latitude=60, longitude=11)
        self.assertAlmostEqual(runway['distance'], 20972.761983734454)
        del runway['distance']
        self.assertEqual(runway, self.handler.runways[1])
