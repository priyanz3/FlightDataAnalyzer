# -*- coding: utf-8 -*-

import numpy as np

from flightdatautilities import units as ut

from analysis_engine.library import any_deps, blend_two_parameters
from analysis_engine.node import DerivedParameterNode, P, A
from analysis_engine.derived_parameters import CoordinatesStraighten


class Groundspeed(DerivedParameterNode):
    
    align = False
    units = ut.KT

    @classmethod
    def can_operate(cls, available):
        return any_deps(cls, available)

    def derive(self,
               # aeroplane
               source_A=P('Groundspeed (1)'),
               source_B=P('Groundspeed (2)')):
        self.array, self.frequency, self.offset = blend_two_parameters(source_A, source_B)

class LongitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    See Latitude Smoothed for notes.
    """
    name = 'Longitude Prepared'
    align_frequency = 1
    units = ut.DEGREE

    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon=P('Longitude'), lat=P('Latitude'),
               ac_type=A('Aircraft Type')):
        """
        This removes the jumps in longitude arising from the poor resolution of
        the recorded signal.
        """
        self.array = self._smooth_coordinates(lon, lat, ac_type)

class LatitudePrepared(DerivedParameterNode, CoordinatesStraighten):
    """
    Creates Latitude Prepared from smoothed Latitude and Longitude parameters.
    See Latitude Smoothed for notes.
    """
    name = 'Latitude Prepared'
    align_frequency = 1
    units = ut.DEGREE

    # Note force to 1Hz operation as latitude & longitude can be only
    # recorded at 0.25Hz.
    def derive(self,
               # align to longitude to avoid wrap around artifacts
               lon=P('Longitude'),
               lat=P('Latitude'),
               ac_type=A('Aircraft Type')):
        self.array = self._smooth_coordinates(lat, lon, ac_type)