# -*- coding: utf-8 -*-
from __future__ import print_function

from flightdatautilities import units as ut

from analysis_engine.node import (
    A, App, DerivedParameterNode, KPV, KTI, M, P, S
)

from analysis_engine.library import (
    bearings_and_distances,
    np_ma_masked_zeros_like,
)


class ApproachRange(DerivedParameterNode):
    '''
    This is the range to the touchdown point for both ILS and visual
    approaches including go-arounds. The reference point is the ILS Localizer
    antenna where the runway is so equipped, or the end of the runway where
    no ILS is available.

    The array is masked where no data has been computed, and provides
    measurements in metres from the reference point where the aircraft is on
    an approach.

    A simpler function is provided for helicopter operations as they may
    not - in fact normally do not - have a runway to land on.
    '''

    units = ut.METER

    def derive(self,
               alt_aal=P('Altitude AAL'),
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown')):
        app_range = np_ma_masked_zeros_like(alt_aal.array)

        #Helicopter compuation does not rely on runways!
        stop_delay = 10 # To make sure the helicopter has stopped moving

        for tdwn in tdwns:
            end = tdwn.index
            endpoint = {'latitude': lat.array[end], 'longitude': lon.array[end]}
            try:
                begin = tdwns.get_previous(end).index+stop_delay
            except:
                begin = 0
            this_leg = slice(begin, end+stop_delay)
            _, app_range[this_leg] = bearings_and_distances(lat.array[this_leg],
                                                            lon.array[this_leg],
                                                            endpoint)
        self.array = app_range
