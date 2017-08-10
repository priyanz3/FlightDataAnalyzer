from analysis_engine.node import (
    KeyPointValueNode, P, S
)

from flightdatautilities import units as ut


class HeadingDuringLanding(KeyPointValueNode):
    '''
    We take the median heading during the landing roll as this avoids problems
    with drift just before touchdown and heading changes when turning off the
    runway. The value is "assigned" to a time midway through the landing phase.

    This KPV is a helicopter varient to accommodate helicopter transitions,
    so that the landing runway can be identified where the aircraft is
    operating at a conventional airport.
    '''

    units = ut.DEGREE

    def derive(self,
               hdg=P('Heading Continuous'),
               land_helos=S('Transition Flight To Hover')):
        for land_helo in land_helos:
            index = land_helo.slice.start
            self.create_kpv(index, float(round(hdg.array[index], 8)) % 360.0)
