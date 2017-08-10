import numpy as np

from flightdatautilities import units as ut

from analysis_engine.library import (
    all_of,
    any_of,
    index_at_value,
    runs_of_ones,
    slices_and,
    slices_overlap,
    slices_remove_small_gaps,
    slices_remove_small_slices,
)

from analysis_engine.node import (
    A, App, FlightPhaseNode, P, S, KTI, KPV, M)

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME_RW,
    TAKEOFF_PERIOD,
)


class Airborne(FlightPhaseNode):
    '''
    Periods where the aircraft is in the air.
    We do not use Altitude AGL as the smoothing function causes values close to the
    ground to be elevated.

    On the AS330 Puma, the Gear On Ground signal is only sampled once per frame
    so is only used to confirm validity of the radio altimeter signal and for
    preliminary data validation flight phase computation.
    '''

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT'):
            return False
        return all_of(('Gear On Ground',), available)

    def derive(self,
               alt_rad=P('Altitude Radio'),
               alt_agl=P('Altitude AGL'),
               gog=M('Gear On Ground'),
               rtr=S('Rotors Turning')):
        # When was the gear in the air?
        gear_off_grounds = runs_of_ones(gog.array == 'Air')

        if alt_rad and alt_agl and rtr:
            # We can do a full analysis.
            # First, confirm that the rotors were turning at this time:
            gear_off_grounds = slices_and(gear_off_grounds, rtr.get_slices())

            # When did the radio altimeters indicate airborne?
            airs = slices_remove_small_gaps(
                np.ma.clump_unmasked(np.ma.masked_less_equal(alt_agl.array, 1.0)),
                time_limit=AIRBORNE_THRESHOLD_TIME_RW, hz=alt_agl.frequency)
            # Both is a reliable indication of being in the air.
            for air in airs:
                for goff in gear_off_grounds:
                    # Providing they relate to each other :o)
                    if slices_overlap(air, goff):
                        start_index = max(air.start, goff.start)
                        end_index = min(air.stop, goff.stop)

                        better_begin = index_at_value(
                            alt_rad.array, 1.0,
                            _slice=slice(max(start_index-5*alt_rad.frequency, 0),
                                         start_index+5*alt_rad.frequency)
                        )
                        if better_begin:
                            begin = better_begin
                        else:
                            begin = start_index

                        better_end = index_at_value(
                            alt_rad.array, 1.0,
                            _slice=slice(max(end_index+5*alt_rad.frequency, 0),
                                         end_index-5*alt_rad.frequency, -1))
                        if better_end:
                            end = better_end
                        else:
                            end = end_index

                        duration = end - begin
                        if (duration / alt_rad.hz) > AIRBORNE_THRESHOLD_TIME_RW:
                            self.create_phase(slice(begin, end))
        else:
            # During data validation we can select just sensible flights;
            # short hops make parameter validation tricky!
            self.create_phases(
                slices_remove_small_gaps(
                    slices_remove_small_slices(gear_off_grounds, time_limit=30)))


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and make sure the 35ft endpoint is exact.
    """

    @classmethod
    def can_operate(cls, available, seg_type=A('Segment Type')):
        if seg_type and seg_type.value in ('GROUND_ONLY', 'NO_MOVEMENT', 'STOP_ONLY'):
            return False
        else:
            return all_of(('Altitude AGL', 'Liftoff'), available)

    def derive(self,
               alt_agl=P('Altitude AGL'),
               lifts=S('Liftoff')):
        for lift in lifts:
            begin = max(lift.index - TAKEOFF_PERIOD * alt_agl.frequency, 0)
            end = min(lift.index + TAKEOFF_PERIOD * alt_agl.frequency, len(alt_agl.array) - 1)
            self.create_phase(slice(begin, end))
