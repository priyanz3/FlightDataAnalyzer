import numpy as np
# _ezclump clumps bool arrays into slices. Normally called by clump_masked
# and clump_unmasked but used here to clump discrete arrays.
from numpy.ma.extras import _ezclump

from analysis_engine.exceptions import DataFrameError, FlightPhaseError

from analysis_engine.library import (
    all_of,
    any_of,
    bearing_and_distance,
    closest_unmasked_value,
    cycle_finder,
    find_edges,
    first_valid_sample,
    index_at_value,
    index_closest_value,
    intervals_above,
    intervals_below,
    is_index_within_slice,
    is_slice_within_slice,
    max_value,
    moving_average,
    rate_of_change,
    repair_mask,
    runs_of_ones_array,
    shift_slice,
    shift_slices,
    slices_from_to,
    slices_overlap,
    slices_and,
    slices_or,
    slices_not,
    slices_remove_small_gaps,
    value_at_index,
    vstack_params_where_state,
)

from analysis_engine.node import (FlightPhaseNode, A, P, S, Section,\
     SectionNode, KTI, M)

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME,
    AIRSPEED_THRESHOLD,
    BOUNCED_LANDING_THRESHOLD,
    BOUNCED_MAXIMUM_DURATION,
    BOUNCED_MAXIMUM_HEIGHT,
    DESCENT_LOW_CLIMB_THRESHOLD,
    GROUNDSPEED_FOR_MOBILE,
    HEADING_RATE_FOR_MOBILE,
    HEADING_TURN_OFF_RUNWAY,
    HEADING_TURN_ONTO_RUNWAY,
    HOLDING_MAX_GSPD,
    HOLDING_MIN_TIME,
    HYSTERESIS_FPALT_CCD,
    INITIAL_CLIMB_THRESHOLD,
    INITIAL_APPROACH_THRESHOLD,
    KTS_TO_MPS,
    LANDING_THRESHOLD_HEIGHT,
    LEVEL_FLIGHT_MIN_DURATION,
    VERTICAL_SPEED_FOR_CLIMB_PHASE,
    VERTICAL_SPEED_FOR_DESCENT_PHASE,
    VERTICAL_SPEED_FOR_LEVEL_FLIGHT,
    RATE_OF_TURN_FOR_FLIGHT_PHASES,
    RATE_OF_TURN_FOR_TAXI_TURNS
)


class InAir(FlightPhaseNode):
    '''
    These phases are created aligned to the Altitude parameter for maximum
    accuracy when using the clump_unmasked method which does not do any
    interpolation.
    
    Note this uses Altitude AAL For Flight Phases which is simply where the
    Altitude AAL parameter has had it's array repaired.
    
    Q: this will return an open interval where 0 was last recorded - although accurate, perhaps
    setting the altitude to 1ft would be more sensible to encourage
    interpolation and the interval to start after it leaves the ground?
    '''
    #NB: Do NOT align_offset to 0 here!
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        # Just find out when altitude above airfield is non-zero.
        above_ground = intervals_above(alt_aal.array, 0)
        # Ensure that we were going fast enough to be in the air!
        self.intervals = above_ground & fast
                    
                    
class Airborne(FlightPhaseNode):
    '''
    Airborne is used for determination of prolongued periods in-flight and
    excludes short periods of data. 
    
    Essentially this filters 'In Air' phases that are longer than %d seconds.
    ''' % AIRBORNE_THRESHOLD_TIME

    # force phase to 0 offset. 
    align_offset = 0
    
    def derive(self, airs=S('In Air')):
        # Make sure we propogate None ends to data which starts or ends in
        # midflight.
        for air in airs:
            if air.duration(self.frequency) > AIRBORNE_THRESHOLD_TIME:
                self.add(air)
                
                
class BouncedLanding(FlightPhaseNode):
    '''
    Short periods of 'In Air' phases that are less than %d seconds and reach 
    an altitude of between %dft and %dft.
    
    This is as accurate as the Altitude AAL parameter is.
    ''' % (BOUNCED_MAXIMUM_DURATION, BOUNCED_LANDING_THRESHOLD,
           BOUNCED_MAXIMUM_HEIGHT)
    
    # force phase to 0 offset. 
    align_offset = 0
    
    def derive(self, alt_aal=P('Altitude AAL'), airs=S('In Air')):
        for air in airs:
            if air.duration(self.frequency) > BOUNCED_MAXIMUM_DURATION:
                continue
            ht = max_value(alt_aal.array, air)
            if BOUNCED_LANDING_THRESHOLD < ht.value < BOUNCED_MAXIMUM_HEIGHT:
                # we have a bounce, mark the duration
                self.add(air)


class GoAroundAndClimbout(FlightPhaseNode):
    '''
    We already know that the Key Time Instance has been identified at the
    lowest point of the go-around, and that it lies below the 3000ft
    approach thresholds. The function here is to expand the phase 500ft before
    and 2000ft after.
    '''

    def derive(self, alt_aal=P('Altitude AAL'), gas=KTI('Go Around')):
        # Prepare a home for multiple go-arounds. (Not a good day, eh?)
        ga_slice = []

        # Find the ups and downs in the height trace.
        alt_idxs, alt_vals = cycle_finder(alt_aal.array, min_step=500.0)

        for ga in gas:
            ga_idx = ga.index
            for n_alt, alt_idx in enumerate(alt_idxs):
                if abs(ga_idx - alt_idx) < 20:
                    index, value = closest_unmasked_value(
                        alt_aal.array, ga_idx, slice(alt_idxs[n_alt - 1],
                                                     alt_idxs[n_alt + 1]))
                    # We have matched the cycle to the (possibly radio height
                    # based) go-around KTI.
                    start_slice = slice(index, alt_idxs[n_alt - 1], -1)
                    start_array = moving_average(alt_aal.array[start_slice],
                                                 window=15)
                    ga_start = index_closest_value(start_array, value + 500)
                    stop_slice = slice(index, alt_idxs[n_alt + 1])
                    stop_array = moving_average(alt_aal.array[stop_slice],
                                                window=15)
                    ga_stop = index_closest_value(stop_array, value + 2000)

                    ga_slice.append(slice(start_slice.start - ga_start,
                                          ga_stop + stop_slice.start))
        self.create_phases(ga_slice)


class Holding(FlightPhaseNode):
    """
    Holding is a process which involves multiple turns in a short period,
    normally in the same sense. We therefore compute the average rate of turn
    over a long period to reject short turns and pass the entire holding
    period.

    Note that this is the only function that should use "Heading Increasing"
    as we are only looking for turns, and not bothered about the sense or
    actual heading angle.
    """
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               hdg=P('Heading Increasing'),
               lat=P('Latitude Smoothed'), lon=P('Longitude Smoothed')):
        _, height_bands = slices_from_to(alt_aal.array, 20000, 3000)
        # Three minutes should include two turn segments.
        turn_rate = rate_of_change(hdg, 3 * 60)
        for height_band in height_bands:
            # We know turn rate will be positive because Heading Increasing only
            # increases.
            turn_bands = np.ma.clump_unmasked(
                np.ma.masked_less(turn_rate[height_band], 0.5))
            for turn_band in shift_slices(turn_bands, height_band.start):
                # Reject short periods and check that the average groundspeed was
                # low. The index is reduced by one sample to avoid overruns, and
                # this is fine because we are not looking for great precision in
                # this test.
                hold_sec = turn_band.stop - turn_band.start
                if hold_sec < HOLDING_MIN_TIME*alt_aal.hz:
                    continue
                start = turn_band.start
                stop = turn_band.stop - 1
                _, hold_dist = bearing_and_distance(
                    lat.array[start], lon.array[start],
                    lat.array[stop], lon.array[stop])
                if hold_dist/KTS_TO_MPS/hold_sec < HOLDING_MAX_GSPD:
                    self.create_phase(start, stop)


class ApproachAndLanding(FlightPhaseNode):
    # Force to remove problem with desynchronising of approaches and landings
    # (when offset > 0.5)
    align_offset = 0

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               lands=S('Landing'), go_arounds=S('Go Around And Climbout')):
        for land in lands:
            app_start = index_closest_value(
                alt_aal.array, INITIAL_APPROACH_THRESHOLD,
                slice(land.lower_bound, 0, -1))
            self.add(Section(app_start, land.upper_bound))

        last_ga = 0
        for ga in go_arounds:
            # The go-around KTI is based on only a 500ft 'pit' but to include
            # the approach phase we stretch the start point back towards
            # 3000ft. To avoid merging multiple go-arounds, the endpoint is
            # carried across from one to the next, which is a safe thing to
            # do because the KTI algorithm works on the cycle finder results
            # which are inherently ordered.
            gapp_start = index_closest_value(
                alt_aal.array, INITIAL_APPROACH_THRESHOLD,
                slice(ga.lower_bound, last_ga, -1))
            self.add(Section(gapp_start, ga.upper_bound))
            last_ga = ga.upper_bound

        if not self:
            self.warning('Flight with no valid approach or go-around phase. '
                         'Probably truncated data')
            

class Approach(FlightPhaseNode):
    """
    This separates out the approach phase excluding the landing.
    """
    def derive(self, apps=S('Approach And Landing'), lands=S('Landing')):
        self.intervals = apps  - lands
        

class ClimbCruiseDescent(FlightPhaseNode):
    """
    Circuits of climb/cruise/descents grouped over %d feet.
    
    TODO: Review implementation - as this is only used for Cruise and Top Of\
    Climb/Descent will segregating the top part of climbs/cruise/descents be\
    enough (i.e. just everything above 500ft?) 
    It might be as simple as 'climb & cruise & descent'?
    """ % HYSTERESIS_FPALT_CCD
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               airs=S('Airborne')):
        for air in airs:
            pk_idxs, pk_vals = cycle_finder(alt_aal.array[air.slice],
                                            min_step=HYSTERESIS_FPALT_CCD)
            if pk_vals is not None:
                n = 0
                pk_idxs += air.slice.start or 0
                n_vals = len(pk_vals)
                while n < n_vals - 1:
                    pk_val = pk_vals[n]
                    pk_idx = pk_idxs[n]
                    next_pk_val = pk_vals[n + 1]
                    next_pk_idx = pk_idxs[n + 1]
                    if next_pk_val < pk_val:
                        self.create_phase(None, next_pk_idx, closed=False)  #TODO: Should end be inclusive of exclusive of the idx?
                        n += 1
                    else:
                        # We are going upwards from n->n+1, does it go down
                        # again?
                        if n + 2 < n_vals:
                            if pk_vals[n + 2] < next_pk_val:
                                # Hurrah! make that phase
                                self.create_phase(pk_idx, pk_idxs[n + 2], closed=False) #TODO: Should end be inclusive of exclusive of the idx?
                                n += 2
                        else:
                            #Q: Does pk_idx include offset if we're in air
                            #later than start?
                            self.create_phase(pk_idx, None, closed=False)
                            n += 1


class Climb(FlightPhaseNode):
    def derive(self,
               toc=KTI('Top Of Climb'),
               eot=KTI('Climb Start'), # AKA End Of Takeoff
               bod=KTI('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)

        # Now see which follows a takeoff
        for this_eot in eot:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc is None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            self.create_phase(slice(eot, closest_toc))

        return


class Climbing(FlightPhaseNode):
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airborne=S('Airborne')):
        #TODO: Implement this:
        ##array = moving_average(vert_spd.array, window=30*vert_spd.hz)
        ##climb = intervals_above(array, VERTICAL_SPEED_FOR_CLIMB_PHASE)
        ##self.intervals = climb & airborne
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airborne:
            climbing = np.ma.masked_less(vert_spd.array[air.slice],
                                         VERTICAL_SPEED_FOR_CLIMB_PHASE)
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))


class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=S('Climb Cruise Descent'),
               tocs=KTI('Top Of Climb'),
               tods=KTI('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent at this
        # time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.
        for ccd in ccds:
            toc = tocs.get_first(within_slice=ccd.slice)  # TODO: Support section rather than slice!!
            if toc:
                begin = toc.index
            else:
                begin = ccd.slice.start

            tod = tods.get_last(within_slice=ccd.slice)
            if tod:
                end = tod.index
            else:
                end = ccd.slice.stop #FIXME: if not None, this should be -1

            # Some flights just don't cruise. This can cause headaches later
            # on, so we always cruise for at least one second !
            if end <= begin:
                end = begin + 1

            self.create_phase(begin, end)


class Descending(FlightPhaseNode):
    """
    Descending faster than %dfpm towards the ground
    """ % VERTICAL_SPEED_FOR_DESCENT_PHASE
    align_offset = 0
    
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airborne=S('Airborne')):
        # Vertical speed limits of 500fpm gives good distinction with level
        # flight.
        descending = intervals_below(vert_spd.array,
                                     VERTICAL_SPEED_FOR_DESCENT_PHASE)
        self.intervals = descending & airborne


class Descent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod_set:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod_set:
            bod = this_bod.index
            # Scan the TODs
            closest_tod = None
            for this_tod in tod_list:
                if (bod > this_tod and
                    this_tod > closest_tod):
                    closest_tod = this_tod

            # Build the slice from what we have found.
            self.create_phase(slice(closest_tod, bod))
        return


class DescentToFlare(FlightPhaseNode):
    '''
    '''

    def derive(self,
            descents=S('Descent'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        '''
        '''
        for descent in descents:
            end = index_at_value(alt_aal.array, 50.0, descent.slice)
            self.create_phase(slice(descent.slice.start, end))


class DescentLowClimb(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        dlc = np.ma.masked_greater(alt_aal.array,
                                   INITIAL_APPROACH_THRESHOLD)
        for this_dlc in np.ma.clump_unmasked(dlc):
            pk_idxs, pk_vals = cycle_finder(
                dlc[this_dlc], min_step=DESCENT_LOW_CLIMB_THRESHOLD)
            if pk_vals is None or len(pk_vals) < 3:
                continue
            for n in range(1, len(pk_vals) - 1):
                if (pk_vals[n-1]-pk_vals[n]) > DESCENT_LOW_CLIMB_THRESHOLD and \
                   (pk_vals[n+1]-pk_vals[n]) > DESCENT_LOW_CLIMB_THRESHOLD:
                    self.create_phase(
                        shift_slice(slice(pk_idxs[n-1], pk_idxs[n+1]),
                                    this_dlc.start))


class Fast(FlightPhaseNode):
    '''
    Fast is used to determine that the aircraft was going fast enough to
    become airborne.
    
    Data will have been sliced into single flights before entering the
    analysis engine, so there should only be one Fast phase.
    '''

    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        self.intervals = intervals_above(airspeed.array, AIRSPEED_THRESHOLD)


class FinalApproach(FlightPhaseNode):
    '''
    '''
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_phases(alt_aal.slices_from_to(1000, 50))
        
        
class GearTransition(object):
    '''
    Shared calculation of transitioning phases (Retracting / Extending).
    '''
    def gear_transition_estimate(self, state, gear_down, airborne):
        '''
        state: 'Up' or 'Down'
        '''
        # Aircraft without red warning captions for travelling have no
        # transition state, so allow 5 seconds for the gear to retract.
        edges = find_edges(gear_down.array == state, slice(None), 
                           direction='rising_edges')
        s = SectionNode()
        for now_up in edges:
            if now_up not in airborne:
                continue
            est_down = now_up - (5.0 * gear_down.frequency)
            s.add(Section(est_down, now_up))
        return s

    def gear_red_warning(self, state, gear_down, gear_warn_n,
                         gear_warn_l, gear_warn_r):
        # Aircraft with red warning captions to show travelling
        gear_warn = vstack_params_where_state(
            (gear_warn_l, 'Warning'),
            (gear_warn_r, 'Warning'),
            (gear_warn_n, 'Warning'),
        )
        gear_moving = gear_warn.any(axis=0)
        s = SectionNode()
        for (start, end, dur) in zip(*runs_of_ones_array(gear_moving, 1)):
            if gear_down.array[start-1] != state:
                # Gear was down, so now we're retracting
                ##self.create_phase(start, end-1)
                s.add(Section(start-0.5, end-0.5))
            else:
                # Moving to a gear down transition, not interested!
                continue
        # be sure that these transitions happened while airborne!
        return s
    

class GearExtending(FlightPhaseNode, GearTransition):
    '''
    If any of the Gear Red Warnings (Left, Right or Nose) are active, the
    Gear is Extending.
    
    Aircraft without red warning captions for travelling have no transition
    state, so allow 5 seconds for the gear to transition to the 'Down' position.
    '''
    @classmethod
    def can_operate(cls, available):
        return all_of(('Gear Down', 'Airborne'), available)
    
    def derive(self, gear_down=M('Gear Down'),
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning'),
               airborne=S('Airborne')):
        if any((gear_warn_l, gear_warn_n, gear_warn_r)):
            # use the Red Warning technique
            retracting = self.gear_red_warning(
                'Down', gear_down, gear_warn_n, gear_warn_l, gear_warn_r)
            self.intervals = retracting & airborne
        else:
            # use the estimation technique
            self.intervals = self.gear_transition_estimate(
                'Down', gear_down, airborne)
        
        
class GearRetracting(FlightPhaseNode, GearTransition):
    '''
    If any of the Gear Red Warnings (Left, Right or Nose) are active, the
    Gear is Retracting.
    
    Aircraft without red warning captions for travelling have no transition
    state, so allow 5 seconds for the gear to retract to the 'Up' position.
    '''
    @classmethod
    def can_operate(cls, available):
        return all_of(('Gear Down', 'Airborne'), available)
    
    def derive(self, gear_down=M('Gear Down'),
               gear_warn_l=P('Gear (L) Red Warning'),
               gear_warn_n=P('Gear (N) Red Warning'),
               gear_warn_r=P('Gear (R) Red Warning'),
               airborne=S('Airborne')):
        if any((gear_warn_l, gear_warn_n, gear_warn_r)):
            # use the Red Warning technique
            retracting = self.gear_red_warning(
                'Up', gear_down, gear_warn_n, gear_warn_l, gear_warn_r)
            self.intervals = retracting & airborne
        else:
            # use the estimation technique
            self.intervals = self.gear_transition_estimate(
                'Up', gear_down, airborne)


def scan_ils(beam, ils_dots, height, scan_slice):
    '''
    :param beam: 'localizer' or 'glideslope'
    :type beam: str
    :returns: (ils_capture_start, ils_capture_end)
    :rtype: (float, float)
    '''
    if beam not in ['localizer', 'glideslope']:
        raise ValueError('Unrecognised beam type in scan_ils')

    # Let's check to see if we have something sensible to work with...
    if np.ma.count(ils_dots[scan_slice]) < 5 or \
       np.ma.count(ils_dots)/float(len(ils_dots)) < 0.8:
        return None

    # Find where we first see the ILS indication. We will start from 200ft to
    # avoid getting spurious glideslope readings (hence this code is the same
    # for glide and localizer).

    # Scan for going through 200ft, or in the case of a go-around, the lowest
    # point - hence 'closing' condition.
    idx_200 = index_at_value(height, 200, slice(scan_slice.stop,
                                                scan_slice.start, -1),
                             endpoint='closing')

    # In some error cases the height can be all zero or masked, leaving the
    # end of the scan as the result. This is invalid so we return empty
    # handed.
    if idx_200 == scan_slice.start:
        return None

    # Now work back to 2.5 dots when the indication is first visible.
    dots_25 = index_at_value(np.ma.abs(ils_dots), 2.5,
                             slice(idx_200, scan_slice.start, -1))
    if dots_25 is None:
        dots_25 = scan_slice.start

    # And now work forwards to the point of "Capture", defined as the first
    # time the ILS goes below 1 dot.
    if int(dots_25) == int(idx_200):
        ils_capture_idx = dots_25
    elif first_valid_sample(
        np.ma.abs(ils_dots[dots_25:idx_200]))[1] < 1.0:
        # Aircraft came into the approach phase already on the centreline.
        ils_capture_idx = dots_25
    else:
        ils_capture_idx = index_at_value(np.ma.abs(ils_dots), 1.0,
                                         slice(dots_25, idx_200, +1))
        if ils_capture_idx is None:
            # Did we start with the ILS captured?
            if np.ma.abs(ils_dots[dots_25]) < 1.0:
                ils_capture_idx = dots_25

    if beam == 'localizer':
        # We ended either where the aircraft left the beam or when the
        # approach or go-around phase ended.
        ils_end_idx = index_at_value(np.ma.abs(ils_dots), 2.5,
                                     slice(idx_200, scan_slice.stop))
        if ils_end_idx is None:
            # Can either never have captured, or data can end at less than 2.5
            # dots.
            countback_idx, last_loc = first_valid_sample(ils_dots[scan_slice.stop::-1])
            if abs(last_loc) < 2.5:
                ils_end_idx = scan_slice.stop - countback_idx
    elif beam == 'glideslope':
        ils_end_idx = idx_200
    else:
        raise ValueError("Unrecognised beam type '%s' in scan_ils" % beam)

    if ils_capture_idx and ils_end_idx:
        return ils_capture_idx, ils_end_idx
    else:
        return None, None


class ILSLocalizerEstablished(FlightPhaseNode):
    '''
    Region of approaches where ILS Localizer was captured.
    '''
    name = 'ILS Localizer Established'
    
    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=S('Approach And Landing')):
        for app in apps:
            ils_capture_start, ils_capture_end = scan_ils(
                'localizer', ils_loc.array, alt_aal.array, app.slice)
            if ils_capture_start is not None:
                self.create_phase(ils_capture_start, ils_capture_end)


'''
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    def derive(self, ils_loc = P('ILS Localizer'),
               ils_loc_ests = S('ILS Localizer Established')):
        # For most of the flight, the ILS will not be valid, so we scan only
        # the periods with valid data, ignoring short breaks:
        locs = np.ma.clump_unmasked(repair_mask(ils_loc.array))
        for loc_slice in locs:
            for ils_loc_est in ils_loc_ests:
                est_slice = ils_loc_est.slice
                if slices_overlap(loc_slice, est_slice):
                    before_established = slice(est_slice.start, loc_slice.start, -1)
                    begin = index_at_value(np.ma.abs(ils_loc.array),
                                                     3.0,
                                                     _slice=before_established)
                    end = est_slice.stop
                    self.create_phase(slice(begin, end))
                    '''


class ILSGlideslopeEstablished(FlightPhaseNode):
    """
    Within the Localizer Established phase, compute duration of approach with
    (repaired) Glideslope deviation continuously less than 1 dot,. Where > 10
    seconds, identify as Glideslope Established.
    """
    name = "ILS Glideslope Established"

    def derive(self, ils_gs = P('ILS Glideslope'),
               ils_loc_ests = S('ILS Localizer Established'),
               alt_aal=P('Altitude AAL')):
        # We don't accept glideslope approaches without localizer established
        # first, so this only works within that context. If you want to
        # follow a glidepath without a localizer, seek flight safety guidance
        # elsewhere.
        for ils_loc_est in ils_loc_ests:
            # Only look for glideslope established if the localizer was
            # established.
            if not ils_loc_est.slice.start or not ils_loc_est.slice.stop:
                continue
            gs_est_start, gs_est_end = scan_ils(
                'glideslope', ils_gs.array, alt_aal.array, ils_loc_est.slice)
            if gs_est_start is None:
                continue
            # If the glideslope signal is corrupt or there is no
            # glidepath (not fitted or out of service) there may be no
            # glideslope established phase, or the proportion of unmasked
            # values may be small.
            good_data = np.ma.count(ils_gs.array[gs_est_start:gs_est_end])
            all_data = len(ils_gs.array[gs_est_start:gs_est_end]) or 1
            if (float(good_data)/all_data) >= 0.7:
                self.create_phase(gs_est_start, gs_est_end)
            else:
                self.warning('ILS glideslope signal poor quality in '
                             'approach - considered not established.')
                continue


        """
        for ils_loc_est in ils_loc_ests:
            # Reduce the duration of the ILS localizer established period
            # down to minimum altitude. TODO: replace 100ft by variable ILS
            # category minima, possibly variable by operator.
            min_index = index_closest_value(alt_aal.array, 100, ils_loc_est.slice)

            # ^^^
            #TODO: limit this to 100ft min if the ILS Glideslope established threshold is reduced.

            # Truncate the ILS establiched phase.
            ils_loc_2_min = slice(ils_loc_est.slice.start,
                                  min(ils_loc_est.slice.stop,min_index))
            gs = repair_mask(ils_gs.array[ils_loc_2_min]) # prepare gs data
            gsm = np.ma.masked_outside(gs,-1,1)  # mask data more than 1 dot
            ends = np.ma.flatnotmasked_edges(gsm)  # find the valid endpoints
            if ends is None:
                self.debug("Did not establish localiser within +-1dot")
                continue
            elif ends[0] == 0 and ends[1] == -1:  # TODO: Pythonese this line !
                # All the data is within one dot, so the phase is already known
                self.create_phase(ils_loc_2_min)
            else:
                # Create the reduced duration phase
                reduced_phase = shift_slice(slice(ends[0],ends[1]),ils_loc_est.slice.start)
                # Cases where the aircraft shoots across the glidepath can
                # result in one or two samples within the range, in which
                # case the reduced phase will be None.
                if reduced_phase:
                    self.create_phase(reduced_phase)
            ##this_slice = ils_loc_est.slice
            ##on_slopes = np.ma.clump_unmasked(
                ##np.ma.masked_outside(repair_mask(ils_gs.array)[this_slice],-1,1))
            ##for on_slope in on_slopes:
                ##if slice_duration(on_slope, ils_gs.hz)>10:
                    ##self.create_phase(shift_slice(on_slope,this_slice.start))



class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                   app_land.slice.start))
                                                   """


class LevelFlight(FlightPhaseNode):
    '''
    Find periods of flight where the vertical speed is less than +/- %dfpm
    for at least %d minutes.
    ''' % (VERTICAL_SPEED_FOR_LEVEL_FLIGHT,
           int(LEVEL_FLIGHT_MIN_DURATION / 60))
    align_offset = 0

    def derive(self, vrt_spd=P('Vertical Speed For Flight Phases'), 
               airborne=S('Airborne')):
        level = intervals_below(abs(vrt_spd.array),
                                VERTICAL_SPEED_FOR_LEVEL_FLIGHT)
        level_in_air = level & airborne
        # filter out those which are too short
        for section in level_in_air:
            if section.duration(vrt_spd.hz) > LEVEL_FLIGHT_MIN_DURATION:
                self.add(section)


class Grounded(FlightPhaseNode):
    '''
    The opposite of 'In Air'! This is whenever the aircraft is deemed to be
    on the ground both in taxiing and takeoff / landing runs.
    
    TODO: Ignore short periods between bounced landings?
    
    Was "On Ground" but this name conflicts with a recorded 737-6 parameter name.
    '''
    def derive(self, inair=S('In Air')):
        self.intervals = ~inair


class Mobile(FlightPhaseNode):
    """
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases. As Rate Of Turn is derived directly from heading, this
    phase is guaranteed to be operable for very basic aircraft.
    """
    @classmethod
    def can_operate(cls, available):
        return 'Rate Of Turn' in available

    def derive(self, rot=P('Rate Of Turn'), gspd=P('Groundspeed')):
        move = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                         (np.ma.abs(rot.array),
                                          HEADING_RATE_FOR_MOBILE))

        if move is None:
            return # for the case where nothing happened

        if gspd:
            move_gspd = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                                  (np.ma.abs(gspd.array),
                                                   GROUNDSPEED_FOR_MOBILE))
            # moving is a numpy array so needs to be converted to a list of one
            # slice
            move[0] = min(move[0], move_gspd[0])
            move[1] = max(move[1], move_gspd[1])

        moves = [slice(move[0], move[1])]
        self.create_phases(moves)


class Landing(FlightPhaseNode):
    '''
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and thereby make sure the 50ft startpoint is exact.
    '''
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL'), fast=S('Fast')):

        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data
            # without having to go back and test against the airspeed array?
            # TODO: Improve endpoint checks. DJ
            if (speedy.slice.stop is None or \
                speedy.slice.stop >= len(alt_aal.array)):
                break

            landing_run = speedy.slice.stop
            datum = head.array[landing_run]

            first = landing_run - (300 * alt_aal.frequency)
            landing_begin = index_at_value(alt_aal.array,
                                           LANDING_THRESHOLD_HEIGHT,
                                           slice(first, landing_run))

            # The turn off the runway must lie within eight minutes of the
            # landing. (We did use 5 mins, but found some landings on long
            # runways where the turnoff did not happen for over 6 minutes
            # after touchdown).
            last = landing_run + (480 * head.frequency)

            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            if landing_end is None:
                # The data ran out before the aircraft left the runway so use
                # all we have.
                landing_end = len(head.array)-1

            self.create_phases([slice(landing_begin, landing_end)])


class LandingRoll(FlightPhaseNode):
    '''
    FDS developed this node to support the UK CAA Significant Seven
    programme. This phase is used when computing KPVs relating to the
    deceleration phase of the landing.

    "CAA to go with T/D to 60 knots with the T/D defined as less than 2 deg
    pitch (after main gear T/D)."

    The complex index_at_value ensures that if the aircraft does not flare to
    2 deg, we still capture the highest attitude as the start of the landing
    roll, and the landing roll starts as the aircraft passes 2 deg the last
    time, i.e. as the nosewheel comes down and not as the flare starts.
    '''
    @classmethod
    def can_operate(cls, available):
        if all_of(('Pitch', 'Landing'), available) and \
           any_of(('Groundspeed', 'Airspeed True'), available):
            return True
            
    def derive(self, pitch=P('Pitch'), gspd=P('Groundspeed'),
               aspd=P('Airspeed True'), lands=S('Landing')):
        if gspd:
            speed=gspd.array
        else:
            speed=aspd.array
        for land in lands:
            end = index_at_value(speed, 60.0, land.slice)
            begin = index_at_value(pitch.array, 2.0,
                                   slice(end, land.slice.start, -1),
                                   endpoint='nearest')
            self.create_phase(begin, end)


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through %dft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.

    We use Altitude AAL For Flight Phases as it has its mask repaired. This
    makes the assumption that no hysteresis is included as this will
    introduce a small error and we need to make sure the endpoint is
    exact.
    """ % (INITIAL_CLIMB_THRESHOLD)
    align_offset = 0
    
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        for speedy in fast:
            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if speedy.slice.start is None:  #Q: and alt_aal.array[0] > 35?
                continue
            # The aircraft is part way down it's takeoff run at the start of
            # the section.
            takeoff_run = speedy.slice.start
            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[takeoff_run]

            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = max(0, takeoff_run - (300 * head.frequency))
            takeoff_begin = index_at_value(np.ma.abs(head.array - datum),
                                           HEADING_TURN_ONTO_RUNWAY,
                                           slice(takeoff_run, first, -1))
            # Where the data starts in line with the runway, default to the
            # start of the data
            if takeoff_begin is None:
                takeoff_begin = first
            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.

            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            last = takeoff_run + (300 * alt_aal.frequency)
            takeoff_end = index_at_value(alt_aal.array, INITIAL_CLIMB_THRESHOLD,
                                         slice(takeoff_run, last))
            if takeoff_end is None:
                raise FlightPhaseError('Takeoff starts but does not end!')
            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            self.create_phase(takeoff_begin, takeoff_end)


class TakeoffRoll(FlightPhaseNode):
    '''
    Sub-phase originally written for the correlation tests but has found use
    in the takeoff KPVs where we are interested in the movement down the
    runway, not the turnon or liftoff.
    
    Will align to the frequency of Pitch to improve accuracy of measurements.
    '''
    align_offset = 0
    
    @classmethod
    def can_operate(cls, available):
        if all_of(('Takeoff', 'Pitch'), available):
            return True
        
    def derive(self, pitch=P('Pitch'),
               toffs=S('Takeoff'),
               acc_starts=KTI('Takeoff Acceleration Start')):
        for toff in toffs:
            for acc_start in acc_starts or []:
                if acc_start.index in toff:
                    begin = acc_start.index
                    break  # Found a valid start
            else:
                # Default if acceleration term not available.
                begin = toff.lower_bound
            two_deg_idx = index_at_value(pitch.array, 2.0, toff.slice)
            self.create_phase(begin, two_deg_idx)


class TakeoffRotation(FlightPhaseNode):
    '''
    This is used by correlation tests to check control movements during the
    rotation and lift phases.
    
    Defined by 10 seconds before liftoff to 15 seconds after.
    
    Note: Currently allows for multiple Liftoffs therefore creating multiple
    TakeoffRotations
    '''
    align_offset = 0
    
    def derive(self, lifts=KTI('Liftoff')):
        for lift in lifts:
            start = lift.index - (10 * self.frequency)
            end = lift.index + (15 * self.frequency)
            self.create_phase(start, end)


################################################################################
# Takeoff/Go-Around Ratings


class Takeoff5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff acceleration.
    
    See GoAround5MinRating for the same application in the case of a go-around.
    '''
    align_offset = 0
    
    def derive(self, toff_rolls=S('Takeoff Roll')):
        for roll in toff_rolls:
            end = roll.lower_bound + (300 * self.frequency)
            self.create_phase(roll.lower_bound, end)


class GoAround5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is measured from the
    start of a Go Around.
    '''
    align_offset = 0
    
    def derive(self, gas=S('Go Around And Climbout')):
        for ga in gas:
            end = ga.lower_bound + (300 * self.frequency)
            self.create_phase(ga.slice.lower_bound, end)


################################################################################


class TaxiIn(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('Grounded'), lands=S('Landing')):
        land = lands.get_last()
        if not land:
            return
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                taxi_start = land.slice.stop
                taxi_stop = gnd.slice.stop
                self.create_phase(slice(taxi_start, taxi_stop),
                                  name="Taxi In")


class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('Grounded'), toffs=S('Takeoff')):
        if toffs:
            toff = toffs[0]
            for gnd in gnds:
                if slices_overlap(gnd.slice, toff.slice):
                    taxi_start = gnd.slice.start+1
                    taxi_stop = toff.slice.start-1
                    self.create_phase(slice(taxi_start, taxi_stop), name="Taxi Out")


class Taxiing(FlightPhaseNode):
    '''
    The combination of Taxi In and Taxi Out.
    '''
    align_offset = 0
    
    def derive(self, t_out=S('Taxi Out'), t_in=S('Taxi In')):
        self.intervals = t_out | t_in


class Turning(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES (%.2f)
    """ % RATE_OF_TURN_FOR_FLIGHT_PHASES
    # force phase to 0 offset
    align_offset = 0
    
    def derive(self, rate_of_turn=P('Rate Of Turn')):
        # Note: No need to repair mask as Rate of Turn uses Heading
        # Continuous which has the mask repaired.
        rate = RATE_OF_TURN_FOR_FLIGHT_PHASES
        #TODO: Review whether masked_inside is best as it is exclusive of
        # the range outer limits
        turning = np.ma.masked_inside(rate_of_turn.array, -rate, rate)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn in turn_slices:
            # Interpolate between the values for closer takeoff position
            start = turn.start - 0.5  # half sample earlier
            stop = turn.stop - 0.5    # account for stop being +1
            # Use infinity in intervals!
            start_pos = start if start > 0 else None
            stop_pos = stop if stop < len(rate_of_turn.array)-1 else None
            self.create_phase(start_pos, stop_pos)
            

class TurningInAir(FlightPhaseNode):
    """
    Based on Turning phase whilst Airborne.
    
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES (%.2f)
    whilst airborne.
    """ % RATE_OF_TURN_FOR_FLIGHT_PHASES
    
    def derive(self, turning=P('Turning'), airborne=S('Airborne')):
        self.intervals = turning & airborne


class TurningOnGround(FlightPhaseNode):
    """
    Based on Turning phase whilst Grounded.

    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_TAXI_TURNS (%.2f)
    whilst grounded.
    """ % RATE_OF_TURN_FOR_TAXI_TURNS
    
    def derive(self, turning=P('Turning'), grounded=S('Grounded')):
        self.intervals = turning & grounded


class TwoDegPitchTo35Ft(FlightPhaseNode):
    """
    """
    # NOTE: Python class name restriction: '2DegPitchTo35Ft' not permitted.

    name='2 Deg Pitch To 35 Ft'

    def derive(self, pitch=P('Pitch'), takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            reversed_slice = slice(takeoff.slice.stop, takeoff.slice.start, -1)
            # Endpoint closing allows for the case where the aircraft is at
            # more than 2 deg of pitch at takeoff.
            pitch_2_deg_idx = index_at_value(pitch.array, 2.0, reversed_slice,
                                             endpoint='closing')
            self.create_section(slice(pitch_2_deg_idx, takeoff.slice.stop),
                                begin=pitch_2_deg_idx,
                                end=takeoff.stop_edge)
