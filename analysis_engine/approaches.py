# -*- coding: utf-8 -*-
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
##############################################################################

'''
Flight Data Analyzer: Approaches
'''

##############################################################################
# Imports


import numpy as np

from flightdatautilities import api

from analysis_engine import settings
from analysis_engine.library import all_of, nearest_runway
from analysis_engine.node import A, aeroplane, ApproachNode, KPV, KTI, P, S, helicopter

from analysis_engine.library import (ils_established,
                                     index_at_value,
                                     is_index_within_slice,
                                     nearest_neighbour_mask_repair,
                                     runs_of_ones,
                                     distance_from_cl,
                                     peak_curvature,
                                     rate_of_change_array,
                                     slice_duration,
                                     slices_and,
                                     slices_remove_small_gaps,
                                     shift_slice,
                                     shift_slices,
                                     )


##############################################################################
# Nodes


##############################################################################
# Helper function

def is_heliport(ac_type, airport):
    return ac_type == helicopter and airport['runways'][0]['strip']['length']==0

##############################################################################
# TODO: Update docstring for ApproachNode.
class ApproachInformation(ApproachNode):
    '''
    Details of all approaches that were made including landing.

    If possible we attempt to determine the airport and runway associated with
    each approach.

    We also attempt to determine an approach type which may be one of the
    following:

    - Landing
    - Touch & Go
    - Go Around

    The date and time at the start and end of the approach is also determined.

    When determining the airport and runway, we use the heading, latitude and
    longitude at:

    a. landing for landing approaches, and
    b. the lowest point on the approach for any other approaches.

    If we are unable to determine the airport and runway for a landing
    approach, it is also possible to fall back to the achieved flight record.
    
    We then determine the periods of use of the ILS localizer and glideslope,
    based on the installed equipment at the runway, the tuned frequency and 
    the ILS signals themselves.
    
    Analysis allows for offset ILS localizers and runway changes. In both cases
    only the first established period of operation on the localizer is used to 
    determine established flight on the localizer (and possibly glideslope) as
    flight after turning off the offset localizer or stepping across to another
    runway will be flown visually.
    
    Backcourse operation is considered not established, and hence will not
    trigger safety events.

    '''

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        required = ['Approach And Landing']
        required.append('Altitude AGL' if ac_type == helicopter else 'Altitude AAL')
        # Force both Latitude and Longitude to be available if one is available
        if 'Latitude Prepared' in available and not 'Longitude Prepared' in available:
            return False
        elif 'Longitude Prepared' in available and not 'Latitude Prepared' in available:
            return False
        return all_of(required, available)

    def _lookup_airport_and_runway(self, _slice, precise, lowest_lat,
                                   lowest_lon, lowest_hdg, appr_ils_freq,
                                   land_afr_apt=None, land_afr_rwy=None,
                                   hint='approach'):
        handler = api.get_handler(settings.API_HANDLER)
        kwargs = {}
        airport, runway = None, None

        # A1. If we have latitude and longitude, look for the nearest airport:
        if lowest_lat is not None and lowest_lon is not None:
            kwargs.update(latitude=lowest_lat, longitude=lowest_lon)
            try:
                airport = handler.get_nearest_airport(**kwargs)
            except api.NotFoundError:
                msg = 'No approach airport found near coordinates (%f, %f).'
                self.warning(msg, lowest_lat, lowest_lon)
                # No airport was found, so fall through and try AFR.
            else:
                self.debug('Detected approach airport: %s', airport)
        else:
            # No suitable coordinates, so fall through and try AFR.
            self.warning('No coordinates for looking up approach airport.')
            # return None, None

        # A2. If and we have an airport in achieved flight record, use it:
        # NOTE: AFR data is only provided if this approach is a landing.
        if not airport and land_afr_apt:
            airport = handler.get_airport(land_afr_apt.value['id'])
            self.debug('Using approach airport from AFR: %s', airport['name'])

        # A3. After all that, we still couldn't determine an airport...
        if not airport:
            self.error('Unable to determine airport on approach!')
            return None, None

        if lowest_hdg is not None:

            # R1. If we have airport and heading, look for the nearest runway:
            if appr_ils_freq:
                kwargs['ilsfreq'] = appr_ils_freq

                # We already have latitude and longitude in kwargs from looking up
                # the airport. If the measurments are not precise, remove them.
                if not precise:
                    kwargs['hint'] = hint
                    del kwargs['latitude']
                    del kwargs['longitude']

            runway = nearest_runway(airport, lowest_hdg, **kwargs)
            if not runway:
                msg = 'No runway found for airport #%d @ %03.1f deg with %s.'
                self.warning(msg, airport['id'], lowest_hdg, kwargs)
                # No runway was found, so fall through and try AFR.
                if 'ilsfreq' in kwargs:
                    # This is a trap for airports where the ILS data is not
                    # available, but the aircraft approached with the ILS
                    # tuned. A good prompt for an omission in the database.
                    self.warning('Fix database? No runway but ILS was tuned.')
            else:
                self.debug('Detected approach runway: %s', runway)

        # R2. If we have a runway provided in achieved flight record, use it:
        if not runway and land_afr_rwy:
            runway = land_afr_rwy.value
            self.debug('Using approach runway from AFR: %s', runway)

        # R3. After all that, we still couldn't determine a runway...
        if not runway:
            self.error('Unable to determine runway on approach!')

        return airport, runway

        
    def derive(self,
               alt_aal=P('Altitude AAL'),
               alt_agl=P('Altitude AGL'),
               ac_type=A('Aircraft Type'),
               app=S('Approach And Landing'),
               hdg=P('Heading Continuous'),
               lat=P('Latitude Prepared'),
               lon=P('Longitude Prepared'),
               ils_loc=P('ILS Localizer'),
               ils_gs=S('ILS Glideslope'),
               ils_freq=P('ILS Frequency'),
               land_afr_apt=A('AFR Landing Airport'),
               land_afr_rwy=A('AFR Landing Runway'),
               lat_land=KPV('Latitude At Touchdown'),
               lon_land=KPV('Longitude At Touchdown'),
               precision=A('Precise Positioning'),
               ):

        precise = bool(getattr(precision, 'value', False))

        alt = alt_agl if ac_type == helicopter else alt_aal

        app_slices = app.get_slices()

        for index, _slice in enumerate(app_slices):
            # a) The last approach is assumed to be landing:
            if index == len(app_slices) - 1:
                approach_type = 'LANDING'
                landing = True
            # b) We have a touch and go if Altitude AAL reached zero:
            elif np.ma.any(alt.array[_slice] <= 0):
                if ac_type == aeroplane:
                    approach_type = 'TOUCH_AND_GO'
                    landing = False
                elif ac_type == helicopter:
                    approach_type = 'LANDING'
                    landing = True
                else:
                    raise ValueError('Not doing hovercraft!')
            # c) In any other case we have a go-around:
            else:
                approach_type = 'GO_AROUND'
                landing = False

            # Rough reference index to allow for go-arounds
            ref_idx = index_at_value(alt.array, 0.0, _slice=_slice, endpoint='nearest')

            turnoff = None
            if landing:
                tdn_hdg = np.ma.median(hdg.array[ref_idx:_slice.stop])
                lowest_hdg = tdn_hdg.tolist()%360.0
                
                # While we're here, let's compute the turnoff index for this landing.
                head_landing = hdg.array[(ref_idx+_slice.stop)/2:_slice.stop]
                peak_bend = peak_curvature(head_landing, curve_sense='Bipolar')
                fifteen_deg = index_at_value(
                    np.ma.abs(head_landing - head_landing[0]), 15.0)
                if peak_bend:
                    turnoff = ref_idx + peak_bend
                else:
                    if fifteen_deg and fifteen_deg < peak_bend:
                        turnoff = start_search + landing_turn
                    else:
                        # No turn, so just use end of landing run.
                        turnoff = _slice.stop
            else:
                # We didn't land, but this is indicative of the runway heading
                lowest_hdg = hdg.array[ref_idx].tolist()%360.0

            # Pass latitude, longitude and heading
            if lat and lon and ref_idx:
                lowest_lat = lat.array[ref_idx]
                lowest_lon = lon.array[ref_idx]
            elif lat_land and lon_land:
                lowest_lat = lat_land[-1].value
                lowest_lon = lon_land[-1].value
            else:
                lowest_lat = None
                lowest_lon = None

            kwargs = dict(
                precise=precise,
                _slice=_slice,
                lowest_lat=lowest_lat,
                lowest_lon=lowest_lon,
                lowest_hdg=lowest_hdg,
                appr_ils_freq=None,
            )

            # If the approach is a landing, pass through information from the
            # achieved flight record in case we cannot determine airport and
            # runway:
            if landing:
                kwargs.update(
                    land_afr_apt=land_afr_apt,
                    land_afr_rwy=land_afr_rwy,
                    hint='landing',
                )

            airport, landing_runway = self._lookup_airport_and_runway(**kwargs)
            if not airport:
                break
            
            # Simple determination of heliport.
            # This function may be expanded to cater for rig approaches in future.
            heliport = is_heliport(ac_type, airport)
            
            if heliport:
                self.create_approach(
                    approach_type,
                    _slice,
                    runway_change=False,
                    offset_ils=False,
                    airport=airport,
                    landing_runway=None,
                    approach_runway=None,
                    gs_est=None,
                    loc_est=None,
                    ils_freq=None,
                    turnoff=None,
                    lowest_lat=lowest_lat,
                    lowest_lon=lowest_lon,
                    lowest_hdg=lowest_hdg,
                )
                return

            #########################################################################
            ## Analysis of fixed wing approach to a runway
            ## 
            ## First step is to check the ILS frequency for the runway in use
            ## and cater for a change from the approach runway to the landing runway.
            #########################################################################
            
            appr_ils_freq = None
            runway_change = False
            offset_ils = False
            
            # Do we have a recorded ILS frequency? If so, what was it tuned to at the start of the approach??
            if ils_freq:
                appr_ils_freq = round(ils_freq.array[_slice.start], 2)
            # Was this valid, and if so did the start of the approach match the landing runway?
            if appr_ils_freq and not np.isnan(appr_ils_freq):
                kwargs['appr_ils_freq'] = appr_ils_freq
                airport, approach_runway = self._lookup_airport_and_runway(**kwargs)
                if approach_runway['id'] != landing_runway['id']:
                    runway_change = True
            else:
                # Without a frequency source, we just have to hope any localizer signal is for this runway!
                approach_runway = landing_runway

            if approach_runway['localizer'].has_key('frequency'):
                if np.ma.count(ils_loc.array[_slice]) > 10:
                    if runway_change:
                        # We only use the first frequency tuned. This stops scanning across both runways if the pilot retunes.
                        loc_slice = shift_slices(runs_of_ones(np.ma.abs(ils_freq.array[_slice]-appr_ils_freq)<0.001),
                                                 _slice.start)[0]
                    else:
                        loc_slice = _slice
                else:
                    # No localizer or inadequate data for this approach.
                    loc_slice = None
            else:
                # The approach was to a runway without an ILS, so even if it was tuned, we ignore this.
                appr_ils_freq = None
                loc_slice = None

            if appr_ils_freq and loc_slice:
                if appr_ils_freq != approach_runway['localizer']['frequency']/1000.0:
                    loc_slice = None

            #######################################################################
            ## Identification of the period established on the localizer
            #######################################################################
                    
            loc_est = None
            if loc_slice:
                valid_range = np.ma.flatnotmasked_edges(ils_loc.array[_slice])
                # I have some data to scan. Shorthand names;
                loc_start = valid_range[0] + _slice.start
                loc_end = valid_range[1] + _slice.start + 1
                # First trim to within 45 deg of runway heading, to suppress signals that are not related to this approach.
                # The value of 45 deg was selected to encompass Washington National airport with a 40 deg offset.
                hdg_diff = np.ma.abs(np.ma.mod((hdg.array-lowest_hdg)+180.0, 360.0)-180.0)
                ils_45 = index_at_value(hdg_diff, 45.0, _slice=slice(ref_idx, loc_start, -1))
                loc_start = max(loc_start, ils_45)

                # Did I get established on the localizer, and if so, when?
                loc_estab = ils_established(ils_loc.array, slice(loc_start, ref_idx), ils_loc.hz)
                if loc_estab :
                    loc_start = loc_estab
                    # Refine the end of the localizer established phase...
                    if (approach_runway and approach_runway['localizer']['is_offset']):
                        offset_ils = True
                        # The ILS established phase ends when the deviation becomes large.
                        loc_end = ils_established(ils_loc.array, slice(ref_idx, loc_start, -1), ils_loc.hz, duration='immediate')

                    elif approach_type in ['TOUCH_AND_GO', 'GO_AROUND']:
                        # We finish at the lowest point
                        loc_end = ref_idx
    
                    elif approach_type == 'LANDING':
                        if runway_change:
                            # Step across. Search for end of established phase
                            loc_end = ils_established(ils_loc.array, slice(loc_end, loc_start, -1), ils_loc.hz, duration='immediate')
                        else:
                            # Just end at 2 dots where we turn off the runway
                            loc_end_2_dots = index_at_value(np.ma.abs(ils_loc.array), 2.0, _slice=slice(loc_end, loc_start, -1))
                            if loc_end_2_dots:
                                loc_end = loc_end_2_dots
                        
                    loc_est = slice(loc_start, loc_end)

            #######################################################################
            ## Identification of the period established on the glideslope
            #######################################################################

            gs_est = None
            if loc_est and approach_runway.has_key('glideslope') and ils_gs:
                # We only look for glideslope established periods if the localizer is already established.

                # The range to scan for the glideslope starts with localizer capture and ends at the
                # minimum height point for a go-around, or 200ft for a touch-and-go or landing.
                ils_gs_start = loc_start
                ils_gs_end = loc_end
                if landing:
                    ils_gs_200 = index_at_value(alt.array, 200.0, _slice=slice(ils_gs_end, ils_gs_start, -1))
                    if ils_gs_200:
                        # Don't go beyond the localizer end of capture.
                        ils_gs_end = min(loc_end, ils_gs_200)
                else:
                    ils_gs_end = ref_idx

                # Look for ten seconds within half a dot
                ils_gs_estab = ils_established(ils_gs.array, slice(ils_gs_start, ils_gs_end), ils_gs.hz)
                if ils_gs_estab:
                    gs_est = slice(ils_gs_start, ils_gs_end)


            '''
            # These statements help set up test cases.
            print
            print airport['name']
            print approach_runway['identifier']
            print landing_runway['identifier']
            print _slice
            if loc_est:
                print 'Localizer established ', loc_est.start, loc_est.stop
            if gs_est:
                print 'Glideslope established ', gs_est.start, gs_est.stop
            print
            '''

            self.create_approach(
                approach_type,
                _slice,
                runway_change=runway_change,
                offset_ils=offset_ils,
                airport=airport,
                landing_runway=landing_runway,
                approach_runway=approach_runway,
                gs_est=gs_est,
                loc_est=loc_est,
                ils_freq=appr_ils_freq,
                turnoff=turnoff,
                lowest_lat=lowest_lat,
                lowest_lon=lowest_lon,
                lowest_hdg=lowest_hdg,
            )
