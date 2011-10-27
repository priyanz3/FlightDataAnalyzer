import numpy as np
import logging

from collections import namedtuple

from derived import Derived

KeyTimeInstance = namedtuple('KeyTimeInstance', 'index state')
FlightPhase = namedtuple('FlightPhase', 'mask')

"""
<<< Note - this only creates KTIs - no KPVs!! >>>
"""



# requirements:
'''
altitude_std_smoothed
airspeed
altitude_std
altitude_aal_takeoff
altitude_aal_landing
rate_of_turn
rate_of_climb_for_flight_phases
head_mag
altitude_radio
'''


class Airborne(Derived):
    dependencies = [ALTITUDE_STD_SMOOTHED, TAKEOFF_END, LANDING_START]
    returns = [AIRBORNE]
    
    def derive(altitude_std_smoothed, takeoff_end, landing_start):
        # Create a simple "Airborne" mask that covers the period between the takeoff and landing phases.
        # We assign it to altitude_std as this makes it easy to plot and check, although it is the mask that is really useful.
        ##airborne_phase = create_phase_inside(altitude_std_smoothed, kpt['TakeoffEndEstimate'], kpt['LandingStartEstimate'])
        airborne_phase = create_phase_inside(altitude_std_smoothed, takeoff_end, landing_start)
        return FlightPhase(airborne_phase)
    
class Turning(Derived):
    dependencies = [RATE_OF_TURN]
    returns = [TURNING]

    def derive(rate_of_turn):
        turning = np.ma.masked_inside(rate_of_turn,-1.5,1.5)
        return FlightPhase(turning)

class LevelFlight(Derived):
    dependencies = [RATE_OF_CLIMB_FOR_FLIGHT_PHASES, AIRSPEED, ALTITUDE_STD]
    returns = [LEVEL_FLIGHT]
    
    def derive(rate_of_climb_for_flight_phases, airspeed, altitude_std):
        # Rate of climb and descent limits of 800fpm gives good distinction with level flight.
        level_flight = np.ma.masked_where(
            np.ma.logical_or(np.ma.abs(rate_of_climb_for_flight_phases) > 300.0, airspeed < 100.0), altitude_std)
        return FlightPhase(level_flight)        
        
class Climbing(Derived):
    dependencies = [RATE_OF_CLIMB_FOR_FLIGHT_PHASES, ALTITUDE_STD]
    returns = [CLIMBING]
    
    def derive(rate_of_climb_for_flight_phases, altitude_std):
        # Rate of climb and descent limits of 800fpm gives good distinction with level flight.
        climbing = np.ma.masked_where(rate_of_climb_for_flight_phases < 800, altitude_std)
        return FlightPhase(climbing)
    
class Descending(Derived):
    dependencies = [RATE_OF_CLIMB_FOR_FLIGHT_PHASES, ALTITUDE_STD]
    returns = [DESCENDING]
    
    def derive(rate_of_climb_for_flight_phases, altitude_std):
        # Rate of climb and descent limits of 800fpm gives good distinction with level flight.
        descending = np.ma.masked_where(rate_of_climb_for_flight_phases > -800, altitude_std)
        return FlightPhase(descending)

    
    
    
    
    
    

# KTI
class TopOfClimbTopOfDescent(Derived):
    name = "Top of Climb and Top of Descent"
    dependencies = [PHASE_AIRBORNE, ALTITUDE_STD, ALTITUDE_STD_SMOOTHED]
    returns = [TOP_OF_CLIMB, TOP_OF_DESCENT]
    
    def derive(airborne_phase, altitude_std, altitude_std_smoothed):
        """
        Threshold was based upon the idea of "Less than 600 fpm for 6 minutes"
        This was often OK, but one test data sample had a 4000ft climb 20 mins
        after level off. This led to increasing the threshold to 600 fpm in 3
        minutes which has been found to give good qualitative segregation
        between climb, cruise and descent phases.
        """
        # Updated 8/10/11 to allow for multiple cruise phases
        cruise_slices = np.ma.flatnotmasked_contiguous(np.ma.masked_less(altitude_std_smoothed,10000))
        logging.info('This block has %d cruise phase.' % len(cruise_list))
        kti_list = []
        for cruise_slice in cruise_slices:
        
            # First establish a simple monotonic timebase
            timebase = np.arange(len(airspeed[cruise_slice]))
            # Then subtract (or for the descent, add) this slope to the altitude data
            slope = timebase * (600/float(180))
            # For airborne data only, compute a climb graph on a slope
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), altitude_std[cruise_slice] - slope)
            # and the peak is the top of climb.
            n_toc = np.ma.argmax(y)
            
            # Record the moment (with respect to this cruise)
            kti_list.append(KeyTimeInstance(cruise_slice.start + n_toc, 'TopOfClimb'))
            
            # Let's find the top of descent.
            y = np.ma.masked_where(np.ma.getmask(airborne_phase[cruise_slice]), altitude_std[cruise_slice] + slope)
            n_tod = np.ma.argmax(y)
    
            kti_list.append(KeyTimeInstance(cruise_slice.start + n_tod, 'TopOfDescent'))
                    
        return kti_list
        
    

    
    
    
    

#TODO: Move below into "Derived" structure!
def takeoff_and_landing(block, fp, ph, kpt, kpv):
    
    #===========================================================================
    # TAKEOFF 
    #===========================================================================

    # Record the heading at airspeed_threshold, part way down the runway:
    head_takeoff = head_mag[kpt['TakeoffStartEstimate']]
    
    # Track back to find where the aircraft turned onto the runway
    '''
    # Version 1 using Numpy - fails when there is no suitable data to trigger a mask edge.
    countback,dummy = np.ma.flatnotmasked_edges(
                      np.ma.masked_where(
                          np.ma.abs(head_takeoff - head_mag[start_toff:0:-1]) < 15.0,
                          head_mag[start_toff:0:-1] ))
    '''
    
    # Version 2 using loop - preferred as it deals with lack of turn data.
    turn_onto_runway = kpt['TakeoffStartEstimate']
    
    while abs(head_takeoff - head_mag[turn_onto_runway]) < 15.0:
        turn_onto_runway -= 1
        if turn_onto_runway == 0:
            logging.info ('\Data did not contain turn onto runway')
            break

    if turn_onto_runway < kpt['TakeoffStartEstimate']: # Only record the runway turnoff if it was recorded.
        kpt['TakeoffTurnOntoRunway'] = turn_onto_runway # A better indication of the start of the takeoff
        
    '''    
    # Version 3 using new Seek method:
    seek (self, start, end, value):
    Not used as it would mean computing a new parameter (abs(heading change)) for little benefit.
    '''    
        
    # Over the takeoff phase, average the stable altimeter readings to get the airport altitude
    '''
    First version:
    altitude_of_takeoff_airfield_a = np.ma.mean(np.ma.masked_where
                                                (np.ma.abs(rate_of_climb[kpt['TakeoffStartEstimate']:kpt['TakeoffEndEstimate']]) > 100,
                                                 altitude_std[kpt['TakeoffStartEstimate']:kpt['TakeoffEndEstimate']]))
    '''
    
    '''
    Second version:
    #Simpler (and better) computation:
    takeoff_level_begin = max(kpt['TakeoffStartEstimate']-30,0) # Trap for running out of data
    takeoff_level_end = kpt['TakeoffStartEstimate']
    takeoff_level_midpoint = (takeoff_level_begin + takeoff_level_end)/2.0
    altitude_of_takeoff_airfield = np.ma.mean(altitude_std[takeoff_level_begin:takeoff_level_end])
    
    kpv['AltitudeTakeoff'] = [( block.start+takeoff_level_midpoint, 
                              altitude_of_takeoff_airfield,
                              altitude_std.param_name)]
    
    altitude_aal_takeoff = DerivedParameter('Altitude_AAL_Takeoff', altitude_std)
    altitude_aal_takeoff.data -= altitude_of_takeoff_airfield
    '''
    '''
    Third version:
    #TODO:
    #Overwrite altitude_aal_takeoff.data with radio altitudes below one span rad alt.
    #Compute smoothed takeoff data and hence point of liftoff
    '''
    
    # Find where we pass through 35ft in climb.
    kpt['TakeoffEnd'] = altitude_radio.seek(block, kpt['TakeoffStartEstimate'], kpt['TakeoffEndEstimate']+30, TAKEOFF_END_HEIGHT)
    kpt['TakeoffGroundEffectEnds'] = altitude_radio.seek(block, kpt['TakeoffStartEstimate'], kpt['TakeoffEndEstimate']+30, WINGSPAN)
    #kpt['InitialClimbEnd'] = altitude_aal_takeoff.seek(block, kpt['TakeoffEnd'], kpt['TopOfClimb'], INITIAL_CLIMB_END_HEIGHT)
    
    # Create a Takeoff phase
    #ph['Takeoff'] = create_phase_inside(altitude_std, kpt['TakeoffTurnOntoRunway'], kpt['TakeoffEnd'])
    #ph['Initial_Climb'] = create_phase_inside(altitude_std, kpt['TakeoffEnd'], kpt['InitialClimbEnd'])
    #ph['Climb'] = create_phase_inside(altitude_std, kpt['InitialClimbEnd'], kpt['TopOfClimb'])

    #===========================================================================
    # LANDING 
    #===========================================================================

    # Find where we descend through 50ft.
    kpt['LandingStart'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['LandingEndEstimate']-30, LANDING_START_HEIGHT)
    
    # Record the heading on the runway
    head_landing = head_mag[kpt['LandingEndEstimate']]
    
    
    # Track on to find where the aircraft turned off the runway
    turn_off_runway = kpt['LandingEndEstimate']
    while abs(head_mag[turn_off_runway] - head_landing) < 15.0:
        turn_off_runway += 1
        if turn_off_runway == block.stop - block.start:
            logging.info ('\Data did not contain turn off of runway')
            break
        
    if block.start + turn_off_runway < block.stop: # Only record the runway turnoff if it was recorded.
        kpt['LandingTurnOffRunway'] = turn_off_runway # A better indication of the end of the landing process.
        
    '''
    # Compute the landing runway altitude:
    landing_level_begin = kpt['LandingEndEstimate'] # Retain the estimate, as this is passing through 80kts
    landing_level_end = min(kpt['LandingEndEstimate']+30, len(altitude_std.data)) # Trap for running out of data
    landing_level_midpoint = (landing_level_begin + landing_level_end)/2.0
    altitude_of_landing_airfield = np.ma.mean(altitude_std[landing_level_begin:landing_level_end])
    
    kpv['AltitudeLanding'] = [( block.start+landing_level_midpoint, 
                              altitude_of_landing_airfield,
                              altitude_std.param_name)]
    
    altitude_aal_landing = DerivedParameter('Altitude_AAL_Landing', altitude_std)
    altitude_aal_landing.data -= altitude_of_landing_airfield
    
    #TODO:
    #Overwrite altitude_aal_takeoff.data with radio altitudes below one span rad alt.
    #Compute smoothed takeoff data and hence point of liftoff
    '''

    
    # Calculate the approach phase transition points
    # Computed backwards from landing to make sure we get the latter moments in case of unstable approaches.
    #kpt['ApproachStart'] = altitude_aal_landing.seek(block, kpt['LandingStart'], kpt['TopOfDescent'], APPROACH_START_HEIGHT)
    #kpt['FinalApproachStart'] = altitude_aal_landing.seek(block, kpt['LandingStart'], kpt['ApproachStart'], FINAL_APPROACH_START_HEIGHT)
    #kpt['LandingGroundEffectBegins'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['FinalApproachStart'], WINGSPAN)
    kpt['LandingGroundEffectBegins'] = altitude_radio.seek(block, kpt['LandingEndEstimate'], kpt['LandingEndEstimate']-60, WINGSPAN)
    
    #ph['Descent'] = create_phase_inside(altitude_std, kpt['TopOfDescent'], kpt['ApproachStart'])
    #ph['Approach'] = create_phase_inside(altitude_std, kpt['ApproachStart'], kpt['FinalApproachStart'])
    #ph['FinalApproach'] = create_phase_inside(altitude_std, kpt['FinalApproachStart'], kpt['LandingStart'])

    # Create the Landing phase, and Ground (outside takeoff and landing scopes)
    ph['Landing'] = create_phase_inside(altitude_std, kpt['LandingStart'], kpt['LandingTurnOffRunway'])
    ph['Ground'] = create_phase_outside(altitude_std, kpt['TakeoffTurnOntoRunway'], kpt['LandingTurnOffRunway'])
        

    
        