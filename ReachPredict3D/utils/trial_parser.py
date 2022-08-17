""" Set of utilities to notate trial times from micro-controller data. """
import numpy as np
from ReachPredict3D.utils.controller_data_parser import get_reach_indices, get_reach_times
from ReachPredict3D.utils.experiment_data_parser import get_exposure_times


def match_times(controller_data, experiment_data):
    """

    Parameters
    ----------
    controller_data : list
        list of experimental controller variables and values
    experiment_data : dict
        dict of trodes experimental data per session
    Returns
    -------
    controller_time_normalized : array
        array of controller times matched to trodes times, syncing controller and trodes signals
    """
    controller_time = np.asarray(controller_data['time'] / 1000)  # convert to s
    exposures = experiment_data['DIO']['top_cam']  # exposure data
    exposures = get_exposure_times(exposures)
    controller_time_normalized = controller_time - controller_time[-1] + exposures[-1]
    return controller_time_normalized


def get_successful_trials(controller_data, matched_time, experiment_data):
    """
    Parameters
    ----------
    controller_data : list
        list of data from the microcontroller
    matched_time : array
        controller event times converted to trodes time
    experiment_data : dict
        trodes experimental data for each session

    Returns
    -------
    success_rate : list
        list of successful trials
    """
    success_rate = []
    lick_data = experiment_data['DIO']['IR_beam']
    reach_indices = get_reach_indices(controller_data)
    reach_times = get_reach_times(matched_time, reach_indices)
    reach_start = reach_times['start']
    reach_stop = reach_times['stop']
    trial_num = 0
    if lick_data.any():
        for xi in range(len(reach_start) - 1):
            i = reach_start[xi]  # these are start and stop times on trodes time
            j = reach_stop[xi]

            if True in np.vectorize(lambda x: i <= x <= j)(lick_data):
                success_rate.append(xi)
    else:
        success_rate.append(-1)
    return success_rate
