""" Set of utilities to notate trial times from micro-controller data. """
import numpy as np
from reachprocess.utils.controller_data_parser import get_reach_indices, get_reach_times
from reachprocess.utils.experiment_data_parser import get_exposure_times


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


def trial_mask(matched_times, r_i_start, r_i_stop, s_t):
    """Function to

    Parameters
    ----------
    matched_times : array of normalized timestamps across entire dataset
    r_i_start : indices of reaching experiments, behavior detected
    r_i_stop : indices of reaching experiments, trial end
    s_t : success or fail indices eg [1, 4, 7..]

    Returns
    -------
    new_times : array of experiment times
    """
    lenx = int(matched_times.shape[0])
    new_times = np.zeros((lenx))
    for i, j in zip(range(0, len(r_i_start) - 1), range(0, len(r_i_stop) - 1)):
        ix = int(r_i_start[i])
        jx = int(r_i_stop[i])
        if any(i == s for s in s_t):
            new_times[ix:jx] = 2
        else:
            new_times[ix:jx] = 1
    return new_times
