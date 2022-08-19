"""
"""

import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from reachprocess.reachprocess_functions.DLC_reconstruction import get_3d_coordinates
from reachprocess.utils.config_parser import import_config_data, get_config_data
from reachprocess.utils.controller_data_parser import import_controller_data, get_reach_indices, get_reach_times
from reachprocess.utils.trial_parser import match_times, get_successful_trials, trial_mask
from reachprocess.utils.experiment_data_parser import import_trodes_data


def load_files(trodes_dir, exp_name, controller_path, config_dir, rat, session, analysis=False,
               cns_flag=False, pns=False, force_rerun_of_data=True, sample_rate=30000, save=False):
    """

    Parameters
    ----------
    trodes_dir : directory containing trodes .rec file
    exp_name : name of folder containing .rec file/ video file
    controller_path : full path to microcontroller data
    config_dir : directory containing .json file with configuration parameters
    rat : name of rat RM16
    session : name of experimental session eg S1
    analysis : boolean, set as True to extract experimental analysis
    cns_flag : boolean, manual set of cns path
    pns : boolean, manual set of pns path
    force_rerun_of_data: bool
    sample_rate: int, sampling rate for Trodes (30kHz)

    Returns
    -------
    dataframe : pandas dataframe containing experimental values for a single experimental session
    """
    # importing data
    exp_names = exp_name[2:-1]
    exp_names = exp_names.rsplit('.', 1)[0]
    trodes_dir = trodes_dir.rsplit('/', 1)[0]
    experimental_data_found = 0
    for ff in glob.glob(trodes_dir + '/**experimental_df.h5'):
        experimental_data_found = ff
        if force_rerun_of_data:
            experimental_data_found = 0
    dataframe = 0
    if experimental_data_found:
        dataframe = pd.read_hdf(experimental_data_found)
    else:
        print('Generating sensor data manually!')
        if cns_flag:
            os.chdir(cns_flag)
        trodes_data = import_trodes_data(trodes_dir, exp_names,
                                         sampling_rate=sample_rate)  # take first entry of list (starmap returns list)
        x_pot = trodes_data['analog']['x_pot']
        y_pot = trodes_data['analog']['y_pot']
        z_pot = trodes_data['analog']['z_pot']
        lick_data = trodes_data['DIO']['IR_beam']
        if pns:
            os.chdir(pns)
        try:
            config_data = import_config_data(config_dir)
            controller_data = import_controller_data(controller_path)
        except:
            print('Cant get config or controller data')
            return
        if analysis:
            true_time = match_times(controller_data, trodes_data)
            reach_indices = get_reach_indices(controller_data)
            successful_trials = get_successful_trials(controller_data, true_time, trodes_data)
            reach_masks = get_reach_times(true_time, reach_indices)
            reach_masks_start = np.asarray(reach_masks['start'])
            reach_masks_stop = np.asarray(reach_masks['stop'])
            reach_indices_start = reach_indices['start']
            reach_indices_stop = reach_indices['stop']
            trial_masks = trial_mask(true_time, reach_indices_start, reach_indices_stop, successful_trials)
            dataframe = to_df(exp_names, config_data, true_time, successful_trials, trial_masks, rat, session,
                              lick_data, controller_data, reach_indices,
                              x_pot, y_pot, z_pot, reach_masks_start, reach_masks_stop)
        if save:
            exp_save_dir = trodes_dir + '/experimental_df.h5'
            dataframe.to_hdf(exp_save_dir, key='df')
    return dataframe


def to_df(file_name, config_data, true_time, successful_trials, trial_masks, rat, session,
          lick_data, controller_data, reach_indices, x_pot, y_pot, z_pot, mstart, mstop):
    """

    Parameters
    ----------
    file_name : name of experiment file
    config_data : experimental parameters saved as a config json file for each experiment
    true_time : normalized time array
    successful_trials : array containing indices of successful trials eg [1,3,6..]
    trial_masks : mask array of normalized times containing binary success [1] and fail [0] values
    rat : rat name such as RM16
    session : experimental session such as S1
    lick_data : array of lick start and stop times
    controller_data : list containing controller data
    reach_indices : list of 'start' and 'stop' indices for reaching trials
    x_pot : array, x potentiometer values
    y_pot : array, y potentiometer values
    z_pot : array, z potentiometer values
    mstart: array, behavioral detection start times
    mstop : array, trial end times

    Returns
    -------
    dict : pandas dataframe containing an experiments data
    """
    # functions to get specific items from config file
    dim, reward_dur, x_pos, y_pos, z_pos, x0, y0, z0, r, t1, t2 = get_config_data(config_data)
    date = get_date_from_file(file_name, Trodes=False)
    moving = controller_data['rob_moving']
    r_w = controller_data['in_Reward_Win']
    exp_response = controller_data['exp_response']
    successful_trials = np.asarray(successful_trials)
    block_dict = pd.DataFrame(
        {'rat': rat, 'S': session, 'Date': date, 'dim': dim, 'time': [np.asarray(true_time).tolist()],
         'SF': [successful_trials], 't_m': [trial_masks], 'm_start': [mstart], 'm_stop': [mstop],
         'lick': [np.asarray(lick_data).tolist()],
         'x_p': [np.asarray(x_pos).tolist()], 'y_p': [np.asarray(y_pos).tolist()],
         'z_p': [np.asarray(z_pos).tolist()], 'x0': [x0], 'y0': [y0], 'z0': [z0],
         'moving': [np.asarray(moving, dtype=int)], 'RW': [r_w], 'r_start': [reach_indices['start']],
         'r_stop': [reach_indices['stop']], 'r': [r], 't2': [t2], 't1': [t1],
         'exp_response': [exp_response], 'x_pot': [x_pot], 'y_pot': [y_pot], 'z_pot': [z_pot]})
    return block_dict


def make_dict():
    """ Function to use defaultdict to make a dictionary."""
    return defaultdict(make_dict)


def get_date_from_file(file_name, Trodes=False):
    """Function to fetch name data from string of names

    Parameters
    ----------
    file_name : un-cleaned file name

    Returns
    -------
    date: string, cleaned experiment data
    """
    # split file name
    if Trodes:
        date = file_name[5:12]
    else:
        try:
            file_name = file_name.split('/')[-2]
            date = file_name[5:12]
        except:
            date = file_name[5:12]
    return date


def get_trial_metadata(file):
    """
    Parameters
    ----------
    file - string of a file name

    Returns
    -------
    controller_file - string containing address of controller file
    trodes_files - string containing address of trodes files
    config_file - string containing address of config file
    exp_name - string containing experiment name eg 'RMxxYYYYMMDD_time', found through parsing the trodes file
    """
    # controller_data
    name = file.split('/')[6]
    path_d = file.rsplit('/', 2)[0]
    path_d = file.replace('/cns', '/PNS_data')
    path_d = path_d.rsplit('/R', 2)[0]
    config_path = path_d + '/workspaces'
    controller_path = path_d + '/sensor_data'
    video_path = path_d + '/videos/**.csv'
    # trodes_data
    n = file.rsplit('/', 1)[1]
    if '/S' in file:
        sess = file.rsplit('/S')
        sess = str(sess[1])  # get 'session' part of the namestring
        ix = 'S' + sess[0]
    exp_name = str(ix) + n
    date = get_date_from_file(file)
    return controller_path, config_path, exp_name, name, ix, n, video_path, date


def get_kinematics(cns, dlt_path, rat_name):
    """Function to iterate over a data directory and extract 3-D positional data from CatScan or other data directory.

    Parameters
    -----------
    cns : str, path to cns directory
    dlt_path : str, path to DLT string
    rat_name : str, name of rat ex: 'RM16', 'RM9', 'RF1'

    Returns
    ---------
    df_list : list, contains dataframe(s) of 3-D positions over an entire experimental block session

    """
    cns_path = cns + rat_name
    cns_pattern = cns_path + '/**/*.rec'
    cl = glob.glob(cns_pattern, recursive=True)
    print(str(len(cl)) + ' of experimental blocks for this rat.')
    predicted_data = []
    for file in tqdm(glob.glob(cns_pattern, recursive=True)):
        controller_path, config_path, exp_name, name, ix, trodes_name, video_path, date = get_trial_metadata(file)
        print(video_path + ' is being processed.')
        predictions, rmse_df = get_3d_coordinates(video_path, dlt_path, '101', name)
    return predictions, rmse_df


