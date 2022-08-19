import json
import os

def get_config(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to experimental metadata file

    Returns
    -------
    config_file : dict
        dict of experimental metadata from each experiment session
    """
    os.chdir(config_path)
    config_file = str(os.listdir()[0])
    config_file = json.load(open(config_file))
    return config_file


def import_config_data(config_path):
    """

    Parameters
    ----------
    config_path : str
        path to the experimental configuration file

    Returns
    -------
    config data : dict
        dict containing experimental metadata for a given session config file
    """
    data = get_config(config_path)
    return data


def get_config_data(config_data):
    """

    Parameters
    ----------
    config_data : list containing config parameters

    Returns
    -------
    [config parameters] : various parameters and values from the config file
    """
    exp_type = config_data['RobotSettings']['commandFile']
    reward_dur = config_data['ExperimentSettings']['rewardWinDur']
    x_p = config_data['RobotSettings']['xCommandPos']
    y_p = config_data['RobotSettings']['yCommandPos']
    z_p = config_data['RobotSettings']['zCommandPos']
    x0 = config_data['RobotSettings']['x0']
    y0 = config_data['RobotSettings']['y0']
    z0 = config_data['RobotSettings']['z0']
    r = config_data['RobotSettings']['x']
    t1 = config_data['RobotSettings']['y']
    t2 = config_data['RobotSettings']['z']
    return exp_type, reward_dur, x_p, y_p, z_p, x0, y0, z0, r, t1, t2
