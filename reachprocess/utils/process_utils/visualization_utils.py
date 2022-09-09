""" Code intended to visualize raw data and statistics for DeepLabCut keypoint predictions
    and their associated probabilities, 3-D positions, velocities, speeds and accelerations. These visualization functions
    may be extended to individual segments of timeseries data (such as reaching).  Brett Nelson 8/2022"""
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def mkdir_p(my_path):
    """Creates a directory. equivalent to using mkdir -p on the command line.

    Returns
    -------
    object
    """
    Path(my_path).mkdir(parents=True, exist_ok=True)


def load_raw_data_from_disk(disk_path):
    """ Utility function to load a csv file from disk path, used in debugging mode.
        Parameters
        -----------
        disk_path : str, path to disk where data lays

        Returns
        ---------
        dataframe : array, pandas dataframe object containing experimental data. """
    dataframe = pd.read_csv(disk_path, index_col=False)
    return dataframe


def create_directory_for_session(root_dir, rat, date, session, win_dir=True):
    """ Function to intake current video path, create new directories in a main directory to save newly created plots
        in a structured directory similar to ones used to hold experimental data. """
    if win_dir:
        process_path = root_dir + "ReachProcess\\" + rat + '\\' + date + '\\' + session
        mkdir_p(process_path)
        mkdir_p(process_path + "\\boxplot")
        mkdir_p(process_path + "\\colorplot")
        mkdir_p(process_path + "\\3d_plots")
        mkdir_p(process_path + "\\classification_videos")
        mkdir_p(process_path + "\\timeseries")
    else:
        process_path = root_dir + "ReachProcess/" + rat + '/' + date + '/' + session
        mkdir_p(process_path)
        mkdir_p(process_path + "/boxplot")
        mkdir_p(process_path + "/colorplot")
        mkdir_p(process_path + "/3d_plots")
        mkdir_p(process_path + "/classification_videos")
        mkdir_p(process_path + "/timeseries")
        mkdir_p(process_path + '/trials')
    return process_path


def preprocessing_boxplot(save_path, rmse_dataframe, prob_data):
    """ Function to create standardized boxplots for keypoint variables within ecosystem. """
    rmse_dataframe.boxplot(fontsize=3, showfliers=False)
    plt.ylabel('RMSE (px)')
    plt.savefig(save_path + '\\boxplot\\rmse_values_boxplot.png', dpi=1400)
    plt.close()
    prob_data.boxplot(fontsize=3, showfliers=False)
    plt.ylabel('DLC Confidence (p-value)')
    plt.savefig(save_path + '\\boxplot\\prob_values_boxplot.png', dpi=1400)
    plt.close()


def preprocessing_colormaps(save_path, rmse_dataframe, prob_data, sensor_data):
    """ Function to plot 2-D colordepth map for data types used in evaluating goodness of fit within data.
        """
    trial_starts = sensor_data['r_start'][0]
    ax = sns.heatmap(rmse_dataframe, cbar_kws={'label': 'RMSE error (px)'})
    plt.hlines(trial_starts, *ax.get_xlim())
    plt.savefig(save_path + '/colorplot/heatmap_rmse_trials.png', dpi=1400)
    plt.close()
    ax = sns.heatmap(rmse_dataframe, cbar_kws={'label': 'RMSE error (px)'})
    plt.savefig(save_path + '/colorplot/heatmap_rmse.png', dpi=1400)
    plt.close()
    ax = sns.heatmap(prob_data,  cbar_kws={'label': 'Mean Certainty of DLC Predictions (p-value)'})
    plt.hlines(trial_starts, *ax.get_xlim())
    plt.savefig(save_path + '/colorplot/heatmap_probs_trials.png', dpi=1400)
    plt.close()
    ax = sns.heatmap(prob_data,  cbar_kws={'label': 'Mean Certainty of DLC Predictions (p-value)'})
    plt.savefig(save_path + '/colorplot/heatmap_probs.png', dpi=1400)
    plt.close()


def preprocessing_timeseries(save_path, pred_data, prob_data, sensor_data):
    """ Function to plot a given set of time-series data from 3-D predictions of keypoints,
        their probability of location, and the rmse. Option to display windows of behavior vs non-behavior."""
    behavior_starts = sensor_data['r_start'].to_numpy()[0]
    plt.figure()
    for name, col in pred_data.items():
        plt.plot(col.to_numpy().T, label=name)
    plt.vlines(behavior_starts, -0.2, 1, colors='r', label='Behavior Start')
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.ylim([-0.2, 1])
    plt.savefig(save_path + '/timeseries/total_3D_timeseries_behavior_starts.png', dpi=1400)
    plt.close()
    plt.figure()
    for name, col in pred_data.items():
        plt.plot(col.to_numpy().T, label=name)
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.ylim([-0.2, 1])
    plt.savefig(save_path + '/timeseries/total_3D_timeseries.png', dpi=1400)
    plt.close()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    mean = []
    for row in pred_data.iterrows():
        mean.append(np.mean(row[1].to_numpy()))
    ax1.plot(mean, color='g', label='Mean Values 3-D Positions')
    plt.xlabel('Time')
    ax1.set_ylabel('Mean Position', color='g')
    mean = []
    for row in prob_data.iterrows():
        mean.append(np.mean(row[1].to_numpy()))
    ax2.plot(mean, color='b', label='Mean Probability')
    ax2.set_ylabel('P-Value (DLC)', color='b')
    plt.vlines(behavior_starts, -0.2, 1, colors='r', label='Behavior Start')
    plt.legend()
    plt.savefig(save_path + '/timeseries/timeseries_summary_behavior_starts.png', dpi=1400)
    plt.close()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    mean = []
    for row in pred_data.iterrows():
        mean.append(np.mean(row[1].to_numpy()))
    ax1.plot(mean, color='g', label='Mean Values 3-D Positions')
    plt.xlabel('Time')
    ax1.set_ylabel('Mean Position', color='g')
    mean = []
    for row in prob_data.iterrows():
        mean.append(np.mean(row[1].to_numpy()))
    ax2.plot(mean, color='b', label='Mean Probability')
    ax2.set_ylabel('P-Value (DLC)', color='b')
    plt.legend()
    plt.savefig(save_path + '/timeseries/timeseries_summary.png', dpi=1400)
    plt.close()


def get_column_names_kinematics(kinematics_data):
    names_vx = []
    names_vy = []
    names_vz = []
    names_ax = []
    names_ay = []
    names_az = []
    names_s = []
    for col_names in kinematics_data.columns:
        if 'v' in col_names:
            if 'Xv' in col_names:
                names_vx.append(col_names)
            elif 'Yv' in col_names:
                names_vy.append(col_names)
            elif 'Zv' in col_names:
                names_vz.append(col_names)
        if 'a' in col_names:
            if 'Xa' in col_names:
                names_ax.append(col_names)
            elif 'Ya' in col_names:
                names_ay.append(col_names)
            elif 'Za' in col_names:
                names_az.append(col_names)
        if 'S' in col_names:
            names_s.append(col_names)
    return names_vx, names_vy, names_vz, names_ax, names_ay, names_az, names_s


def kinematics_boxplot(save_path, kinematics_data):
    """ Function to create standardized boxplots for keypoint variable kinematics (velocity, acceleration). """
    names_vx, names_vy, names_vz, names_ax, names_ay, names_az, names_s = get_column_names_kinematics(kinematics_data)
    pdb.set_trace()
    kinematics_data.boxplot(column=names_vx, showfliers=False)
    plt.savefig(save_path + '/boxplots/velocity_x.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_vy, showfliers=False)
    plt.savefig(save_path + '/boxplots/velocity_y.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_vz, showfliers=False)
    plt.savefig(save_path + '/boxplots/velocity_z.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_ax, showfliers=False)
    plt.savefig(save_path + '/boxplots/acceleration_x.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_ay, showfliers=False)
    plt.savefig(save_path+'/boxplots/acceleration_y.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_az, showfliers=False)
    plt.savefig(save_path + '/boxplots/acceleration_z.png', dpi=1400)
    plt.close()
    kinematics_data.boxplot(column=names_s, showfliers=False)
    plt.savefig(save_path + '/boxplots/speed.png')
    plt.close()


def make_general_kinematic_timeseries_plots(save_path, kinematics_data):
    """ Function to plot general aspects of the kinematic data in time-series format. """
    names_vx, names_vy, names_vz, names_ax, names_ay, names_az, names_s = get_column_names_kinematics(kinematics_data)
    pdb.set_trace()
    kinematics_data.plot(column=names_vx)
    plt.savefig(save_path + '/timeseries/velocity_x.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_vy)
    plt.savefig(save_path + '/timeseries/velocity_y.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_vz)
    plt.savefig(save_path + '/timeseries/velocity_z.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_ax)
    plt.savefig(save_path + '/timeseries/acceleration_x.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_ay)
    plt.savefig(save_path+'/timeseries/acceleration_y.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_az)
    plt.savefig(save_path + '/timeseries/acceleration_z.png', dpi=1400)
    plt.close()
    kinematics_data.plot(column=names_s)
    plt.savefig(save_path + '/timeseries/speed.png')
    plt.close()


def make_palm_timeseries_plots(save_path, kinematic_data):
    """ Function to obtain heuristics for palm only. """



def make_lick_event(licking_times, window_length=1, num_events=10):
    """ Function to filter out noise in licking sensor, given a window length in seconds and a input number
        of events to threshold over.
    """
    if window_length == 1:
        licking_times = np.rint(licking_times)  # round to second
    key_val = np.unique(licking_times, return_index=True, return_counts=True)
    unique_times = key_val[0]
    num_set = key_val[2]
    for i, k in enumerate(unique_times):
        if num_set[i] < num_events:
            ld = np.where(licking_times == k)[0]  # get index of values.
            licking_times = np.delete(licking_times, ld)  # delete non-working indexes.
    return licking_times


def make_behavior_mask(start_times, stop_times, time):
    behavior_mask = np.zeros(time.shape)
    for i, s in enumerate(start_times):
        stop = stop_times[i]
        behavior_mask[s:stop] = 1
    return behavior_mask




def visualize_data(root_dir, rat, date, session, pred_data, prob_data, rmse_data, sensor_data, kinematics):
    """ Function, callable within ReachProcess, to visualize experimental data. Creates new directory structure to
        hold obtained plots, saves them within this structure. For more information, please see the documentation. """
    save_path = create_directory_for_session(root_dir, rat, date, session, win_dir=True)
    #correlation_plots(pred_data, 0, 90000)
    #preprocessing_boxplot(save_path, rmse_data, prob_data)
    #preprocessing_colormaps(save_path, rmse_data, prob_data,sensor_data)
    pdb.set_trace()
    preprocessing_timeseries(save_path, pred_data, prob_data, sensor_data)
    #kinematics_boxplot(save_path, kinematics)
    #make_general_kinematic_timeseries_plots(save_path, kinematics)
    behavior_start = sensor_data['r_start'][0]
    behavior_stop = sensor_data['r_stop'][0]
    for i, times in enumerate(behavior_start):
        stop = behavior_stop[i]
        trial_path = save_path + '/trials/trial '+str(i)+'/visualization/'
        mkdir_p(trial_path)
        make_3d_scatter(pred_data, times, stop, trial_path)
    licking_events = make_lick_event(sensor_data['lick'][0])
    behavior_mask = make_behavior_mask(sensor_data['r_start'][0], sensor_data['r_stop'][0],
                                          np.asarray(sensor_data['time'][0]))


def make_3d_scatter(pred_data, trial_start, trial_stop, save_path):
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
