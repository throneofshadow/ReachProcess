""" Code intended to visualize raw data and statistics for DeepLabCut keypoint predictions
    and their associated probabilities, 3-D positions, velocities, speeds and accelerations. These visualization functions
    may be extended to individual segments of timeseries data (such as reaching).  Brett Nelson 8/2022"""
import glob
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import os
from matplotlib import gridspec
import imageio
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
    ax = sns.heatmap(prob_data, cbar_kws={'label': 'Mean Certainty of DLC Predictions (p-value)'})
    plt.hlines(trial_starts, *ax.get_xlim())
    plt.savefig(save_path + '/colorplot/heatmap_probs_trials.png', dpi=1400)
    plt.close()
    ax = sns.heatmap(prob_data, cbar_kws={'label': 'Mean Certainty of DLC Predictions (p-value)'})
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
    plt.savefig(save_path + '/boxplots/acceleration_y.png', dpi=1400)
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
    plt.savefig(save_path + '/timeseries/acceleration_y.png', dpi=1400)
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
    # Make 'test' set of diagnostic d
    # correlation_plots(pred_data, 0, 90000)
    # preprocessing_boxplot(save_path, rmse_data, prob_data)
    # preprocessing_colormaps(save_path, rmse_data, prob_data,sensor_data)
    # preprocessing_timeseries(save_path, pred_data, prob_data, sensor_data)

    # kinematics_boxplot(save_path, kinematics)
    # make_general_kinematic_timeseries_plots(save_path, kinematics)
    behavior_start = sensor_data['r_start'][0]
    behavior_stop = sensor_data['r_stop'][0]
    # for i, times in enumerate(behavior_start):
    #    stop = behavior_stop[i]
    #    trial_path = save_path + '/trials/trial ' + str(i) + '/visualization/'
    #    mkdir_p(trial_path)
    #    mkdir_p(trial_path + '/3d_plots/')
    #make_3d_scatter(pred_data, prob_data, save_path)
    make_3_d_gif_from_plots(save_path, fps_val=10)
    #plot_kinematics_for_gif(save_path, sensor_data, kinematics, prob_data)
    #make_kin_gif_from_plots(save_path, fps_val=10)
    licking_events = make_lick_event(sensor_data['lick'][0])
    behavior_mask = make_behavior_mask(sensor_data['r_start'][0], sensor_data['r_stop'][0],
                                       np.asarray(sensor_data['time'][0]))


def make_3d_scatter(pred_data, prob_data, save_path):
    # pdb.set_trace()
    for n in range(0, pred_data.shape[0]-90000):  # Iterate over entire block of data
        # Check if overlap in lag times for 3-D plot, lag = 5
        if n < 6:
            m = 0
        else:
            m = n - 5
        plt.tight_layout()
        # plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        # ax = fig.add_subplot(gs[0], projection='3d')
        # fig, (ax, ax1) = plt.subplots(2, 1, figsize=(12, 12), subplot_kw={'projection': "3d"},
        #                              gridspec_kw={'height_ratios': [3, 1]})
        ax.set_zlabel('Z (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
        ax.set_xlim([.15, .25])
        ax.set_ylim([0.15, 0.25])
        ax.set_zlim([0.1, 0.2])
        # plt.subplots_adjust(top=0.95, bottom=0.9)
        # ax.text()
        # ax.text(5, 5, "Rat is currently: 404 Status not Found", fontsize="xx-large", va="top")
        for i in range(0, n - m):  # Loop through each value in alpha for plotting
            if i == m-n:  # set marker to star
                marker_point = '*'
            else:
                marker_point = 'o'
            ax.plot(pred_data['Handle X'][m + i:m + i + 1], pred_data['Handle Y'][m + i:m + i + 1],
                    pred_data['Handle Z'][m + i:m + i + 1],
                    color='k', alpha=prob_data['Handle P'][m + i:m + 1 + i].loc[m+i], linestyle='None', marker=marker_point,)
            ax.plot(pred_data['Right Forearm X'][m + i:m + i + 1],
                    pred_data['Right Forearm Y'][m + i:m + i + 1],
                    pred_data['Right Forearm Z'][m + i:m + i + 1],
                    color='b', linestyle='None', marker=marker_point, alpha=prob_data['Forearm 1 P'][m + i:m + i + 1].loc[m+i])
            ax.plot(pred_data['Right Wrist X'][m + i:m + i + 1],
                    pred_data['Right Wrist Y'][m + i:m + i + 1],
                    pred_data['Right Wrist Z'][m + i:m + i + 1],
                    color='dodgerblue', linestyle='None', marker=marker_point,
                    alpha=prob_data['Wrist 1 P'][m + i:m + i + 1].loc[m+i])
            #  alpha=prob_data['Wrist 1 P'][m:n+10],
            ax.plot(pred_data['Right Palm X'][m:n + 10],
                    pred_data['Right Palm Y'][m:n + 10],
                    pred_data['Right Palm Z'][m:n + 10],
                    color='navy', linestyle='None', marker=marker_point, alpha=prob_data['Palm 1 P'].loc[m+i])
            #  alpha=prob_data['Palm 1 P'][m:n+10],
            ax.plot(pred_data['Left Wrist X'][m + i:m + i + 1],
                    pred_data['Left Wrist Y'][m + i:m + i + 1],
                    pred_data['Left Wrist Z'][m + i:m + i + 1],
                    color='chartreuse', linestyle='None', marker=marker_point,
                    alpha=prob_data['Wrist 2 P'].loc[m+i])
            #  alpha=prob_data['Wrist 2 P'][m:n+10],
            ax.plot(pred_data['Left Palm X'][m + i:m + i + 1],
                    pred_data['Left Palm Y'][m + i:m + i + 1],
                    pred_data['Left Palm Z'][m + i:m + i + 1],
                    color='g', linestyle='None', marker=marker_point, alpha=prob_data['Palm 2 P'].loc[m+i])
            # alpha=prob_data['Palm 2 P'][m:n+10],
            ax.plot(pred_data['Left Forearm X'][m + i:m + i + 1],
                    pred_data['Left Forearm Y'][m + i:m + i + 1],
                    pred_data['Left Forearm Z'][m + i:m + i + 1], marker=marker_point,
                    color='limegreen', linestyle='None', alpha=prob_data['Forearm 2 P'][m + i:m + i + 1].loc[m+i])
        # alpha = prob_data['Forearm 2 P'][m:n + 10],
        plt.legend(['Handle', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Left Wrist', 'Left Palm', 'Left Forearm'],
                   loc='upper right')
        plt.margins(0.0005)
        # plt.show()
        #pdb.set_trace()
        # check to see if previous file exists
        filename = save_path + '\\3d_plots\\' + str(n) + '_scatter_trial.png'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename)
        plt.close('all')


def plot_kinematics_for_gif(save_path, sensor_data, kinematics_data, probabilities):
    time = sensor_data['time'][0]
    licking_data = sensor_data['lick'][0]
    for n in range(0, len(time) - 9000):
        if n < 10:
            l = 0
        else:
            l = n - 10
        plt.figure()
        c_time = time[l:n]
        for i, c in enumerate(c_time):
            if c in licking_data:
                plt.vlines(time[i], c='k', ymin=0.0, ymax=1.2, label='Lick')
        for i in range(0, n-l):  # Iterate over lag terms, plot
            plt.scatter(time[l+i:l+i+1], kinematics_data['Left Palm S'][l+i:l+i+1], alpha=probabilities['Palm 2 P'].loc[l+i],
                     c='g')
            plt.scatter(time[l+i:l+i+1], kinematics_data['Right Palm S'][l+i:l+i+1], alpha=probabilities['Palm 1 P'].loc[l+i],
                     c='dodgerblue')
            plt.scatter(time[l+i:l+i+1], kinematics_data['Handle S'][l+i:l+i+1], alpha=probabilities['Handle P'].loc[l+i],c='y')
            plt.scatter(time[l+i:l+i+1], kinematics_data['Nose S'][l+i:l+i+1], c='r', alpha=probabilities['Nose P'].loc[l+i])
            plt.scatter(time[l+i:l+i+1], kinematics_data['Left Wrist S'][l+i:l+i+1], alpha=probabilities['Wrist 2 P'].loc[l+i],
                     c='chartreuse')
            plt.plot(time[l+i:l+i+1], kinematics_data['Right Wrist S'][l+i:l+i+1], c='b',
                     alpha=probabilities['Wrist 1 P'].loc[l+i])
        plt.plot(time[n], kinematics_data['Right Wrist S'][n], color='b', marker='*', label='Right Wrist Speed')
        plt.plot(time[n], kinematics_data['Left Wrist S'][n], color='chartreuse', marker='*', label='Left Wrist Speed')
        plt.plot(time[n], kinematics_data['Nose S'][n], color='r', marker='*', label='Nose Speed')
        plt.plot(time[n], kinematics_data['Left Palm S'][n], color='g', marker='*', label='Left Palm Speed')
        plt.plot(time[n], kinematics_data['Right Palm S'][n], color='dodgerblue', marker='*', label='Right Palm Speed')
        plt.plot(time[n], kinematics_data['Handle S'][n], color='y', marker='*', label='Handle Speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (M/s)')
        plt.legend(loc='upper right')
        plt.ylim([0.0, 1.2])
        plt.savefig(save_path + '/timeseries/' + str(n) + 'kinematics_summary.png')
        plt.close('all')


def make_3_d_gif_from_plots(plot_path, fps_val=10):
    search_path = plot_path + '/3d_plots/'
    images = []
    for file in glob.glob(search_path + '*_scatter_trial.png'):
        images.append(imageio.imread(file))
    imageio.mimsave(plot_path + '/classification_videos/total_3d_movie.mp4', images, fps=fps_val)


def make_kin_gif_from_plots(plot_path, fps_val=10):
    search_path = plot_path + '\\timeseries\\'
    images = []
    for file in glob.glob(search_path + '*kinematics_summary.png'):
        try:
            images.append(imageio.imread(file))
        except:
            pdb.set_trace()
    imageio.mimsave(plot_path + '\\classification_videos\\sensor_movie.mp4', images, fps=fps_val)


def make_combined_video_gif(root_path, fps_val=10):
    search_path = root_path + "\\classification_videos\\"
    videos_to_resize = []
    for file in glob.glob(search_path + '*sensor_movie.mp4'):
        kinematic_plot = imageio.get_reader(file)
    for file in glob.glob(search_path + '*total_3d_movie.mp4'):
        pred_movie = imageio.get_reader(file)
    # Temporary fix, must be able to cd into PNS for this path and get DLC prediction video.
    for file in glob.glob(search_path + '*.mp4'):
        if 'cam2' in file:
            video = imageio.get_reader(file)

    # Create writer object
    new_gif = imageio.get_writer('output.gif')
    number_of_frames = min(video.get_length(), pred_movie.get_length(), kinematic_plot.get_length())
    for frame_number in range(number_of_frames):
        img1 = video.get_next_data()
        img2 = pred_movie.get_next_data()
        img3 = kinematic_plot.get_next_data()
        # here is the magic
        # reshape images to regions of interest, match indices to make a ~ 1080 x 720 px size video.
        #new_image = np.zeros((img1.shape[0] + img2))
        #kinematic_image = np.zeros((img1.shape[0]))
        kinematic_image = np.vstack([img2, img3])# sh
        new_image = np.hstack((img1, kinematic_image))
        new_gif.append_data(new_image)

    video.close()
    pred_movie.close()
    kinematic_image.close()
    new_gif.close()

