""" Code intended to visualize raw data and statistics for DeepLabCut keypoint predictions
    and their associated probabilities, 3-D positions, velocities, speeds and accelerations. These visualization functions
    may be extended to individual segments of timeseries data (such as reaching).  Brett Nelson 8/2022"""
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from tqdm import tqdm
import imageio
import glob
import cv2
from multiprocessing import Pool

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


class Viz:
    def __init__(self, root_dir, DLC_video_path, rat, date, session, pred_data, prob_data, rmse_data, sensor_data,
                 kinematics, n_pools):
        """"
            Method to create visualizations for high-series data (3-D predictions, associated key-point probabilities,
            root mean square error of keypoints, and kinematics of 3-D keypoints). Each visualization is created in a
            tree-structured index (Rat, Date, Session) inside a data-lake (/nsds/cluster/storage for example).
        """
        self.pool = Pool(n_pools) # Initialize number of compute cores to run (local = 1, non-local >1)
        self.pred_data = pred_data
        self.prob_data = prob_data
        self.kinematics_data = kinematics
        self.rmse_data = rmse_data
        self.time = sensor_data['time']
        self.save_path = self.create_directory_for_session(root_dir, rat, date, session, win_dir=False)
        try:
            self.preprocessing_boxplot(self.save_path, rmse_data, prob_data)
        except:
            print('Could not create boxplots')
        try:
            self.preprocessing_colormaps(self.save_path, rmse_data, prob_data, sensor_data)
        except:
            print('Could not create colormaps')
        try:
            self.preprocessing_timeseries(self.save_path, pred_data, prob_data, sensor_data)
        except:
            print('Could not create timeseries plots')
        behavior_start = sensor_data['r_start'][0]  # Get sensor start times (defined by standard deviation of pixels)
        # Make the classification file (for annotation purposes)
        # Check to see if classification file exists (if exists, don't make)
        save_address = self.save_path + '/classification_videos/' + str(rat) + str(date) + str(session) + '_predictions.csv'
        sim_df = self.make_classification_file(behavior_start)
        sim_df.to_csv(save_address, index=False)
        self.make_3d_scatter()
        self.make_3_d_gif_from_plots(self.save_path, fps_val=10)
        self.plot_kinematics_for_gif(self.save_path)
        self.make_kin_gif_from_plots(self.save_path, fps_val=10)
        self.make_combined_video_gif(DLC_video_path, self.save_path, fps_val=10)
        self.pool.join()  # Close pool

    def create_directory_for_session(self, root_dir, rat, date, session, win_dir=False):
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
            process_path = root_dir + "/ReachProcess/" + rat + '/' + date + '/' + session
            mkdir_p(process_path)
            #mkdir_p(process_path+
            mkdir_p(process_path + "/boxplot")
            mkdir_p(process_path + "/colorplot")
            mkdir_p(process_path + "/3d_plots")
            mkdir_p(process_path + "/classification_videos")
            mkdir_p(process_path + "/timeseries")
            mkdir_p(process_path + '/trials')
        return process_path
    def preprocessing_boxplot(self, save_path, rmse_dataframe, prob_data):
        """ Function to create standardized boxplots for keypoint variables within ecosystem. """
        rmse_dataframe.boxplot(fontsize=3, showfliers=False)
        plt.ylabel('RMSE (px)')
        plt.savefig(save_path + '/boxplot/rmse_values_boxplot.png', dpi=1400)
        plt.close()
        try:
            prob_data.boxplot(fontsize=3, showfliers=False)
            plt.ylabel('DLC Confidence (p-value)')
            plt.savefig(save_path + '/boxplot/prob_values_boxplot.png', dpi=1400)
            plt.close()
        except:
            pdb.set_trace()

    def preprocessing_colormaps(self, save_path, rmse_dataframe, prob_data, sensor_data):
        """ Function to plot 2-D colordepth map for data types used in evaluating goodness of fit within data.
            """
        trial_starts = sensor_data['r_start'][0]
        try:
            ax = sns.heatmap(rmse_dataframe,cbar_kws={'label': 'RMSE error (px)'})
            plt.hlines(trial_starts, *ax.get_xlim())
            plt.savefig(save_path + '/colorplot/heatmap_rmse_start_times.png', dpi=1400)
            plt.close()
            ax = sns.heatmap(rmse_dataframe)
            plt.savefig(save_path + '/colorplot/heatmap_rmse.png', dpi=1400)
            plt.close()
        except:
            pdb.set_trace()
        try:
            ax = sns.heatmap(prob_data, cbar_kws={'label': 'Mean Certainty of DLC Predictions (p-value)'})
            plt.hlines(trial_starts, *ax.get_xlim())
            plt.savefig(save_path + '/colorplot/heatmap_probs_start_times.png', dpi=1400)
            plt.close()
            ax = sns.heatmap(prob_data)
            plt.savefig(save_path + '/colorplot/heatmap_probs.png', dpi=1400)
            plt.close()
        except:
            pdb.set_trace()

    def preprocessing_timeseries(self, save_path, pred_data, prob_data, sensor_data, window_times=None):
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


    def get_column_names_kinematics(self, kinematics_data):
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


    def kinematics_boxplot(self, save_path, kinematics_data):
        """ Function to create standardized boxplots for keypoint variable kinematics (velocity, acceleration). """
        names_vx, names_vy, names_vz, names_ax, names_ay, names_az, names_s = self.get_column_names_kinematics(kinematics_data)
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


    def make_general_kinematic_timeseries_plots(self, save_path, kinematics_data):
        """ Function to plot general aspects of the kinematic data in time-series format. """
        names_vx, names_vy, names_vz, names_ax, names_ay, names_az, names_s = self.get_column_names_kinematics(kinematics_data)
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
    def make_lick_event(self, licking_times, window_length=1, num_events=10):
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

    def make_behavior_mask(self, start_times, stop_times, time):
        behavior_mask = np.zeros(time.shape)
        for i, s in enumerate(start_times):
            stop = stop_times[i]
            behavior_mask[s:stop] = 1
        return behavior_mask

    def make_classification_file(self, behavior_start):
        header = ['Trial', 'Start Time', 'Trial?', 'Number Reaches', 'Reach Start Time', 'Reach Stop Time', 'Num Grasps',
                  'Handedness', 'Tug of War', 'Notes']
        trials = np.arange(0, len(behavior_start), 1)
        reach_start_times = np.zeros(len(behavior_start))
        trial_class = np.zeros(len(behavior_start))
        number_reaches = np.zeros(len(behavior_start))
        handedness = np.zeros(len(behavior_start))
        tug_of_war = np.zeros(len(behavior_start))
        num_grasps = np.zeros(len(behavior_start))
        notes = np.zeros(len(behavior_start))
        data = np.array(
            [trials, behavior_start, trial_class, number_reaches, reach_start_times, reach_start_times, num_grasps, handedness,
             tug_of_war, notes]).T
        pdb.set_trace()
        sim_df = pd.DataFrame(data, columns=header)
        return sim_df

    def plot_3d(self, n):
        pred_data = self.pred_data
        prob_data = self.prob_data
        save_path = self.save_path
        # Check if overlap in lag times for 3-D plot, lag = 5
        if n < 6:
            m = 0
        else:
            m = n - 5
        plt.tight_layout()
        plt.figure(figsize=(4, 4))
        ax = plt.axes(projection='3d')
        ax.set_zlabel('Z (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlabel('X (m)')
        ax.set_xlim([.1, .2])
        ax.set_ylim([0.15, 0.25])
        ax.set_zlim([0.14, 0.2])
        for i in range(0, n - m):  # Loop through each value in alpha for plotting
            if i == m - n:  # set marker to star
                marker_point = '*'
            else:
                marker_point = 'o'
            ax.plot(pred_data['Handle X'][m + i:m + i + 1], pred_data['Handle Y'][m + i:m + i + 1],
                    pred_data['Handle Z'][m + i:m + i + 1],
                    color='k', alpha=prob_data['Handle P'][m + i:m + 1 + i].loc[m + i], linestyle='None',
                    marker=marker_point, )
            ax.plot(pred_data['Right Forearm X'][m + i:m + i + 1],
                    pred_data['Right Forearm Y'][m + i:m + i + 1],
                    pred_data['Right Forearm Z'][m + i:m + i + 1],
                    color='b', linestyle='None', marker=marker_point,
                    alpha=prob_data['Forearm 1 P'][m + i:m + i + 1].loc[m + i])
            ax.plot(pred_data['Right Wrist X'][m + i:m + i + 1],
                    pred_data['Right Wrist Y'][m + i:m + i + 1],
                    pred_data['Right Wrist Z'][m + i:m + i + 1],
                    color='dodgerblue', linestyle='None', marker=marker_point,
                    alpha=prob_data['Wrist 1 P'][m + i:m + i + 1].loc[m + i])
            ax.plot(pred_data['Right Palm X'][m:n + 10],
                    pred_data['Right Palm Y'][m:n + 10],
                    pred_data['Right Palm Z'][m:n + 10],
                    color='navy', linestyle='None', marker=marker_point, alpha=prob_data['Palm 1 P'].loc[m + i])
            ax.plot(pred_data['Left Wrist X'][m + i:m + i + 1],
                    pred_data['Left Wrist Y'][m + i:m + i + 1],
                    pred_data['Left Wrist Z'][m + i:m + i + 1],
                    color='chartreuse', linestyle='None', marker=marker_point,
                    alpha=prob_data['Wrist 2 P'].loc[m + i])
            ax.plot(pred_data['Left Palm X'][m + i:m + i + 1],
                    pred_data['Left Palm Y'][m + i:m + i + 1],
                    pred_data['Left Palm Z'][m + i:m + i + 1],
                    color='g', linestyle='None', marker=marker_point, alpha=prob_data['Palm 2 P'].loc[m + i])
            ax.plot(pred_data['Left Forearm X'][m + i:m + i + 1],
                    pred_data['Left Forearm Y'][m + i:m + i + 1],
                    pred_data['Left Forearm Z'][m + i:m + i + 1], marker=marker_point,
                    color='limegreen', linestyle='None', alpha=prob_data['Forearm 2 P'][m + i:m + i + 1].loc[m + i])
        plt.legend(['Handle', 'Right Forearm', 'Right Wrist', 'Right Palm', 'Left Wrist', 'Left Palm', 'Left Forearm'],
                   loc='upper right', fontsize='small')
        plt.margins(0.0005)
        # check to see if previous file exists
        filename = save_path + '/3d_plots/' + str(n) + '_scatter_trial.png'
        if os.path.exists(filename):
            os.remove(filename)
        plt.savefig(filename, dpi=120)
        plt.close('all')

    def make_3d_scatter(self):
        iter_n = np.arange(0, self.pred_data.shape[0], 1)
        self.pool.map_async(self.plot_3d, iter_n)

    def make_kinematics_gif_mp(self):
        iter_n = np.arange(0, self.pred_data.shape[0])
        self.pool.map_async(self.plot_kinematics_for_gif, iter_n)

    def plot_kinematics_for_gif(self, n):
        kinematics_data = self.kinematics_data
        probabilities = self.prob_data
        time = self.sensor_data['time']
        licking_data = self.sensor_data['licking']
        save_path = self.save_path
        if n < 10:
            l = 0
        else:
            l = n - 10
        plt.figure(figsize=(4, 2))
        c_time = time[l:n]
        for i, c in enumerate(c_time):
            if c in licking_data:
                plt.vlines(time[i], c='k', ymin=0.0, ymax=1.2, label='Lick')
        for i in range(0, n - l+5):  # Iterate over lag terms, plot
            plt.scatter(time[l + i:l + i + 1], kinematics_data['Left Palm S'][l + i:l + i + 1],
                        alpha=probabilities['Palm 2 P'].loc[l + i],
                        c='g')
            plt.scatter(time[l + i:l + i + 1], kinematics_data['Right Palm S'][l + i:l + i + 1],
                        alpha=probabilities['Palm 1 P'].loc[l + i],
                        c='dodgerblue')
            plt.scatter(time[l + i:l + i + 1], kinematics_data['Handle S'][l + i:l + i + 1],
                        alpha=probabilities['Handle P'].loc[l + i], c='y')
            plt.scatter(time[l + i:l + i + 1], kinematics_data['Nose S'][l + i:l + i + 1], c='r',
                        alpha=probabilities['Nose P'].loc[l + i])
            plt.scatter(time[l + i:l + i + 1], kinematics_data['Left Wrist S'][l + i:l + i + 1],
                        alpha=probabilities['Wrist 2 P'].loc[l + i],
                        c='chartreuse')
            plt.plot(time[l + i:l + i + 1], kinematics_data['Right Wrist S'][l + i:l + i + 1], c='b',
                     alpha=probabilities['Wrist 1 P'].loc[l + i])
        plt.plot(time[n], kinematics_data['Right Wrist S'][n], color='b', marker='*', label='Right Wrist Speed')
        plt.plot(time[n], kinematics_data['Left Wrist S'][n], color='chartreuse', marker='*', label='Left Wrist Speed')
        plt.plot(time[n], kinematics_data['Nose S'][n], color='r', marker='*', label='Nose Speed')
        plt.plot(time[n], kinematics_data['Left Palm S'][n], color='g', marker='*', label='Left Palm Speed')
        plt.plot(time[n], kinematics_data['Right Palm S'][n], color='dodgerblue', marker='*', label='Right Palm Speed')
        plt.plot(time[n], kinematics_data['Handle S'][n], color='y', marker='*', label='Handle Speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (M/s)')
        plt.legend(loc='upper right', fontsize='x-small')
        plt.ylim([0.0, 1.2])
        plt.savefig(save_path + '/timeseries/' + str(n) + 'kinematics_summary.png', dpi=120)
        plt.close('all')

    def make_3_d_gif_from_plots(self, plot_path, fps_val=10):
        search_path = plot_path + '/3d_plots/'
        images = []
        for file in glob.glob(search_path + '*_scatter_trial.png'):
            images.append(imageio.imread(file))
        imageio.mimsave(plot_path + '/classification_videos/total_3d_movie.mp4', images, fps=fps_val)
        print('3-D GIF made. ')

    def make_kin_gif_from_plots(self, plot_path, fps_val=10):
        search_path = plot_path + '\\timeseries\\'
        images = []
        for file in glob.glob(search_path + '*kinematics_summary.png'):
            try:
                images.append(imageio.imread(file))
            except:
                pdb.set_trace()
        imageio.mimsave(plot_path + '/classification_videos/sensor_movie.mp4', images, fps=fps_val)
        print('Kinematic GIF made. ')

    def make_combined_video_gif(self, DLC_video_path, root_path, fps_val=10):
        search_path = root_path + "/classification_videos/"
        i = 0
        for file in glob.glob(search_path + '*sensor_movie.mp4'):
            kinematic_plot = cv2.VideoCapture(file)
            frame_count = kinematic_plot.get(cv2.CAP_PROP_FRAME_COUNT)
            height = kinematic_plot.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = kinematic_plot.get(cv2.CAP_PROP_FRAME_WIDTH)
            kinematic_frames = np.zeros((frame_count, width, height))
            while kinematic_plot.isOpened():
                ret, frame = kinematic_plot.read()
                kinematic_frames[i, :, :] = frame
                i += 1
            kinematic_plot.release()
            i = 0
        for file in glob.glob(search_path + '*total_3d_movie.mp4'):
            pred_movie = cv2.VideoCapture(file)
            frame_count = pred_movie.get(cv2.CAP_PROP_FRAME_COUNT)
            height = pred_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = pred_movie.get(cv2.CAP_PROP_FRAME_WIDTH)
            pred_frames = np.zeros((frame_count, width, height))
            while pred_movie.isOpened():
                ret, frame = pred_movie.read()
                pred_frames[i, :, :] = frame
                i += 1
            pred_movie.release()
            i = 0
        # Temporary fix, must be able to cd into PNS for this path and get DLC prediction video.
        for file in glob.glob(DLC_video_path):
            if 'cam2' in file:
                video = cv2.VideoCapture(file)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                video_frames = np.zeros((frame_count, width, height))
                i = 0
                while video.isOpened():
                    ret, frame = video.read()
                    video_frames[i, :, :] = frame
                    i += 1
                video.release()
                i = 0
        gif_path = search_path + 'Classification_Video.avi'
        frame_width = 688 + 480  # 1080p
        frame_height = 720
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        new_gif = cv2.VideoWriter(gif_path, apiPreference=0, fourcc=fourcc,
                                  fps=fps_val, frameSize=(frame_width, frame_height))  # Create writer object
        number_of_frames = min(len(video_frames), len(pred_frames), len(kinematic_frames))
        for frame_number in range(number_of_frames):
            img1 = np.squeeze(video_frames[frame_number, :, :])
            img2 = np.squeeze(pred_frames[frame_number, :, :]) # 480 x 640
            img3 = np.squeeze(kinematic_frames[frame_number, :, :])  # 480 x 640
            # here is the magic
            # reshape images to regions of interest, match indices to make a ~ 1080 x 720 px size video.
            new_image = np.zeros((720, 688 + 480, 3)).astype(np.uint8)  # we want to reshape image into this size, keep video aspect ratio.
            kinematic_image = np.zeros((720, 480, 3))  # this is the size of the kinematic plots.
            kinematic_image[0:480, :, :] = img2  # fill with 3-D plots
            kinematic_image[480:720, :, :] = img3
            new_image[0:688, 0:688, :] = img1.astype(np.uint8)
            new_image[:, 688:688 + 480, :] = kinematic_image.astype(np.uint8)
            new_gif.write(new_image)
        new_gif.release()
