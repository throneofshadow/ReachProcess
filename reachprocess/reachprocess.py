""" This class is intended to automate the splitting and batch prediction of experimental videos from DeepLabCut.
    Written by Brett Nelson, Lawrence Berkeley National Lab, 2022. """

import os
from tqdm import tqdm
import pandas as pd
import utils.process_utils.video_split_batching as RP_V
import utils.process_utils.data_extraction_utils as extract_predictions
import pickle
import deeplabcut
import pdb
import tensorflow as tf


class ReachProcess:
    """ Class to split experimental videos, use a pre-trained DLC network to predict positional keypoints, and compile
        the predicted kinematics into use-able 3-D predictions. """

    def __init__(self, input_data_path, rat, split=True, predict=True, transform=True, save_all=True, DLC_path='',
                 DLT_path='', shuffle='shuffle2', resnet_version='101',manual_extraction=True,
                 gpu_num='3', sample_rate=30000, n_cores=40):
        """ ReachProcess! Initialize with a data path, set your desired output flags, and run!
            Parameters
            ----------
            input_data_path: str, path to data
            split: bool, flag for splitting videos
            predict: bool, flag for predicting videos
            transform: bool, flag for compiling 3-D predictions
            DLC_path: str, path to DLC network
            DLT_path: str, path to DLT transformation matrix
            """
        self.data_path = input_data_path
        self.rat_path = rat
        self.data_path = input_data_path[0:-8]
        self.DLC_shuffle = shuffle
        self.DLC_network_version = resnet_version
        self.manual_extraction = manual_extraction
        self.gpu_num = gpu_num
        self.num_cores = n_cores
        self.sample_rate = sample_rate
        self.session_predictions, self.total_predictions, self.session_rmse, self.total_rmse = [], [], [], []
        if split:
            # Find video files with no split videos currently.
            self.unsplit_video_path_list = RP_V.findFilesInFolder(input_data_path)
            self.split_videos()
        if predict:
            self.DLC_path = DLC_path
            self.cam1_files, self.cam2_files, self.cam3_files = RP_V.find_cam_files(input_data_path)
            self.predict_with_deeplabcut()
        if transform:
            self.DLT_path = DLT_path
            self.extract_predictions_3d()  # Function to ETL data, returns NWB file containing experimental data
        if save_all:
            # check, load for previously saved NWB file
            self.save_all_predictions()  # saves NWB file containing data

    def split_videos(self):
        """ Function to iterate over un-split video files."""
        for files in tqdm(self.unsplit_video_path_list):
            RP_V.mainrun_split(files)

    def extract_predictions_3d(self):
        """ Function to iterate over DLC predictions, transform them using a given set of DLT co-effecients,
            and save into a dataframe. """
        self.session_predictions, self.session_rmse = \
            extract_predictions.get_3d_predictions(self.data_path, self.DLT_path,
                                                   resnet_version=self.DLC_network_version, shuffle=self.DLC_shuffle,
                                                   manual_extraction=self.manual_extraction, n_cores=self.num_cores,
                                                   sample_rate=self.sample_rate)

    def save_all_predictions(self, pik=False):  # Update to NWB-ified functions.
        print('Saving All Sessions into a DataFrame. ')
        os.chdir(self.data_path)
        save_path = self.data_path + '/total_dataframe.pkl'
        save_path_csv = self.data_path + '/total_dataframe.csv'
        if pik:
            with open(save_path, 'wb') as output:
                pickle.dump(self.total_predictions, output, pickle.HIGHEST_PROTOCOL)
        total_p = pd.DataFrame(self.total_predictions)
        total_p.to_csv(save_path_csv)

    def run_analysis_videos_deeplabcut(self, cam_video_paths, shuffle=2, train_index=9, filter=True, label_video=True):
        print("Starting to extract files..")
        # initialize deeplabcut
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_num  # hard-code for now.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        deeplabcut.analyze_videos(self.DLC_path, cam_video_paths, videotype='.mp4', shuffle=shuffle,
                                  trainingsetindex=train_index, save_as_csv=True)
        if filter:
            deeplabcut.filterpredictions(self.DLC_path, cam_video_paths, videotype='.mp4', shuffle=shuffle,
                                         trainingsetindex=train_index, p_bound=0.8,
                                         filtertype='arima', ARdegree=2, MAdegree=2)  # linear exponential smoothing
        if label_video:
            deeplabcut.create_labeled_video(self.DLC_path, cam_video_paths, videotype='.mp4',
                                            trainingsetindex=train_index, shuffle=shuffle)

    def predict_with_deeplabcut(self):
        print('Starting Cam 1')
        self.run_analysis_videos_deeplabcut(self.cam1_files, label_video=False)
        print('Starting Cam 2')
        self.run_analysis_videos_deeplabcut(self.cam2_files, label_video=True)
        print('Starting Cam 3')
        self.run_analysis_videos_deeplabcut(self.cam3_files, label_video=False)
        print('Finished extracting!')
