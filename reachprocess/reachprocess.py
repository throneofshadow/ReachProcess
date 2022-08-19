""" This class is intended to automate the splitting and batch prediction of experimental videos from DeepLabCut.
    Written by Brett Nelson, Lawrence Berkeley National Lab, 2022. """

import os
from tqdm import tqdm
import pandas as pd
import reachprocess.utils.video_split_batching as RP_V
import reachprocess.utils.data_extraction_utils as extract_predictions
import deeplabcut
import pickle


class ReachProcess:
    """ Class to split experimental videos, use a pre-trained DLC network to predict positional keypoints, and compile
        the predicted kinematics into use-able 3-D predictions. """

    def __init__(self, input_data_path, split=True, predict=True, analyze=True, visualize=True, make_kinematics = True,
                 save_all=True, DLC_path='', DLT_path=''):
        """ ReachProcess! Initialize with a data path, set your desired output flags, and run!
            Parameters
            ----------
            input_data_path: str, path to data
            split: bool, flag for splitting videos
            predict: bool, flag for predicting videos
            analyze: bool, flag for compiling 3-D predictions
            make_kinematics : bool, flag for compiling kinematic data
            visualize : bool, flag for visualizing 3-D predictions
            DLC_path: str, path to DLC network
            DLT_path: str, path to DLT transformation matrix
            """
        self.data_path = input_data_path
        self.individual_session_predictions, self.total_predictions, self.individual_rmse, self.total_rmse = [], [], [], []
        if split:
            # Find video files with no split videos currently.
            self.unsplit_video_path_list = RP_V.findFilesInFolder(input_data_path)
            self.split_videos()
        if predict:
            self.DLC_path = DLC_path
            self.cam1_files, self.cam2_files, self.cam3_files = RP_V.find_cam_files(self.data_path)
            self.predict_with_deeplabcut()
        if analyze:
            self.DLT_path = DLT_path
            self.extract_predictions_3d(save_all=save_all)
            if save_all:
                self.save_all_predictions()
        if make_kinematics:
            self.create_kinematics_from_predictions()
        if visualize:
            self.visualize_data()

    def split_videos(self):
        """ Function to iterate over un-split video files."""
        for files in tqdm(self.unsplit_video_path_list):
            RP_V.mainrun_split(files)

    def create_kinematics_from_predictions(self):
        """ Function to create kinematics dataframe from 3-D predictions. """
        return

    def visualize_data(self):

        return

    def extract_predictions_3d(self, save_all = True):
        """ Function to iterate over DLC predictions, transform them using a given set of DLT co-effecients,
            and save into a dataframe. """
        self.individual_session_predictions, self.individual_rmse = extract_predictions.get_3d_predictions(self.data_path, self.DLT_path)
        if save_all:
            self.total_predictions.append(self.individual_session_predictions)
            self.total_rmse.append(self.individual_rmse)

    def save_all_predictions(self, pik=False):
        print('Saving All Sessions into a DataFrame. ')
        os.chdir(self.data_path)
        save_path = self.data_path + '/total_dataframe.pkl'
        save_path_csv = self.data_path + '/total_dataframe.csv'
        if pik:
            with open(save_path, 'wb') as output:
                pickle.dump(self.total_predictions, output, pickle.HIGHEST_PROTOCOL)
        total_p = pd.DataFrame(self.total_predictions)
        total_p.to_csv(save_path_csv)

    def run_analysis_videos_deeplabcut(self, cam_video_paths, shuffle=2, train_index = 9,  filter=False, label_video = True):
        print("Starting to extract files..")
        # initialize deeplabcut
        deeplabcut.analyze_videos(self.DLC_path, cam_video_paths, videotype='.mp4', shuffle=shuffle,
                                  trainingsetindex=train_index, save_as_csv=True)
        if filter:
            deeplabcut.filterpredictions(self.DLC_path, cam_video_paths, videotype='.mp4',
                                         filtertype='arima', ARdegree=5, MAdegree=2)
        if label_video:
            deeplabcut.create_labeled_video(self.DLC_path, cam_video_paths, videotype='.mp4')

    def predict_with_deeplabcut(self):
        print('Starting Cam 1')
        self.run_analysis_videos_deeplabcut(self.cam1_files)
        print('Starting Cam 2')
        self.run_analysis_videos_deeplabcut(self.cam2_files)
        print('Starting Cam 3')
        self.run_analysis_videos_deeplabcut(self.cam3_files)
        print('Finished extracting!')
