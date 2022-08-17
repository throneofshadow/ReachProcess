""" This class is intended to automate the splitting and batch prediction of experimental videos from DeepLabCut.
    Written by Brett Nelson, Lawrence Berkeley National Lab, 2022. """


import glob
import os
from tqdm import tqdm
import reachprocess.utils.video_split_batching as RP_V


class ReachProcess:
    """ Class to split experimental videos, use a pre-trained DLC network to predict positional keypoints, and compile
        the predicted kinematics into use-able 3-D predictions. """
    def __init__(self, input_data_path, split=True, predict=True, analyze=True, DLC_path = '', DLT_path = ''):
        """ ReachProcess! Initialize with a data path, set your desired output flags, and run!
            Parameters
            ----------
            input_data_path: str, path to data
            split: bool, flag for splitting videos
            predict: bool, flag for predicting videos
            analyze: bool, flag for compiling 3-D kinematics
            DLC_path: str, path to DLC network
            DLT_path: str, path to DLT transformation matrix
            """
        self.path = input_data_path
        if split:
            # Find video files with no split videos currently.
            self.unsplit_video_path_list = RP_V.findFilesInFolder(input_data_path)
            self.split_videos()
        if predict:
            self.predict_videos()
        if analyze:
            self.extract_kinematics()

    def split_videos(self):
        """ Function to iterate over un-split video files."""
        for files in tqdm(self.unsplit_video_path_list):
            RP_V.mainrun_split(files)

    def predict_videos(self):
        """ Function to iterate and predict keypoints in videos using a pre-trained DLC network."""

    def extract_kinematics(self):
        """ Function to iterate over DLC predictions, transform them using a given set of DLT co-effecients,
            and save into a dataframe. """

