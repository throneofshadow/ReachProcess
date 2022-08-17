""" This class is intended to automate the splitting and batch prediction of experimental videos from DeepLabCut.
    Written by Brett Nelson, Lawrence Berkeley National Lab, 2022. """


import glob
import os
from tqdm import tqdm
import reachprocess.utils.video_split_batching as RP_V


class ReachProcess:
    def __init__(self, input_data_path, split=True, predict=True, analyze=True, DLC_path = ''):
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
        for files in tqdm(self.unsplit_video_path_list):
            RP_V.mainrun_split(files)

    def predict_videos(self):


    def extract_kinematics(self):


