""" This class is intended to automate the splitting and batch prediction of experimental videos from DeepLabCut.
    Written by Brett Nelson, Lawrence Berkeley National Lab, 2022. """

import glob
import os
from tqdm import tqdm
import reachprocess.utils.video_split_batching as RP_V
import deeplabcut
import glob


def find_cam_files(root_dir):
    """
    Function to find cam files for 3-D calibration
    Parameters
    ----------
    root_dir : path directory
    Returns
    -------
       cam arrays
    """
    cam1_array = []
    cam2_array = []
    cam3_array = []
    sig_flag = True
    for file in glob.glob(root_dir, recursive=True):
        path = file.rsplit('/', 1)[0] + '/'
        if "shuffle2" in file:
            print(file + "has been analyzed already! ")
            sig_flag = False
        else:
            sig_flag = True
        if "cam1" in file:  # check and make sure that the files have been split
            if sig_flag:
                cam1_array.append(file)
        elif "cam2" in file:
            if sig_flag:
                cam2_array.append(file)
        elif "cam3" in file:
            if sig_flag:
                cam3_array.append(file)
    return cam1_array, cam2_array, cam3_array


class ReachProcess:
    """ Class to split experimental videos, use a pre-trained DLC network to predict positional keypoints, and compile
        the predicted kinematics into use-able 3-D predictions. """

    def __init__(self, input_data_path, split=True, predict=True, analyze=True, DLC_path='', DLT_path=''):
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
        self.data_path = input_data_path
        if split:
            # Find video files with no split videos currently.
            self.unsplit_video_path_list = RP_V.findFilesInFolder(input_data_path)
            self.split_videos()
        if predict:
            self.DLC_path = DLC_path
            self.cam1_files, self.cam2_files, self.cam3_files = find_cam_files(self.data_path)
            self.predict_with_deeplabcut()
        if analyze:
            self.DLT_path = DLT_path
            self.extract_kinematics()

    def split_videos(self):
        """ Function to iterate over un-split video files."""
        for files in tqdm(self.unsplit_video_path_list):
            RP_V.mainrun_split(files)

    def extract_kinematics(self):
        """ Function to iterate over DLC predictions, transform them using a given set of DLT co-effecients,
            and save into a dataframe. """

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
