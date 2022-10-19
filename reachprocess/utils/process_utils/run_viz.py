""" Script to run visualization_utils in debugging mode (single node of data per run). User can manually specify
    paths to data, run the visualization utilities.
"""
import pdb

import visualization_utils as vu
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')

cwd = "C:\\Users\\bassp\\PycharmProjects\\ReachProcess\\reachprocess\\"
path_to_predictions = 'predictions_s3.csv'
path_to_probabilities = 'probabilities_s3.csv'
path_to_rmse = 'rmse_s3.csv'
path_to_video = 'DLC_cam2_s3.mp4'
path_to_kinematics = 'kinematics_s3.csv'
path_to_experimental_data = 'experimental_df.h5'
pred_data = vu.load_raw_data_from_disk(path_to_predictions)
prob_data = vu.load_raw_data_from_disk(path_to_probabilities)
rmse_data = vu.load_raw_data_from_disk(path_to_rmse)
kinematics = vu.load_raw_data_from_disk(path_to_kinematics)
sensor_data = pd.read_hdf(path_to_experimental_data)
#full_data = pd.concat([pred_data, prob_data, kinematics, sensor_data])c


vu.visualize_data(cwd, cwd, 'RM14', '09202019', 'S3', pred_data, prob_data, rmse_data, sensor_data, kinematics)

# Heatmap: index (time), bodyparts, values (p-val, rmse length), plot ticks on x-axis/grid for "trials"


