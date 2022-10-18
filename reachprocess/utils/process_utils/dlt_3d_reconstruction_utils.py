"""
Functions intended to import and transform behavioral reaching data from ReachMaster experiments into 3-D kinematics
Use with DLT co-effecients obtained through easyWand or other procedures + multi-camera predictions from DLC
Author: Brett Nelson, NSDS Lab 2020

"""
import os
import numpy as np
import glob
import pandas as pd
import pdb
from tqdm import tqdm
import reachprocess.utils.kinematic_utils.kinematic_utilities as k_u
import reachprocess.utils.kinematic_utils.viz_utils as vu


# dlt reconstruct adapted by An Chi Chen, using DLT transformation matrix obtained from DLTdv5 (Tyson Hedrick). 
def dlt_reconstruct(c, camPts, weights):
    """
    Function to reconstruct multi-camera predictions from 2-D camera space into 3-D euclidean space
    Credit: adapted by An Chi Chen, using DLT transformation matrix obtained from DLTdv5 by Tyson Hedrick
    Parameters
    ----------
    c : list or array of DLT co-effecients for the camera system in question
    camPts : array of points from the camera system (can be 2, 3 cameras etc)
    weights : bool, option to use p-values from camera predictions to weight transform

    Returns
    -------
    xyz : array of positions in 3-D space for N bodyparts over T timeframe
    """
    # number of frames
    nFrames = len(camPts)
    # number of cameras
    nCams = len(camPts[0]) / 2

    # setup output variables
    xyz = np.empty((nFrames, 3))
    rmse = np.empty((nFrames))
    # process each frame
    for i in range(nFrames):

        # get a list of cameras with non-NaN [u,v]
        cdx_size = 0
        cdx_temp = np.where(np.isnan(camPts[i, 0:int(nCams * 2) - 1:2]) == False, 1, 0)
        for x in range(len(cdx_temp)):
            if cdx_temp[x - 1] == 1:
                cdx_size = cdx_size + 1
        cdx = np.empty((1, cdx_size))
        for y in range(cdx_size):
            cdx[0][y] = y + 1

        # if we have 2+ cameras, begin reconstructing
        if cdx_size >= 2:

            # initialize least-square solution matrices
            m1 = np.empty((cdx_size * 2, 3))
            m2 = np.empty((cdx_size * 2, 1))

            temp1 = 1
            temp2 = 1
            for z in range(cdx_size * 2):
                if z % 2 == 0:
                    m1[z, 0] = camPts[i, (temp1 * 2) - 2] * c[8, (temp1 - 1)] - c[0, (temp1 - 1)]
                    m1[z, 1] = camPts[i, (temp1 * 2) - 2] * c[9, (temp1 - 1)] - c[1, (temp1 - 1)]
                    m1[z, 2] = camPts[i, (temp1 * 2) - 2] * c[10, (temp1 - 1)] - c[2, (temp1 - 1)]
                    m2[z, 0] = c[3, temp1 - 1] - camPts[i, (temp1 * 2) - 2]
                    temp1 = temp1 + 1
                else:
                    m1[z, 0] = camPts[i, (temp2 * 2) - 1] * c[8, temp2 - 1] - c[4, temp2 - 1]
                    m1[z, 1] = camPts[i, (temp2 * 2) - 1] * c[9, temp2 - 1] - c[5, temp2 - 1]
                    m1[z, 2] = camPts[i, (temp2 * 2) - 1] * c[10, temp2 - 1] - c[6, temp2 - 1]
                    m2[z, 0] = c[7, temp2 - 1] - camPts[i, (temp2 * 2) - 1]
                    temp2 = temp2 + 1

            # get the least squares solution to the reconstruction
            if isinstance(weights, np.ndarray):
                w = np.sqrt(np.diag(weights[i, :]))
                # print(w.shape,m1.shape,m2.shape)
                m1 = np.matmul(w, m1)
                m2 = np.matmul(w, m2)
            Q, R = np.linalg.qr(m1)  # QR decomposition with qr function
            y = np.dot(Q.T, m2)  # Let y=Q'.B using matrix multiplication
            x = np.linalg.solve(R, y)  # Solve Rx=y
            xyz[i, 0:3] = x.transpose()
            # compute ideal [u,v] for each camera
            uv = np.matmul(m1, xyz[i, 0:3].T) # Get inverse coordinates in camera frame
            # compute the number of degrees of freedom in the reconstruction
            dof = m2.size - 3
            # estimate the root mean square reconstruction error
            rmse[i] = sum(np.sqrt(sum(m2-uv)**2/dof)) / (6 * 100) # reconstruction error in m, 3 camera x,y to average over
    return xyz, rmse



def norm_coordinates(kin_three_vector):
    """ Function to import and transform kinematic data using pre-generated affine transformation. For more information on
    generating this transformation, see ReachPredict3D's documentation on handle matching."""
    ax = -1.0
    by = -1.0
    cz = 1.0
    a = 0.15
    b = 0.15
    c = 0.4
    xkin_three_vector = np.zeros(kin_three_vector.shape)
    xkin_three_vector[:, 0] = kin_three_vector[:, 0] * ax + a
    xkin_three_vector[:, 1] = kin_three_vector[:, 1] * by + b
    xkin_three_vector[:, 2] = kin_three_vector[:, 2] * cz + c
    return np.copy(xkin_three_vector)


def filter_vector_hamming(input_vector, window_length=3.14):
    """ Function to filter input vectors using Hamming-Cosine window. Used exclusively for 3-D trajectories. """
    filtered_vector = np.zeros(input_vector.shape)
    for i in range(0, input_vector.shape[1]):
        win = np.hamming(window_length)
        filtered_vector[:, i] = np.convolve(win / win.sum(), input_vector[:, i], mode='same')
    return filtered_vector


def reconstruct_3d(dlt_coeffs_file, dlc_files, weighted=True):
    """Perform 3-D reconstruction using DLT co-effecients and extracted multi-camera predictions
    Parameters
    ----------
    dlt_coeffs_file : array containing 3x4 matrix of DLT co-effecients, found using easyWand

    dlc_files : list of paths to .csv files (extracted predictions from DLC for each camera)
    weighted : bool, flag to weight contributions of each camera's predictions by their p-value
    Returns
    -------
    xyz_all : N x T array,where N is the number of parts tracked and T is the length of frames in a given video or trial

    """

    # Load DLT Coefficient
    dlt_coefs = np.loadtxt(dlt_coeffs_file, delimiter=",")
    # Get Names of Labels
    first_dataset = np.loadtxt(dlc_files[0], dtype=str, delimiter=',')
    names = first_dataset[1, range(1, first_dataset.shape[1], 3)]
    frames = first_dataset.shape[0] - 3
    cameras = len(dlc_files)
    xyz_all = np.empty([frames, 4, len(names)])
    rmse_all = np.empty([frames,len(names)])
    weights_all = np.empty([frames, len(names)])
    for k in tqdm(range(len(names))):
        # read in data from DLC
        cam_data = np.empty([frames, 2 * cameras], dtype=float)
        weights = np.empty([frames, 2 * cameras], dtype=float)
        csv_index = int((k * 3) + 1)
        for cam in range(cameras):
            col = cam * 2
            cam_data[:, col] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3, usecols=csv_index)
            cam_data[:, col + 1] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3,
                                              usecols=(csv_index + 1))
            weights[:, col] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3,
                                         usecols=(csv_index + 2))
            weights[:, col + 1] = np.loadtxt(dlc_files[cam], dtype=float, delimiter=',', skiprows=3,
                                             usecols=(csv_index + 2))
        # combine
        if weighted:
            xyz, rmse = dlt_reconstruct(dlt_coefs, cam_data, weights)
        else:
            xyz, rmse = dlt_reconstruct(dlt_coefs, cam_data)
        xyz = norm_coordinates(xyz)
        xyz = vu.interpolate_3d_vector_using_probabilities(xyz, np.mean(weights, axis=1)[:,np.newaxis], p_threshold = 0.5, gap_num = 4)
        xyz = filter_vector_hamming(xyz, window_length=3.1)
        xyz_k = np.append(xyz, np.mean(weights, axis=1)[:, np.newaxis], axis=1)
        xyz_all[:, :, k] = xyz_k
        rmse_all[:, k] = rmse
        w = np.mean(weights, axis=1)
        weights_all[:,k] = w
    return xyz_all, names, rmse_all, weights_all, cam_data


def filter_cam_lists(cam_list):
    """ Function to determine if 3-D transformation is possible with prediction files. Function examines the amount
        of files for each camera, to ensure successful DLT transformation.
        Parameters
        -----------
        cam_list : list, list of camera files

        Returns
        ---------
        cu : bool, warning flag
    """
    c1 = cam_list[0]
    c2 = cam_list[1]
    c3 = cam_list[2]
    # compare lengths
    if len(c1) == len(c2) == len(c3):
        cu = False
    else:
        if len(c1) < len(c2) or len(c1) < len(c3):
            print('Camera 1 predictions not yet generated. Please generate using the predict function in ReachProcess.')
        elif len(c2) < len(c1) or len(c2) < len(c3):
            print('Camera 2 predictions not yet generated. Please generate using the predict function in ReachProcess.')
        elif len(c3) < len(c1) or len(c3) < len(c2):
            print('Camera 3 predictions not yet generated. Please generate using the predict function in ReachProcess.')
        cu = True
    return cu


def get_prediction_file_sets_session(prediction_file_path, resnet_version, filter_by_shuffle=None, filtered_predictions=False):
    """ Function to fetch prediction file sets. Intended to work on single directory, of path type
        path/to/video/dir/
        Parameters
        -------------
        prediction_file_path : str, path to prediction data
        resnet_version : str, version of network (50, 101, 151 supported options)
        filter_by_network : bool, option to filter by specific network type

        Returns
        ---------
        cam1_list, cam2_list, cam3_list : list, set of lists containing file paths for each camera's predicted 2-D positions
    """
    cam1_list = []
    cam2_list = []
    cam3_list = []
    prediction_file_path = prediction_file_path + '**.csv' # Search for .csv files.
    all_files = glob.glob(prediction_file_path, recursive=True) # recursively fetch .csv files
    all_files = [x for x in all_files if resnet_version in x]
    if filter_by_shuffle:
        filter_by_shuffle = 'shuffle' + str(filter_by_shuffle)
        all_files = [ x for x in all_files if filter_by_shuffle in x ]
    if filtered_predictions:
        all_files = [x for x in all_files if 'filtered' not in x]
    for file in all_files:
        if 'cam1' in file:
            # find rest of files containing exp names
            files = file.rsplit('/', 1)[1]
            names = str(files.rsplit('_cam1', 1)[0])
            file_list = [file_list for file_list in all_files if names in file_list]
            for s in file_list:
                if "cam1" in s:
                    cam1_list.append(s)
                if "cam2" in s:
                    cam2_list.append(s)
                if "cam3" in s:
                    cam3_list.append(s)
    return cam1_list, cam2_list, cam3_list


def get_3d_coordinates(prediction_file_path, dlt_path, time_vector, resnet_version, shuffle_version='5', save=True, manual_run=False):
    """ Function to generate 3-D predictions of keypoints from multiple camera predictions. Input of predictions
        should be in the deeplabcut format. However, it is possible to edit the method and use other types of
        keypoint prediction.
        Parameters
        -----------
        prediction_file_path : str, path to prediction videos for a given experimental session
        dlt_path : str, path to correct dlt coefficient file
        resnet_version : str, network type (resnets, 50, 101, 151 are currently supported options)
        save : boolean, save the keypoint predictions 3-D dataframe in working directory
    """
    df = pd.DataFrame()
    rmse_df = pd.DataFrame()
    probabilities_df = pd.DataFrame()
    kinematics_df = pd.DataFrame()
    path_to_save = prediction_file_path[0:-6] + 'predictions.csv'
    path_to_save_rmse = prediction_file_path[0:-6] + 'rmse.csv'
    path_to_save_pval = prediction_file_path[0:-6] + 'probabilities.csv'
    path_to_save_kinematics = prediction_file_path[0:-6] + 'kinematics.csv'
    file_path = prediction_file_path[0:-6]
    if os.path.isfile(path_to_save) and manual_run is False:
        df = pd.read_csv(path_to_save, index_col=False)
        rmse_df = pd.read_csv(path_to_save_rmse, index_col=False)
        probabilities_df = pd.read_csv(path_to_save_pval, index_col=False)
        kinematics_df = pd.read_csv(path_to_save_kinematics, index_col=False)
    else:
        cam1_list, cam2_list, cam3_list = get_prediction_file_sets_session(file_path, resnet_version, filter_by_shuffle=shuffle_version)
        cam_list = [cam1_list[0], cam2_list[0], cam3_list[0]]
        cu = filter_cam_lists(cam_list)
        if cu:
            df = 0 
            rmse_df = 0
            print('Error in the video extraction!! Make sure all your files are extracted from  ' + str(file_path))
        xyzAll, labels, rmse_values, p_values, DLC_data = reconstruct_3d(dlt_path, cam_list)
        xyzAll = xyzAll[:,0:3,:]
        # interpolate between points with a low probability using cubic splines
        
        pred_header = ['Handle X', 'Handle Y', 'Handle Z', 'B Handle X', 'B Handle Y', 'B Handle Z', 'Nose X', 
                        'Nose Y', 'Nose Z', 'Right Shoulder X', 'Right Shoulder Y', 'Right Shoulder Z',
                        'Right Forearm X', 'Right Forearm Y', 'Right Forearm Z', 'Right Wrist X', 'Right Wrist Y', 
                        'Right Wrist Z', 'Right Palm X', 'Right Palm Y', 'Right Palm Z', 'Right Index Base X', 'Right Index Base Y', 
                        'Right Index Base Z', 'Right Index Tip X', 'Right Index Tip Y', 'Right Index Tip Z', 'Right Middle Base X', 
                        'Right Middle Base Y', 'Right Middle Base Z', 'Right Middle Tip X', 'Right Middle Tip Y', 'Right Middle Tip Z', 
                        'Right Third Base X', 'Right Third Base Y', 'Right Third Base Z', 'Right Third Tip X', 'Right Third Tip Y', 
                        'Right Third Tip Z', 'Right Fourth Base X', 'Right Fourth Base Y', 'Right Fourth Base Z', 'Right Fourth Tip X', 
                        'Right Fourth Tip Y', 'Right Fourth Tip Z', 'Left Shoulder X', 'Left Shoulder Y', 'Left Shoulder Z', 
                        'Left Forearm X', 'Left Forearm Y', 'Left Forearm Z', 'Left Wrist X', 'Left Wrist Y', 
                        'Left Wrist Z', 'Left Palm X', 'Left Palm Y', 'Left Palm Z', 'Left Index Base X', 'Left Index Base Y', 
                        'Left Index Base Z', 'Left Index Tip X', 'Left Index Tip Y', 'Left Index Tip Z', 'Left Middle Base X', 
                        'Left Middle Base Y', 'Left Middle Base Z', 'Left Middle Tip X', 'Left Middle Tip Y', 'Left Middle Tip Z', 
                        'Left Third Base X', 'Left Third Base Y', 'Left Third Base Z', 'Left Third Tip X', 'Left Third Tip Y', 
                        'Left Third Tip Z', 'Left Fourth Base X', 'Left Fourth Base Y', 'Left Fourth Base Z', 'Left Fourth Tip X', 
                        'Left Fourth Tip Y', 'Left Fourth Tip Z' ]
        df = pd.DataFrame(xyzAll.reshape((xyzAll.shape[0], xyzAll.shape[1] * xyzAll.shape[2])), columns=pred_header)
        labels_rmse = [x + ' rmse' for x in labels]
        labels_p = [x + ' P' for x in labels]
        rmse_df = pd.DataFrame(rmse_values, columns=labels_rmse)
        probabilities_df = pd.DataFrame(p_values, columns=labels_p)  
        kinematics_header_velocity = ['Handle Xv', 'Handle Yv', 'Handle Zv', 'B Handle Xv', 'B Handle Yv', 'B Handle Zv', 'Nose Xv', 
                        'Nose Yv', 'Nose Zv','Right Shoulder Xv', 'Right Shoulder Yv', 'Right Shoulder Zv',
                        'Right Forearm Xv', 'Right Forearm Yv', 'Right Forearm Zv', 'Right Wrist Xv', 'Right Wrist Yv', 
                        'Right Wrist Zv', 'Right Palm Xv', 'Right Palm Yv', 'Right Palm Zv', 'Right Index Base Xv', 'Right Index Base Yv', 
                        'Right Index Base Zv', 'Right Index Tip Xv', 'Right Index Tip Yv', 'Right Index Tip Zv', 'Right Middle Base Xv', 
                        'Right Middle Base Yv', 'Right Middle Base Zv', 'Right Middle Tip Xv', 'Right Middle Tip Yv', 'Right Middle Tip Zv', 
                        'Right Third Base Xv', 'Right Third Base Yv', 'Right Third Base Zv', 'Right Third Tip Xv', 'Right Third Tip Yv', 
                        'Right Third Tip Zv', 'Right Fourth Base Xv', 'Right Fourth Base Yv', 'Right Fourth Base Zv', 'Right Fourth Tip Xv', 
                        'Right Fourth Tip Yv', 'Right Fourth Tip Zv','Left Shoulder Xv', 'Left Shoulder Yv', 'Left Shoulder Zv', 
                        'Left Forearm Xv', 'Left Forearm Yv', 'Left Forearm Z', 'Left Wrist X', 'Left Wrist Yv', 
                        'Left Wrist Zv', 'Left Palm Xv', 'Left Palm Yv', 'Left Palm Zv', 'Left Index Base Xv', 'Left Index Base Yv', 
                        'Left Index Base Zv', 'Left Index Tip Xv', 'Left Index Tip Yv', 'Left Index Tip Zv', 'Left Middle Base Xv', 
                        'Left Middle Base Yv', 'Left Middle Base Zv', 'Left Middle Tip Xv', 'Left Middle Tip Yv', 'Left Middle Tip Zv', 
                        'Left Third Base Xv', 'Left Third Base Yv', 'Left Third Base Zv', 'Left Third Tip Xv', 'Left Third Tip Yv', 
                        'Left Third Tip Zv', 'Left Fourth Base Xv', 'Left Fourth Base Yv', 'Left Fourth Base Zv', 'Left Fourth Tip Xv', 
                        'Left Fourth Tip Yv', 'Left Fourth Tip Zv' ]
        kinematics_header_acceleration = ['Handle Xa', 'Handle Ya', 'Handle Za', 'B Handle Xa', 'B Handle Ya', 'B Handle Za', 'Nose Xa', 
                        'Nose Ya', 'Nose Z','Right Shoulder Xa', 'Right Shoulder Ya', 'Right Shoulder Za',
                        'Right Forearm Xa', 'Right Forearm Ya', 'Right Forearm Za', 'Right Wrist Xa', 'Right Wrist Ya', 
                        'Right Wrist Za', 'Right Palm Xa', 'Right Palm Ya', 'Right Palm Za', 'Right Index Base Xa', 'Right Index Base Ya', 
                        'Right Index Base Za', 'Right Index Tip Xa', 'Right Index Tip Ya', 'Right Index Tip Za', 'Right Middle Base Xa', 
                        'Right Middle Base Ya', 'Right Middle Base Za', 'Right Middle Tip Xa', 'Right Middle Tip Ya', 'Right Middle Tip Za', 
                        'Right Third Base Xa', 'Right Third Base Ya', 'Right Third Base Za', 'Right Third Tip Xa', 'Right Third Tip Ya', 
                        'Right Third Tip Za', 'Right Fourth Base Xa', 'Right Fourth Base Ya', 'Right Fourth Base Za', 'Right Fourth Tip Xa', 
                        'Right Fourth Tip Ya', 'Right Fourth Tip Za', 'Left Shoulder Xa', 'Left Shoulder Ya', 'Left Shoulder Za', 
                        'Left Forearm Xa', 'Left Forearm Ya', 'Left Forearm Za', 'Left Wrist Xa', 'Left Wrist Ya', 
                        'Left Wrist Za', 'Left Palm Xa', 'Left Palm Ya', 'Left Palm Za', 'Left Index Base Xa', 'Left Index Base Ya', 
                        'Left Index Base Za', 'Left Index Tip Xa', 'Left Index Tip Ya', 'Left Index Tip Za', 'Left Middle Base Xa', 
                        'Left Middle Base Ya', 'Left Middle Base Za', 'Left Middle Tip Xa', 'Left Middle Tip Ya', 'Left Middle Tip Za', 
                        'Left Third Base Xa', 'Left Third Base Ya', 'Left Third Base Za', 'Left Third Tip Xa', 'Left Third Tip Ya', 
                        'Left Third Tip Za', 'Left Fourth Base Xa', 'Left Fourth Base Ya', 'Left Fourth Base Za', 'Left Fourth Tip Xa', 
                        'Left Fourth Tip Ya', 'Left Fourth Tip Za' ]
        speed_header = ['Handle S', 'B Handle S', 'Nose S', 'Right Shoulder S', 'Right Forearm S', 'Right Wrist S', 'Right Palm S', 'Right Index Base S', 
                     'Right Index Tip S', 'Right Middle Base S', 'Right Middle Tip S', 'Right Third Base S', 
                    'Right Third Tip S', 'Right Fourth Base S', 'Right Fourth Tip S', 'Left Shoulder S', 'Left Forearm S', 'Left Wrist S', 
                    'Left Palm S', 'Left Index Base S', 'Left Index Tip S', 'Left Middle Base S', 'Left Middle Tip S', 'Left Third Base S',
                    'Left Third Tip S',  'Left Fourth Base S', 'Left Fourth Tip S']
        print(' Calculating kinematic parameters for block ')    
        kinematics_df = k_u.calculate_kinematics_from_position(df, time_vector, kinematics_header_velocity, kinematics_header_acceleration, speed_header)
        print('Finished calculating kinematics!')
        if save:
            df.to_csv(path_to_save, index=False)
            rmse_df.to_csv(path_to_save_rmse, index=False)
            probabilities_df.to_csv(path_to_save_pval, index=False)
            kinematics_df.to_csv(path_to_save_kinematics, index=False)
    return df, rmse_df, probabilities_df, kinematics_df
