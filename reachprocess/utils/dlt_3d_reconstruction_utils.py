"""
Functions intended to import and transform behavioral reaching data from ReachMaster experiments into 3-D kinematics
Use with DLT co-effecients obtained through easyWand or other procedures + multi-camera predictions from DLC
Author: Brett Nelson, NSDS Lab 2020

"""
# imports
import numpy as np
import glob
import pandas as pd


# dlt reconstruct adapted by An Chi Chen from DLTdv5 by Tyson Hedrick
def dlt_reconstruct(c, camPts, weights):
    """
    Function to reconstruct multi-camera predictions from 2-D camera space into 3-D euclidean space
    Credit: adapted by An Chi Chen from DLTdv5 by Tyson Hedrick, edited by BN 8/3/2020
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
    rmse = np.empty((nFrames, 1))
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
            xyz_pts = x.transpose()

            xyz[i, 0:3] = xyz_pts
            # compute ideal [u,v] for each camera
            uv = m1 * xyz[i - 1, 0:2].transpose

            # compute the number of degrees of freedom in the reconstruction
            dof = m2.size - 3

            # estimate the root mean square reconstruction error
            rmse[i, 1] = (sum((m2 - uv) ** 2) / dof) ^ 0.5
    return xyz, rmse


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
    for k in range(len(names)):
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
        xyz_k = np.append(xyz, np.mean(weights, axis=1)[:, np.newaxis], axis=1)
        xyz_all[:, :, k] = xyz_k
        rmse_all[:, k] = rmse
    return xyz_all, names, rmse_all


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


def get_prediction_file_sets_session(prediction_file_path, resnet_version, filter_by_network=None):
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
    all_files = glob.glob(prediction_file_path, recursive=True)
    all_files.extend(glob.glob(resnet_version))
    if filter_by_network:
        all_files.extend(glob.glob(filter_by_network))
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


def get_3d_coordinates(prediction_file_path, dlt_path, resnet_version, save=False):
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
    df = pd.Dataframe()
    rmse_df = pd.DataFrame()
    cam1_list, cam2_list, cam3_list = get_prediction_file_sets_session(prediction_file_path, resnet_version)
    for i, val in enumerate(cam1_list):
        cam_list = [cam1_list[i], cam2_list[i], cam3_list[i]]
        cu = filter_cam_lists(cam_list)
        if cu:
            df = 0
            print('Error in the video extraction!! Make sure all your files are extracted.')
            break
        xyzAll, labels, rmse_values = reconstruct_3d(dlt_path, cam_list)
        coords = ['X', 'Y', 'Z']
        scorer = ['Brett Nelson']
        header = pd.MultiIndex.from_product([scorer, labels,
                                             coords],
                                            names=['scorer', 'bodyparts', 'coords'])
        df = pd.DataFrame(xyzAll.reshape((xyzAll.shape[0], xyzAll.shape[1] * xyzAll.shape[2])), columns=header)
        rmse_df = pd.DataFrame(rmse_values, columns=labels)
        if save:
            df.to_csv('kinematics_df.csv')
    return df, rmse_df
