import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from scipy import linalg
import pandas as pd
from moviepy.editor import *
import cv2
import numpy as np
from errno import EEXIST, ENOENT
import shutil
from scipy import interpolate, signal
from scipy.signal import butter, sosfiltfilt
from csaps import csaps


def get_single_trial(df, date, session, rat):
    rr = df.loc[df['Date'] == date]
    rr = rr.loc[rr['S'] == session]
    new_df = rr.loc[rr['rat'] == rat]
    return new_df


def sample_around_point(list_of_data, n):
    l = []
    for i in list_of_data:
        d = sample_around_n_rand(i, n)
        l.append(d)
    li = np.asarray(l)
    return li


def sample_around_n_rand(i, n):
    d = np.random.uniform(i - .65, i + .65, size=(n, 1))
    d = np.random.permutation(d)
    return d


def oned_plot(X, Y, Z, zeros, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, zeros, zeros, marker='o', color='r', s=6, label='Reach Locations 1-D (X)')
    ax.scatter(zeros, Y, zeros, marker='o', color='g', s=6, label='Reach Locations 1-D (Y)')
    ax.scatter(zeros, zeros, Z, marker='o', color='b', s=6, label='Reach Locations 1-D (Z)')
    ax.scatter(0, 0, 0, marker='x', color='k', s=40, label='Origin')
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='m', s=20, label='Handle Initialization Positions')
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='k', label='Reward Zone')
    offset = [20, 20, 15]
    ax.quiver(20, 10, 15, 1, 0, 0, length=8, linewidths=5, color='r', alpha=0.8)
    ax.text(25, 20, 10, '%s' % ('X'), size=20, zorder=1,
            color='r')
    ax.quiver(20, 10, 15, 0, 1, 0, length=8, linewidths=5, color='g', alpha=0.8)
    ax.text(23, 15, 18, '%s' % ('Y'), size=20, zorder=1,
            color='g')
    ax.quiver(20, 10, 15, 0, 0, 1, length=8, linewidths=5, color='b', alpha=0.8)
    ax.text(24, -5, 30, '%s' % ('Z'), size=20, zorder=1,
            color='b')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    plt.legend()
    plt.title('Reaching Workspace for Fall 2020 Experiments- 3D Points: 1-D Line(s)')
    if savepath:
        plt.savefig('Reaching_1dplanes_FinalF2020.png')
    plt.show()


def single_plot(X, Y, Z, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, Y, Z, marker='o', color='b', s=6, label='Reach Locations 2-D (X-Y Plane)')
    # ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z),color='y')
    ax.scatter(0, 0, 0, marker='x', color='black', s=40, label='Origin')
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='r', s=20, label='Starting Positions')
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='r', label='Reward Zone')
    XN, YN = np.meshgrid(np.ravel(X), np.ravel(Y))
    ZN = np.ravel(Z)
    ax.plot_surface(XN, YN, ZN.reshape(ZN.shape[0], 1), rstride=2, cstride=2, color='b', alpha=0.4, linewidth=0.0001,
                    edgecolors='g')
    ax.set_xlabel('x(mm)')
    ax.set_ylabel('y(mm)')
    ax.set_zlabel('z(mm)')
    plt.legend()
    plt.title('Reaching Workspace for Fall 2020 Experiments- 3D Points')
    if savepath:
        plt.savefig('Reaching_1dplanes_xy.png')
    plt.show()


def twod_plot(X, Y, Z, zeros, x_rewz_s, y_rewz_s, z_rewz_s, savepath=False, surfaces=False):
    sns.set_style("whitegrid", {'axes.grid': False})
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection='3d', label='Reaching Volume Projection')
    ax.scatter(X, Y, Z, marker='o', color='r', s=6, label='Reach Locations 2-D (X-Y)')
    # ax.plot(np.ravel(X),np.ravel(Y),np.ravel(Z),color='b',label='X-Y')
    ax.scatter(X, Z, Y, marker='o', color='g', s=6, label='Reach Locations 2-D (X-Z)')
    # ax.plot(np.ravel(X),np.ravel(Z),np.ravel(Y),color='m',label='X-Z')
    ax.scatter(Z, X, Y, marker='o', color='b', s=6, label='Reach Locations 2-D (Y-Z)')
    # ax.plot(np.ravel(Z),np.ravel(X),np.ravel(Y),color='y',label='Y-Z')
    ax.scatter(0, 0, 0, marker='x', color='k', s=40, label='Origin')
    ax.scatter(x_rewz_s, y_rewz_s, z_rewz_s, marker='x', color='m', s=20, label='Origin')
    # take Reward Zone coordinates from config file, use forward kinematics to transform
    # these values are hard-coded
    x_rewz = [4.3 - 30, 4.0 - 30, 4.0 - 30, 4.3 - 30]
    y_rewz = [24.5, -20.03, 20.03, -24.5]
    z_rewz = np.asarray([23.3, -23.3, 25.1, -25.1]).reshape(4, 1)
    ax.plot_wireframe(x_rewz, y_rewz, z_rewz, color='k', label='Reward Zone')
    # Making Surfaces: Re-sizing/MatPlotlib
    if surfaces:
        XN, YN = np.meshgrid(np.ravel(X), np.ravel(Y))
        ZN = np.ravel(Z)
        ax.plot_surface(XN, YN, ZN.reshape(ZN.shape[0], 1), rstride=1, cstride=1, color='r', alpha=0.2, linewidth=0,
                        edgecolors='r', antialiased=True)
        XN1, YN1 = np.meshgrid(np.ravel(X), np.ravel(Z))
        ZN1 = np.ravel(Y)
        ax.plot_surface(XN1, YN1, ZN1.reshape(ZN1.shape[0], 1), rstride=1, cstride=1, color='g', alpha=0.2, linewidth=0,
                        edgecolors='g', antialiased=True)
        XN2, YN2 = np.meshgrid(np.ravel(Z), np.ravel(X))
        ZN2 = np.ravel(Y)
        ax.plot_surface(XN2, YN2, ZN2.reshape(ZN2.shape[0], 1), rstride=1, cstride=1, color='b', alpha=0.2, linewidth=0,
                        edgecolors='b', antialiased=True)
    ax.quiver(20, 10, 15, 1, 0, 0, length=8, linewidths=5, color='r', alpha=0.8)
    ax.text(25, 20, 10, '%s' % ('X'), size=20, zorder=1,
            color='r')
    ax.quiver(20, 10, 15, 0, 1, 0, length=8, linewidths=5, color='g', alpha=0.8)
    ax.text(23, 15, 18, '%s' % ('Y'), size=20, zorder=1,
            color='g')
    ax.quiver(20, 10, 15, 0, 0, 1, length=8, linewidths=5, color='b', alpha=0.8)
    ax.text(24, -5, 30, '%s' % ('Z'), size=20, zorder=1,
            color='b')
    ax.set_xlabel('x (pos mm)')
    ax.set_ylabel('y (pos mm)')
    ax.set_zlabel('z (pos mm)')
    plt.legend()
    plt.title('Reaching Workspace for Fall 2020 Experiments- 2D Planes (X-Y,Y-Z, X-Z)')
    if savepath:
        plt.savefig('Plane_Scatter_Points.png')
    plt.show()
    return


def xform_coords_euclidean(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z


def inverse_xform_coords(r, theta_y, theta_z):
    xgimbal_xoffset = 168
    ygimbal_to_joint = 64
    ygimbal_yoffset = 100
    zgimbal_to_joint = 47
    zgimbal_zoffset = 117
    x_origin = 1024
    y_origin = 608
    z_origin = 531
    Ax = np.sqrt(
        xgimbal_xoffset ** 2 + r ** 2 - 2 * xgimbal_xoffset * r * np.cos(theta_y) * np.cos(theta_z)
    )
    gammay = -np.arcsin(
        np.sin(theta_y) *
        np.sqrt(
            (r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        ) /
        np.sqrt(
            (xgimbal_xoffset - r * np.cos(theta_y) * np.cos(theta_z)) ** 2 +
            (r * np.sin(theta_y) * np.cos(theta_z)) ** 2
        )
    )
    gammaz = -np.arcsin(r * np.sin(theta_z) / Ax)
    Ay = np.sqrt(
        (ygimbal_to_joint - ygimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_yoffset - ygimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (ygimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Az = np.sqrt(
        (zgimbal_to_joint - zgimbal_to_joint * np.cos(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_to_joint * np.sin(gammay) * np.cos(gammaz)) ** 2 +
        (zgimbal_zoffset - zgimbal_to_joint * np.sin(gammaz)) ** 2
    )
    Ax = np.round((Ax - xgimbal_xoffset) / 50 * 1024 + x_origin, decimals=1)  # bits
    Ay = np.round((Ay - ygimbal_yoffset) / 50 * 1024 + y_origin, decimals=1)  # bits
    Az = np.round((Az - zgimbal_zoffset) / 50 * 1024 + z_origin, decimals=1)  # bits
    return Ax, Ay, Az


def forward_xform_coords(x, y, z):
    Axx = 168
    Ly = 64
    Ayy = 100
    Lz = 47
    Azz = 117
    X0 = 1024
    Y0 = 608
    Z0 = 531
    Ax_est = (x - X0) / (1024 * 50) + Axx
    Ay_est = (y - Y0) / (1024 * 50) + Ayy
    Az_est = (z - Z0) / (1024 * 50) + Azz
    c1 = np.asarray((0, 0, 0))
    c2 = np.asarray((Ly, Ayy, 0))
    c3 = np.asarray((Lz, 0, Azz))
    u = np.asarray((Ly, Ayy, 0)) / np.sqrt(Ly ** 2 + Ayy ** 2)
    v = c3 - np.dot(c3, u) * u
    v = v / np.sqrt(np.dot(v, v))
    w = np.cross(u, v)
    y1 = np.asarray((0, 1, 0))
    z1 = np.asarray((0, 0, 1))
    U2 = np.sqrt(np.sum((c2 - c1) ** 2))
    U3 = np.dot(c3, u)
    V3 = np.dot(c3, v)
    sd = np.dot(c3, c3)
    r3 = np.sqrt(
        Az_est ** 2 + (Ly - Lz) ** 2 - (2 * Az_est * (Ly - Lz) * np.cos(np.pi - np.arccos((Az_est ** 2 + Lz ** 2 - sd)
                                                                                          / (2 * Az_est * Lz)))))
    Pu = (Ly ** 2 - Ay_est ** 2 + U2 ** 2) / (2 * U2)
    Pv = (U3 ** 2 + V3 ** 2 - 2 * U3 * Pu + Ly ** 2 - r3 ** 2) / (2 * V3)
    Pw = np.sqrt(-Pu ** 2 - Pv ** 2 + Ly ** 2)
    Py = Pu * np.dot(u, y1) + Pv * np.dot(v, y1) + Pw * np.dot(w, y1)
    Pz = Pu * np.dot(u, z1) + Pv * np.dot(v, z1) + Pw * np.dot(w, z1)
    gammay_est = np.arcsin(Py / (Ly * np.cos(np.arcsin(Pz / Ly))))
    gammaz_est = np.arcsin(Pz / Ly)
    r = np.sqrt(Axx ** 2 + Ax_est ** 2 - (2 * Axx * Ax_est * np.cos(gammay_est) * np.cos(gammaz_est)))
    dz = np.sin(-gammaz_est)
    dy = np.sin(-gammay_est)
    theta = np.arcsin(dz * Ax_est / r)
    phi = np.arcsin(Ax_est * dy * np.cos(-gammaz_est) / r / np.cos(theta))
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    return r, theta, phi, x, y, z


def euclidean_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    theta = np.arcsin(y / (r * np.sin(phi)))
    return r, theta, phi


def read_from_csv(input_filepath):
    """ Function to read in csv. """
    input_df = pd.read_csv(input_filepath)
    return input_df


def autocorrelate(x, t=1):
    """ Function to compute regular auto correlation using numpy. """
    return np.corrcoef(np.array([x[:-t], x[t:]]))


# Code adapted from Github page of agramfort
def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def filter_vector_hamming(input_vector, window_length=3.14):
    """ Function to filter input vectors using Hamming-Cosine window. Used exclusively for DLC-based inputs, not 3-D
    trajectories. """
    filtered_vector = np.zeros(input_vector.shape)
    for i in range(0, input_vector.shape[1]):
        win = np.hamming(window_length)
        filtered_vector[:, i] = np.convolve(win / win.sum(), input_vector[:, i], mode='same')
    return filtered_vector


def butterworth_filtfilt(input_vector, nyquist_freq, cutoff, filt_order=4):
    sos = butter(filt_order, cutoff / nyquist_freq, output='sos')
    y = sosfiltfilt(sos, input_vector.reshape(input_vector.shape[1], input_vector.shape[0]))
    return y.reshape(y.shape[1], y.shape[0])


def cubic_spline_smoothing(input_vector, spline_coeff=0.1):
    timepoints = np.linspace(0, input_vector.shape[0], input_vector.shape[0])
    smoothed_vector = np.zeros(input_vector.shape)
    for i in range(0, 3):
        try:
            smoothed_vector[:, i] = csaps(timepoints, input_vector[:, i], timepoints,
                                          normalizedsmooth=True,
                                          smooth=spline_coeff)
        except:
            pass
    return smoothed_vector


def filter_vector_median(input_vector, window_length=3):
    filtered_vector = np.zeros(input_vector.shape)
    filtered_vector[2:-3, :] = signal.medfilt(input_vector[2:-3, :], kernel_size=window_length)
    filtered_vector[0:2, :] = input_vector[0:2, :]
    filtered_vector[-3:-1, :] = input_vector[-3:-1, :]
    return filtered_vector


def interpolate_1d_vector(vec, int_kind='cubic'):
    """ Function to interpolate and re-sample over a 1-D vector using a cubic interpolation."""
    idx = np.nonzero(vec)
    vec_int = 0
    if idx[0].any():
        xx = np.arange(0, vec.shape[0], 1)
        fx = interpolate.interp1d(xx[idx], vec[idx], kind=int_kind, assume_sorted=False)
        vec_int = fx(vec)
    return vec_int


def interpolate_3d_vector(xkin_three_vectors, velocity_index, prob_index, gap_num=4):
    """ Function to interpolate and re-sample, using specified indices of outliers, over a full 3-D vector. """
    gap_index = []
    interpolation_number = 0
    idx = np.union1d(velocity_index, prob_index)
    for i in range(0, xkin_three_vectors.shape[1]):
        # This gets your interpolation indices
        if idx.any():  # If there are any interpolation gaps
            # Handle gaps, we have to chunk up the array into each "int" piece
            xx = np.arange(0, xkin_three_vectors.shape[0], 1)
            uvs = xkin_three_vectors[:, i]
            uvs_mask = np.zeros((xkin_three_vectors.shape[0]))
            uvs_mask[idx] = 1
            cz = 0
            for il, id in enumerate(uvs_mask):  # We need to make sure we aren't interpolating over large gaps!!
                if 3 < il < (len(uvs_mask) - 2):  # Need to keep boundaries at beginning to ensure we dont overflow
                    if id == 0:
                        cz += 1
                    if id == 1:  # If we have a non-thresholded value
                        if uvs_mask[il + 1] == 1 and uvs_mask[il + 2] == 1:
                            if 0 < cz < gap_num:
                                interpolation_number += 1
                                gap_index.append([il])
                                ff = interpolate.interp1d(xx[il - cz - 1:il + 1], uvs[il - cz - 1:il + 1],
                                                          kind='linear', assume_sorted=False)
                                xkin_three_vectors[il - cz:il, i] = ff(xx[il - cz:il])  # Take middle values
                                cz = 0
                            if cz == 0:  # no gap
                                continue
    return np.asarray(xkin_three_vectors), interpolation_number, np.squeeze(np.asarray(gap_index))


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


def rescale_frame(framez, percent=150):
    """ Function to rescale video arrays. """
    width = int(framez.shape[1] * percent / 100)
    height = int(framez.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(framez, dim, interpolation=cv2.INTER_AREA)


def mkdir_p(my_path):
    """Creates a directory. equivalent to using mkdir -p on the command line. 

    Returns
    -------
    object
    """
    try:
        os.makedirs(my_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(my_path):
            pass
        else:
            raise


def rm_dir(my_path):
    """Deletes a directory. equivalent to using rm -rf on the command line"""
    try:
        shutil.rmtree(my_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == ENOENT:
            pass
        else:
            raise


def import_robot_data(df_path):
    """ Imports experimental "robot" data used in analyzing reaching behavioral data. """
    df = pd.read_pickle(df_path)
    df = preprocessing(df)
    return df


def preprocessing(df_in):
    robot_df = date_wipe(df_in)
    return robot_df


def date_wipe(df):
    d = []
    for index, row_value in df['Date'].iteritems():
        if "_" in row_value:
            d.append(row_value[4:6])
        else:
            d.append(row_value[5:7])
    df['Date'] = d
    return df
