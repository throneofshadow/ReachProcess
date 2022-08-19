""" This module provides functions to split un-split recorded experimental videos
using ffmpeg and vidgear options
"""
import cv2
from vidgear.gears import WriteGear
import pdb
import numpy as np
import glob


def findFilesInFolder(fullpath):
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)
     Parameters
    ----------
    fullpath: str, path to directory function searches over

    Returns
    -------
    pl: list, paths of videos that need to be split
    """
    ps = []
    pl = []
    for file in glob.glob(fullpath, recursive=True):
        ps. append(file)
    for files in ps:
        if 'cam1' in files:
            pass
        elif 'cam2' in files:
            pass
        elif 'cam3' in files:
            pass
        else:
            pl.append(files)
    return pl


def conver2bgr(frame):
    """Function to convert image to bgr color scheme

        Attributes
        ----------------
        frame: array, openCV generated np array

        Returns
        -------------
        frame: array, color-corrected numpy array

        """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
    return frame


def enhanceImage(frame_in):
    """ Function to enhance a given image's resolution using opencv.
    Parameters
    ----------
    frame_in: array, image to be  enhanced
    Returns
    -------
    frame: array, enhanced image array
    """
    cols, rows, ch = frame_in.shape
    brightness = np.sum(frame_in) / (ch * 255 * cols * rows)
    minimum_brightness = 0.2
    frame = cv2.convertScaleAbs(frame_in, alpha=1, beta=255 * (minimum_brightness - brightness))
    return frame


def mainrun_split(input_video_files):
    """ Function to split experimental videos for analysis. Takes in a single data path, splits and saved
        individual camera angles to their own separate videos.
        Parameters
        ----------
        input_video_files: str, path to individual video file to be split

        Returns
        --------
        """
    input_filename = input_video_files[0]
    no_of_cam = 3
    crf = '2'
    pix_Format = 'yuv420p'
    cap = cv2.VideoCapture(str(input_filename))
    print('opening filename' + str(input_filename))
    if cap.isOpened():
        print("Error opening video file")
        pdb.set_trace()
    fps = int(cap.get(5))
    width = int(cap.get(3) / no_of_cam)
    height = int(cap.get(4))
    output_params = {'-c:v': 'h264', '-crf': crf, '-input_framerate': fps, '-pix_fmt': pix_Format,
                     '-preset': 'fast', '-tune': 'zerolatency', '-output_dimensions': (width, height)}
    print('Start converting...      ', end='', flush=True)
    writers = []
    for i in range(no_of_cam):
        output_filename = input_filename.split('.')[0] + '_cam' + str(i + 1) + '.mp4'
        writers.append(WriteGear(output_filename=output_filename, compression_mode=True,
                                 logging=False, **output_params))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            for i, w in enumerate(writers):
                index = range(width * i, width * i + width)
                frame_ = frame[:, index]
                frame_ = conver2bgr(frame_)
                frame_ = enhanceImage(frame_)
                w.write(frame_)
    for w in writers:
        w.close()
    cap.release()
    cv2.destroyAllWindows()
    print('[DONE]')


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
