A library to automate the splitting, prediction, and compilation of rat reaching kinematics from video, microcontroller,
and other experimental data sources.

## Features
- utilizes state-of-the-art technique in markerless pose estimation to extract predicted keypoints from video data
- leverages predicted keypoint positions and the Direct Linear Transform of the camera state-space to estimate 3-D positions
- calculates kinematic variables for each keypoint, such as speed
- renders predicted 3-D variables, kinematics along-side video data in video format
- saves all data within the Neurodata Without Borders ecosystem (see https://www.nwb.org/)


## Requires
- to extract keypoints, a GPU is recommended to speed the process up
- scripts are provided to run ReachProcess in HPC-performance mode, see the Scripts folder


### Example Output from ReachProcess
![alt text](https://github.com/throneofshadow/ReachAnnotation/blob/main/reachannotate_example.png?raw=True)