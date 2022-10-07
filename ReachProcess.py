""" Script to split experimental reaching videos from single videos of 3 cameras to 3 videos of each camera. Provide a path to find un-split videos over"""
from reachprocess.reachprocess import ReachProcess as r


def run_main(path, DLT_path, path_to_DLC):
    r(path, '14', split=False, predict=False, transform=True, save_all=False, DLC_path=path_to_DLC, DLT_path=DLT_path,
      shuffle='2', resnet_version='101', gpu_num='3', manual_extraction=False)


if __name__ == "__main__":
    path = "/clusterfs/NSDS_data/brnelson/PNS_data/RM14/**/*.mp4"
    DLT_path = "/clusterfs/NSDS_data/brnelson/DLT_files/test_dlt.csv"
    path_to_DLC = "/clusterfs/NSDS_data/brnelson/DLC/LabLabel-Lab-2020-10-27/config.yaml"
    run_main(path, DLT_path, path_to_DLC)

