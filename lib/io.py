import json

import torch
import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def load_keypoints_2d(keypoints_2d_file,device="cpu"):
    def prepare_keypoints(data_2d, keypoints_num, keypoints_key, device):
        # returns keypoints reshaped as follows: 1 x N x (3*n_kinects)
        keypoints = []
        for key in sorted(data_2d.keys()):
            keypoints_view = np.array(data_2d[key][keypoints_key], dtype=np.float32)
            # print(keypoints_view.shape, keypoints_num)
            if keypoints_view.ndim == 1 or keypoints_view.shape[0] != keypoints_num:
                keypoints_view = np.zeros((keypoints_num, 3), dtype=np.float32)

            keypoints.append(keypoints_view)
        # keypoints = [np.array(data_2d[key][keypoints_key][:1], dtype=np.float32) for key in sorted(data_2d.keys())]
        keypoints = np.squeeze(np.array(keypoints))

        # if keypoints.ndim < 2 or keypoints.shape[1] == 0:
        #     return None

        # keypoints_num = keypoints.shape[1]
        keypoints = keypoints[np.newaxis, ...]
        keypoints = np.swapaxes(keypoints, 1, 2)
        keypoints = np.reshape(keypoints, (1, keypoints_num, -1))

        keypoints = torch.from_numpy(keypoints).to(device)

        return keypoints

    if not (keypoints_2d_file.is_file()):
        print(f"No joints for {keypoints_2d_file}")
        return None

    with keypoints_2d_file.open('r') as fp:
        keypoints_2d = json.load(fp)

    # hardcoded sizes of keypoint arrays: body 25, hand 21, face 70
    body_2d = prepare_keypoints(keypoints_2d, 25, "pose_keypoints_2d", device)
    if body_2d is None:
        print(f"Bad joints for {keypoints_2d_file}")

    face_2d = prepare_keypoints(keypoints_2d, 70, "face_keypoints_2d", device)

    hand_l_2d = prepare_keypoints(keypoints_2d, 21, "hand_left_keypoints_2d", device)

    hand_r_2d = prepare_keypoints(keypoints_2d, 21, "hand_right_keypoints_2d", device)

    return body_2d, face_2d, hand_l_2d, hand_r_2d


def load_kinect_exrtinsics(extrinsic_folder, device="cpu"):
    """
    Loads poses and compute the inverse transform for reprojecting pc to different kinect views. C is a number of kinects.

    To transform from world coordinate to kinect local coordinate: np.matmul(x - poses[k, :, 3], poses[k, :, 0:3].T)

    Parameters:
        extrinsic_folder
        device

    Returns:
        poses_reverse: (C, 3, 4) transformation of points from world coordinates to local Kinect coordinates
        poses_forward: (C, 3, 4) transformation of points from local Kinect coordinates to world coordinates

    """
    extrinsic_files = extrinsic_folder.glob("**/config.json")
    pose_calibrations = [json.load(file.open('r')) for file in extrinsic_files]
    rotations = [np.array(data['rotation']).reshape((3, 3)) for data in pose_calibrations]
    translations = [np.array(data['translation']).reshape((3, 1)) for data in pose_calibrations]
    rotations_inverse = [np.linalg.inv(R) for R in rotations]

    poses_reverse = [torch.from_numpy(np.concatenate([r, t], axis=1)) for r, t in zip(rotations_inverse, translations)]
    pose_forward = [torch.from_numpy(np.concatenate([r, t], axis=1)) for r, t in zip(rotations, translations)]

    return torch.stack(poses_reverse).type(torch.float).to(device), \
           torch.stack(pose_forward).type(torch.float).to(device)


def load_kinect_intrinsics(intrinsic_folder, device="cpu"):
    """
    Loads intrinsic parameters from calibration data. C is a number of kinects.

    Parameters:
        intrinsic_folder
        device: torch.device or string

    Returns:
        color_intrinsics: (C, 3, 3), camera matrices
        depth2colors: (C, 3, 4), depth to color transformation
    """
    intrinsic_files = intrinsic_folder.glob("**/calibration.json")
    calibrations = [json.load(file.open('r')) for file in intrinsic_files]
    depth2color_rot = [np.array(calib['depth_to_color']['rotation']).reshape((3, 3)) for calib in calibrations]
    depth2color_trans = [np.array(calib['depth_to_color']['translation']).reshape((3, 1)) for calib in calibrations]
    depth2color_combined = \
        [torch.from_numpy(np.concatenate([r, t], axis=1)) for r, t in zip(depth2color_rot, depth2color_trans)]

    color_params = [calib['color'] for calib in calibrations]
    color_mats = []
    for params in color_params:
        mat = np.eye(3)
        mat[0, 0], mat[1, 1] = params['fx'], params['fy']
        mat[0, 2], mat[1, 2] = params['cx'], params['cy']
        color_mats.append(mat)
    color_mats = np.stack(color_mats)

    return torch.from_numpy(color_mats).type(torch.float).to(device), \
           torch.stack(depth2color_combined).type(torch.float).to(device)
