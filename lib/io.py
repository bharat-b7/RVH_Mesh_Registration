import json
from pathlib import Path

import cv2
import torch
import yaml
import numpy as np
from pytorch3d import io
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV


def load_config(config_path):
    with open(config_path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def load_joints_2d(joints_2d_file, device="cpu"):
    if not (joints_2d_file.is_file()):
        print(f"No joints for {joints_2d_file}")
        return None

    with joints_2d_file.open('r') as fp:
        joints_2d = json.load(fp)

    joints_2d = [np.array(joints_2d[key][:1], dtype=np.float32) for key in sorted(joints_2d.keys())]
    joints_2d = np.squeeze(np.array(joints_2d))

    if joints_2d.ndim < 2:
        print(f"Bad joints ({joints_2d.shape}) for {joints_2d_file}")
        return None

    joints_num = joints_2d.shape[1]
    joints_2d = joints_2d[np.newaxis, ...]
    joints_2d = np.swapaxes(joints_2d, 1, 2)
    joints_2d = np.reshape(joints_2d, (1, joints_num, -1))

    joints_2d = torch.from_numpy(joints_2d).to(device)

    return joints_2d


def load_scan(scan_path: Path, texture_path: Path = None, device: torch.device = "cpu"):
    # Load mesh based on file extension
    suffix = scan_path.suffix
    if suffix == '.obj':
        verts, faces, aux = io.load_obj(scan_path)

        texture = None
        if texture_path is not None:
            if texture_path.is_file():
                # Load image
                texture_image = cv2.imread(str(texture_path))
                # It's important to convert image to float
                texture_image = torch.from_numpy(texture_image.astype(np.float32) / 255)

                # Extract representation needed to create Textures object
                verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
                faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
                texture_image = texture_image[None, ...]  # (1, H, W, 3)

                texture = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
            else:
                Warning("No texture file found for provided .obj scan")

        # Initialise the mesh
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=texture).to(device)
    elif suffix == '.ply':
        # Ilya: this part wasn't tested properly
        reader = io.IO()
        mesh = reader.load_mesh(scan_path, device=device)
    else:
        raise RuntimeError(f"Unknown scan format {suffix}")

    return mesh


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
