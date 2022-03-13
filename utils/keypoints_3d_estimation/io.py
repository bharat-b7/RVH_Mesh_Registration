import json
from pathlib import Path

import cv2
import torch
import trimesh
import yaml
import numpy as np
from pytorch3d import io
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesUV


def load_config(config_path):
    with open(config_path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def load_data(input_path: Path, device: torch.device = "cpu", texture_path: Path = None):
    data = trimesh.load(input_path, process=False)
    if isinstance(data, trimesh.points.PointCloud):
        input_type = "pointcloud"
        verts = torch.Tensor(data.vertices).to(device).unsqueeze(0)
        colors = torch.Tensor(data.colors / 255).to(device).unsqueeze(0)

        input_data = Pointclouds(points=verts, features=colors)
    elif isinstance(data, trimesh.base.Trimesh):
        input_type = "mesh"
        # Load mesh based on file extension
        suffix = input_path.suffix
        if suffix == '.obj':
            # had to use pytorch3d loading instead of trimesh, because
            # the latter doesn't fully parse faces texture indexes
            verts, faces, aux = io.load_obj(input_path)
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
            input_data = Meshes(verts=[verts], faces=[faces.verts_idx], textures=texture).to(device)
        elif suffix == '.ply':
            # Ilya: this part wasn't tested properly + no texture support
            verts = torch.Tensor(data.vertices).to(device).unsqueeze(0)
            faces = torch.Tensor(data.faces).to(device).unsqueeze(0)
            input_data = Meshes(verts=verts, faces=faces)
        else:
            raise RuntimeError(f"Unknown scan format {suffix}")
    else:
        raise RuntimeError(f"Unsupported input type {type(data)}")

    return input_type, input_data


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


