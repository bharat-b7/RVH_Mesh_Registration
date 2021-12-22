import sys
import json
import argparse
import multiprocessing as mp
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_ply

sys.path.insert(0, '.')
from lib.body_objectives import batch_reprojection_loss_kinect
from lib.io import load_kinect_exrtinsics, load_kinect_intrinsics, load_config, load_keypoints_2d
from lib.parallel import parallel_map
from lib.smpl.wrapper_naive import SMPLNaiveWrapper
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.priors.th_smpl_prior import get_prior

np.seterr(all="ignore")


def get_joints_3d_initialization(centers, num, smpl_models_path, device="cpu"):
    # Get SMPL faces
    naive_smpl = SMPLNaiveWrapper(model_root=smpl_models_path, gender='male')
    smpl_faces = naive_smpl.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(device)

    # Load pose prior to initialise pose
    prior = get_prior(smpl_models_path, gender='male', device=device)

    # Initialise pose and shape parameters
    batch_sz = centers.shape[0]
    pose_init = torch.zeros((batch_sz, 72))
    pose_init[:, 3:] = prior.mean
    betas, pose, trans = torch.zeros((batch_sz, 300)), pose_init, centers  # init SMPL with the translation

    # Extract 3d joints from smpl model
    smpl = SMPLPyTorchWrapperBatch(smpl_models_path, batch_sz, betas, pose, trans, faces=th_faces).to(device)
    J, face, hands = smpl.get_landmarks()

    if num == 25:
        return J.to(device)
    elif num == 67:
        joints = torch.cat([J, hands], axis=1).to(device)

        return joints
    else:
        joints = torch.cat([J, hands, face], axis=1).to(device)

        return joints


def worker(folder, args, device):
    seq_name = folder.parent
    t_stamp = folder.name

    target_filename = args.keypoints_3d_root / f"{folder}/3D_pose.json"
    target_filename.parent.mkdir(parents=True, exist_ok=True)

    if not(args.force) and target_filename.is_file():
        return False
        # DEBUG
        # with target_filename.open('r') as fp:
        #     prev_res = json.load(fp)

    keypoints_2d_file = args.keypoints_2d_root / f"{folder}/2D_pose.json"
    body_2d, face_2d, hand_l_2d, hand_r_2d = load_keypoints_2d(keypoints_2d_file, device)
    if body_2d is None:
        return False

    if args.hands:
        keypoints_num = 67
        keypoints_2d = torch.cat([body_2d, hand_l_2d, hand_r_2d], axis=1)
    else:
        keypoints_num = 25
        keypoints_2d = body_2d

    scan = load_ply(args.kinect_data_path / f"{folder}/{t_stamp}.ply")
    center = scan[0].mean(dim=0)
    centers = torch.unsqueeze(center, 0).to(device)

    keypoints_3d_init = get_joints_3d_initialization(centers, keypoints_num, args.smpl_models_path, device)
    keypoints_3d = keypoints_3d_init.clone().detach().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([keypoints_3d], 0.005, betas=(0.9, 0.999))

    color_mats, depth2colors = load_kinect_intrinsics(args.kinect_data_path / f"{seq_name}/intrinsics", device)
    poses_reverse, pose_forward = load_kinect_exrtinsics(args.kinect_data_path / f"{seq_name}/extrinsics", device)
    kinect_poses = poses_reverse

    iterations, steps_per_iter = 100, 30

    for it in range(iterations):
        for i in range(steps_per_iter):
            optimizer.zero_grad()
            loss, j_projected = batch_reprojection_loss_kinect(keypoints_2d, keypoints_3d, color_mats, depth2colors, kinect_poses)
            loss.backward()
            optimizer.step()

    res = keypoints_3d.cpu().detach().numpy()[0]

    confidence = keypoints_2d[0, :, 2::3]
    confidence = confidence.mean(dim=1, keepdim=True).cpu().detach().numpy()

    res = np.concatenate([res, confidence], axis=1)
    with target_filename.open('w') as fp:
        json.dump(res.tolist(), fp, indent=4)

    # DEBUG
    # print(np.sum(np.asarray(prev_res) - res))

    return True


def main(args):
    if args.data_split_file is not None:
        with args.data_split_file.open("rb") as fp:
            data_split = pkl.load(fp)

        # TODO remove relative paths
        data_folders = data_split['train']
        data_folders = [Path(v) for v in data_folders]
        data_folders = [v.relative_to('/BS/bharat-4/static00/kinect_data/') for v in data_folders]
    else:
        data_folders = args.kinect_data_path.glob("*/t*.*")
        data_folders = [v.relative_to(args.kinect_data_path) for v in data_folders]

        filtered_data_folders = []
        for data_folder in data_folders[:]:
            # if not('May06_bharat_backpack_lefthand' in data_folder.parent.name):
            #     continue

            keypoints_2d_file = args.keypoints_2d_root / f"{data_folder}/2D_pose.json"
            if not (keypoints_2d_file.is_file()):
                continue

            keypoints_3d_file = args.keypoints_3d_root / f"{data_folder}/3D_pose.json"
            if not(args.force) and keypoints_3d_file.is_file():
                continue

            filtered_data_folders.append(data_folder)
        data_folders = filtered_data_folders

    # Setup the environment
    mp.set_start_method('spawn')
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")

    parallel_map(data_folders, worker, n_jobs=5, const_args={'args': args, 'device': device})


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for lifting 2d keypoints to 3d for kinect based camera setup.")
    # Path to input / output
    parser.add_argument("--config_path", "-c", type=str, default="config.yml",
                        help="Path to yml file with config.")
    parser.add_argument("--kinect-data-path", "-d", type=str,
                        help="Path to input kinect data")
    parser.add_argument("--data-split-file", "-s", type=str, default=None,
                       help="Path to pkl file with data split (default: None)")

    # Rendering parameters
    parser.add_argument("--n-jobs", "-j", type=int, default=5,
                        help="Number of parallel jobs to run (default: 5)")

    # Additional parameters
    parser.add_argument("--hands", action="store_true",
                        help="Lift hand keypoints together with body (default: False)")
    parser.add_argument("--cpu", action="store_true",
                        help="Perform computations on cpu (default: False)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force overwriting existing results (default: False)")

    args = parser.parse_args()
    # Convert str to Path for convenience
    args.kinect_data_path = Path(args.kinect_data_path)
    args.data_split_file = Path(args.data_split_file) if args.data_split_file is not None else None

    # Load config file
    config = load_config(args.config_path)
    args.smpl_models_path = Path(config["SMPL_MODELS_PATH"])
    args.smpl_assets_path = Path(config["SMPL_ASSETS_PATH"])
    args.keypoints_2d_root = Path(config["KEYPOINTS_2D_ROOT"])
    args.keypoints_3d_root = Path(config["KEYPOINTS_3D_ROOT"])

    main(args)
