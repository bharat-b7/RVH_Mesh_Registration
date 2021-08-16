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
from lib.io import load_kinect_exrtinsics, load_kinect_intrinsics, load_config, load_joints_2d
from lib.parallel import parallel_map
from lib.smpl.wrapper_naive import SMPLNaiveWrapper
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.priors.th_smpl_prior import get_prior


def get_joints_3d_initialization(centers, num, smpl_models_path, smpl_assets_path, device="cpu"):
    # Get SMPL faces
    naive_smpl = SMPLNaiveWrapper(model_root=smpl_models_path, assets_root=smpl_assets_path, gender='male')
    smpl_faces = naive_smpl.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(device)

    # Load pose prior to initialise pose
    prior = get_prior(smpl_models_path, smpl_assets_path, gender='male', device=device)

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
    else:
        joints_all = torch.cat([J, face, hands], axis=1)
        return joints_all


def worker(folder, device, smpl_models_path, smpl_assets_path, joints_2d_root, joints_3d_root, kinect_data_root):
    np.seterr(all="ignore")
    seq_name = folder.parent
    t_stamp = folder.name

    target_filename = joints_3d_root / f"{folder}/3D_pose.json"
    target_filename.parent.mkdir(parents=True, exist_ok=True)

    if target_filename.is_file():
        return False
        # DEBUG
        # with target_filename.open('r') as fp:
        #     prev_res = json.load(fp)

    joints_2d_file = joints_2d_root / f"{folder}/2D_pose.json"
    joints_2d = load_joints_2d(joints_2d_file, device)
    if joints_2d is None:
        return False
    joints_num = joints_2d.shape[1]

    scan = load_ply(kinect_data_root / f"{folder}/{t_stamp}.ply")
    center = scan[0].mean(dim=0)
    centers = torch.unsqueeze(center, 0).to(device)

    joints_3d = get_joints_3d_initialization(centers, joints_num, smpl_models_path, smpl_assets_path, device)
    joints_3d = joints_3d.clone().detach().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([joints_3d], 0.005, betas=(0.9, 0.999))

    color_mats, depth2colors = load_kinect_intrinsics(kinect_data_root / f"{seq_name}/intrinsics", device)
    poses_reverse, pose_forward = load_kinect_exrtinsics(kinect_data_root / f"{seq_name}/extrinsics", device)
    kinect_poses = poses_reverse

    iterations, steps_per_iter = 100, 30

    for it in range(iterations):
        for i in range(steps_per_iter):
            optimizer.zero_grad()
            loss, j_projected = batch_reprojection_loss_kinect(joints_2d, joints_3d, color_mats, depth2colors,
                                                               kinect_poses)
            loss.backward()
            optimizer.step()

    res = joints_3d.cpu().detach().numpy()

    with target_filename.open('w') as fp:
        json.dump(res[0].tolist(), fp, indent=4)

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
        data_folders = args.kinect_data_path.rglob("t*.*")
        data_folders = [v.relative_to(args.kinect_data_path) for v in data_folders]

    # Setup the environment
    mp.set_start_method('spawn')
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")

    parallel_map(data_folders[:200], worker, n_jobs=5, const_args={'joints_2d_root': args.keyponts_2d_root,
                                                                   'joints_3d_root': args.keyponts_3d_root,
                                                                   'kinect_data_root': args.kinect_data_path,
                                                                   'smpl_models_path': args.smpl_models_path,
                                                                   'smpl_assets_path': args.smpl_assets_path,
                                                                   'device': device})


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
    parser.add_argument("--cpu", action="store_true",
                        help="Perform computations on cpu (default: False)")

    args = parser.parse_args()
    # Convert str to Path for convenience
    args.kinect_data_path = Path(args.kinect_data_path)
    args.data_split_file = Path(args.data_split_file) if args.data_split_file is not None else None

    # Load config file
    config = load_config(args.config_path)
    args.smpl_models_path = Path(config["SMPL_MODELS_PATH"])
    args.smpl_assets_path = Path(config["SMPL_ASSETS_PATH"])
    args.keyponts_2d_root = Path(config["KEYPOINTS_2D_ROOT"])
    args.keyponts_3d_root = Path(config["KEYPOINTS_3D_ROOT"])

    main(args)
