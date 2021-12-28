import sys
import json
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

sys.path.insert(0, '.')
from lib.body_objectives import batch_reprojection_loss_vcam
from lib import io
from lib.smpl.wrapper_naive import SMPLNaiveWrapper
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.priors.th_smpl_prior import get_prior


def get_joints_3d_initialization(centers, num, smpl_models_path, smpl_assets_path, device="cpu"):
    # load 25 keypoints from SMPL

    # Get SMPL faces
    naive_smpl = SMPLNaiveWrapper(model_root=smpl_models_path, assets_root=smpl_assets_path, gender='male')
    smpl_faces = naive_smpl.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(device)

    prior = get_prior(smpl_models_path, smpl_assets_path, gender='male', device=device)

    batch_sz = centers.shape[0]
    pose_init = torch.zeros((batch_sz, 72))
    pose_init[:, 3:] = prior.mean
    betas, pose, trans = torch.zeros((batch_sz, 300)), pose_init, centers  # init SMPL with the translation

    smpl = SMPLPyTorchWrapperBatch(smpl_models_path, batch_sz, betas, pose, trans, faces=th_faces).to(device)
    J, face, hands = smpl.get_landmarks()

    if num == 25:
        return J.to(device)
    else:
        joints_all = torch.cat([J, face, hands], axis=1)
        return joints_all


def main(args):
    # Setup the environment
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")
    np.seterr(all="ignore")

    # Load 2d pose
    joints_2d = io.load_joints_2d(args.joints2d_path, device)
    if joints_2d is None:
        print("Incorrect 2d joints")
    joints_num = joints_2d.shape[1]

    # Load scan to pose SMPL model
    scan = io.load_scan(args.scan_path)
    center = scan.verts_list()[0].mean(dim=0)
    centers = torch.unsqueeze(center, 0).to(device)

    # Initialize 3d pose from SMPL
    joints_3d = get_joints_3d_initialization(centers, joints_num, args.smpl_models_path, args.smpl_assets_path, device)
    joints_3d = joints_3d.clone().detach().requires_grad_(True).to(device)

    # Load cameras
    if args.n_views is not None and args.image_size is not None:
        image_size = args.image_size,

        # Initialize cameras with default params and given n_views
        dist = 2.0
        elev = 5
        azim = torch.linspace(-180, 180, args.n_views)
    else:
        with (args.scan_path / "pytorch3d_params_and_cameras.pkl").open("rb") as fp:
            rendering_params = pkl.load(fp)
        image_size = rendering_params["image_size"]
        dist, elev, azim = rendering_params["dist"], rendering_params["elev"], rendering_params["azim"]
    image_size = torch.Tensor([[image_size, image_size]]).to(device)
    cameras = []
    for az in azim:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=az)
        cameras.append(FoVPerspectiveCameras(R=R, T=T, device=device))

    # Setup optimization
    optimizer = torch.optim.Adam([joints_3d], 0.005, betas=(0.9, 0.999))
    iterations, steps_per_iter = 100, 30

    for it in range(iterations):
        for i in range(steps_per_iter):
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss, _ = batch_reprojection_loss_vcam(joints_2d[:, :joints_num], joints_3d, cameras, image_size)
            loss.backward()
            optimizer.step()

    # Save results
    res = joints_3d.cpu().detach().numpy()
    args.joints3d_path.parent.mkdir(parents=True, exist_ok=True)
    with args.joints3d_path.open('w') as fp:
        json.dump(res[0].tolist(), fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for lifting 2d keypoints to 3d for virtual camera based setup.")
    # Path to input / output
    parser.add_argument("scan_path", type=str,
                        help="Path to input scan.")
    parser.add_argument("--config_path", "-c", type=str, default="config.yml",
                        help="Path to yml file with config.")
    input_path = parser.add_mutually_exclusive_group(required=True)
    input_path.add_argument("--renders-path", "-r", type=str,
                            help="Path to folder with the renderings."
                                 "(if not provided 2D_pose.json in scan_path is assumed).")
    input_path.add_argument("--joints2d-path", "-j2", type=str,
                            help="Path to 2d joints predictions for the renderings "
                                 "(if not provided 2D_pose.json in scan_path is assumed).")
    parser.add_argument("--joints3d-path", "-j3", type=str, default=None,
                        help="Path to file for resulting 3d pose.")

    # Rendering parameters
    parser.add_argument("--image-size", "-s", type=int,
                        help="Rendered image size (SxS)."
                             "(If provided - overrides camera parameters file).")
    parser.add_argument("--n-views", "-n", type=int,
                        help="Number of viewpoints used to render the scan from."
                             "(If provided - overrides camera parameters file).")

    # Additional parameters
    parser.add_argument("--cpu", action="store_true",
                        help="Perform computations on cpu (default: False)")

    args = parser.parse_args()
    # Convert str to Path for convenience
    args.scan_path = Path(args.scan_path)
    if args.renders_path is not None:
        args.renders_path = Path(args.renders_path)
        args.joints2d_path = Path(args.joints2d_path) if args.joints2d_path is not None else \
            args.renders_path / "2D_pose.json"
        args.joints3d_path = Path(args.joints3d_path) if args.joints3d_path is not None else \
            args.renders_path / "3D_pose.json"
    else:
        args.joints2d_path = Path(args.joints2d_path)
        args.joints3d_path = Path(args.joints3d_path) if args.joints3d_path is not None else \
            args.scan_path / "3D_pose.json"

    # Load config file
    config = io.load_config(args.config_path)
    args.smpl_models_path = Path(config["SMPL_MODELS_PATH"])
    args.smpl_assets_path = Path(config["SMPL_ASSETS_PATH"])
    
    main(args)
