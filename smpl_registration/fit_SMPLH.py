"""Fit smplh to scans.

created by Xianghui, 12, January 2022

the code is tested
"""

import os
import sys

sys.path.append(os.getcwd())
from os.path import exists
from pathlib import Path

import torch
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from tqdm import tqdm

from typing import Optional

from lib.body_objectives import batch_3djoints_loss
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from smpl_registration.base_fitter import BaseFitter


class SMPLHFitter(BaseFitter):
    def fit(
        self,
        scans: list[str],
        pose_files: Optional[list[str]],
        gender: str = "male",
        save_path: Optional[str] = None,
    ):
        """Fit smpl to scans.
        
        Args:
            scans: list of paths to scans
            pose_files: list of paths to 3d body joints
            gender: gender of smpl
            save_path: path to save optimized smpl parameters
        """
        # Batch size
        batch_sz = len(scans)

        # Load scans and center them. Once smpl is registered, move it accordingly.
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        th_scan_meshes, centers = self.load_scans(scans, device=device, ret_cent=True)

        # init smpl
        smpl = self.init_smpl(
            batch_sz, gender, trans=centers
        )  # add centers as initial SMPL translation

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 5, 5, 30, 30

        th_pose_3d = None
        if pose_files is not None:
            th_pose_3d = self.load_j3d(pose_files)

            # Optimize pose first
            self.optimize_pose_only(
                th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, th_pose_3d
            )

        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(
                save_path, scans, smpl, th_scan_meshes, save_name="smplh" if self.hands else "smpl"
            )

    def optimize_pose_shape(
        self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None
    ):
        """Optimize pose and shape of smpl."""
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description("Optimizing SMPL")
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = f"Iter: {i}"
                for k in loss_dict:
                    l_str += ", {}: {:0.4f}".format(
                        k, weight_dict[k](loss_dict[k], it).mean().item()
                    )
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        print("** Optimised smpl pose and shape **")

    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender, device=self.device)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss["s2m"] = point_mesh_face_distance(
            th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list())
        )
        loss["m2s"] = point_mesh_face_distance(
            th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list())
        )
        loss["betas"] = torch.mean(smpl.betas**2)
        loss["pose_pr"] = torch.mean(prior(smpl.pose))
        if self.hands:
            hand_prior = HandPrior(self.model_root, type="grab")
            loss["hand"] = torch.mean(hand_prior(smpl.pose))  # add hand prior if smplh is used
        if th_pose_3d is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss["pose_obj"] = j3d_loss
            # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        return loss

    def compose_smpl_joints(self, J, face, hands, th_pose_3d):
        if th_pose_3d.shape[1] == 25:
            joints = J
        else:
            joints = torch.cat([J, face, hands], 1)
        return joints

    def optimize_pose_only(
        self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d, prior_weight=None
    ):
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam(
            [split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose],
            0.02,
            betas=(0.9, 0.999),
        )

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        iter_for_global = 5
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print("Optimizing SMPL global orientation")
                loop.set_description("Optimizing SMPL global orientation")
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print("Optimizing SMPL pose only")
                loop.set_description("Optimizing SMPL pose only")
                optimizer = torch.optim.Adam(
                    [
                        split_smpl.trans,
                        split_smpl.top_betas,
                        split_smpl.global_pose,
                        split_smpl.body_pose,
                    ],
                    0.02,
                    betas=(0.9, 0.999),
                )
            else:
                loop.set_description("Optimizing SMPL pose only")

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_pose_only(split_smpl, th_pose_3d, prior_weight)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it / 2)
                tot_loss.backward()
                optimizer.step()

                l_str = f"Iter: {i}"
                for k in loss_dict:
                    l_str += ", {}: {:0.4f}".format(
                        k, weight_dict[k](loss_dict[k], it).mean().item()
                    )
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)

        print("** Optimised smpl pose **")

    def copy_smpl_params(self, smpl, split_smpl):
        # Put back pose, shape and trans into original smpl
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """Performs a forward step, given smpl and scan meshes.

        Then computes the losses. currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender, device=self.device)

        # losses
        loss = dict()
        # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        # 3D joints loss
        J, face, hands = smpl.get_landmarks()
        joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        loss["pose_pr"] = torch.mean(prior(smpl.pose))
        loss["betas"] = torch.mean(smpl.betas**2)
        j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
        loss["pose_obj"] = j3d_loss
        return loss

    def get_loss_weights(self):
        """Set loss weights."""
        loss_weight = {
            "s2m": lambda cst, it: 20.0**2 * cst * (1 + it),
            "m2s": lambda cst, it: 20.0**2 * cst / (1 + it),
            "betas": lambda cst, it: 10.0**1.0 * cst / (1 + it),
            "offsets": lambda cst, it: 10.0**-1 * cst / (1 + it),
            "pose_pr": lambda cst, it: 10.0**-5 * cst / (1 + it),
            "hand": lambda cst, it: 10.0**-5 * cst / (1 + it),
            "lap": lambda cst, it: cst / (1 + it),
            "pose_obj": lambda cst, it: 10.0**2 * cst / (1 + it),
        }
        return loss_weight


def main(args):
    fitter = SMPLHFitter(args.model_root, debug=args.display, hands=args.hands)
    if args.pose_file is None:
        pose_files = None
    else:
        pose_files = [args.pose_file]
    fitter.fit([args.scan_path], pose_files, args.gender, args.save_path)


if __name__ == "__main__":
    import argparse

    from utils.configs import load_config

    parser = argparse.ArgumentParser(description="Run Model")
    parser.add_argument("scan_path", type=str, help="path to the 3d scans")
    parser.add_argument("save_path", type=str, help="save path for all scans")
    parser.add_argument("-gender", type=str, default="male")  # can be female
    parser.add_argument("--pose_file", type=str, help="3d body joints file", default=None)
    parser.add_argument("--display", default=False, action="store_true")
    parser.add_argument(
        "--config-path", "-c", type=Path, default="config.yml", help="Path to yml file with config"
    )
    parser.add_argument(
        "-hands", default=False, action="store_true", help="use SMPL+hand model or not"
    )
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.pose_file = 'data/mesh_1/3D_joints_all.json'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_ASSETS_PATH"])

    main(args)
