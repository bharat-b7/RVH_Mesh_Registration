"""
fit smplh to scans

crated by Xianghui, 12, January 2022

the code is tested
"""

import sys, os
sys.path.append(os.getcwd())
import json
import torch
import numpy as np
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from smpl_registration.base_fitter import BaseFitter
from lib.body_objectives import batch_get_pose_obj
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.wrapper_smplh import SMPLHPyTorchWrapperBatchSplitParams


class SMPLHFitter(BaseFitter):
    def fit(self, scans, pose_files, gender='male', save_path=None):
        # Batch size
        batch_sz = len(scans)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender)

        # Load scans and center them. Once smpl is registered, move it accordingly.
        # Do not forget to change the location of 3D joints/ landmarks accordingly.
        th_scan_meshes = self.load_scans(scans)

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 5, 2, 30, 30

        th_pose_3d = None
        if pose_files is not None:
            th_pose_3d = self.load_j3d(pose_files)

            # Optimize pose first
            self.optimize_pose_only(th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, th_pose_3d)

        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes)

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')

    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        hand_prior = HandPrior(self.model_root, type='grab')
        loss['hand'] = torch.mean(hand_prior(smpl.pose))
        if th_pose_3d is not None:
            loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        return loss

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                     betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        iter_for_global = 1
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_pose_only(split_smpl, th_pose_3d, prior_weight)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)

        print('** Optimised smpl pose **')

    def copy_smpl_params(self, smpl, split_smpl):
        # Put back pose, shape and trans into original smpl
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """
        Performs a forward step, given smpl and scan meshes.
        Then computes the losses.
        currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # losses
        loss = dict()
        loss['pose_pr'] = prior(smpl.pose)
        loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        return loss

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 25. ** 2 * cst / (1 + it)
                       }
        return loss_weight


def main(args):
    fitter = SMPLHFitter(args.model_root, debug=args.display)
    fitter.fit([args.scan_path], [args.pose_file], args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    # parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    # parser.add_argument('pose_file', type=str, help='3d body joints file')
    # parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-gender', type=str, default='female')  # can be female
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-mr', '--model_root', default="/BS/xxie2020/static00/mysmpl/smplh")
    args = parser.parse_args()

    # args = lambda: None
    args.scan_path = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/rp_alison_posed_017_30k.obj'
    args.pose_file = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/pose3d/rp_alison_posed_017_30k.json'
    args.display = True
    args.save_path = '/BS/xxie-2/work/MPI_MeshRegistration/test_data'
    args.gender = 'female'

    main(args)