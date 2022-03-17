"""
fit SMPL to person point cloud captured by multiple kinects

Author: Xianghui, 13, January 2022
"""
import sys, os
sys.path.append(os.getcwd())
from tqdm import tqdm
from os.path import join, exists
import torch, json
import numpy as np
from psbody.mesh import Mesh, MeshViewer
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Pointclouds, Meshes

from smpl_registration.fit_SMPLH import SMPLHFitter
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from lib.smpl.const import *
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.torch_functions import batch_chamfer
from lib.body_objectives import batch_3djoints_loss


class SMPLHPCloudFitter(SMPLHFitter):
    def fit(self, pc_files, j3d_files, pose_init=None, gender='male', save_path=None):
        batch_sz = len(pc_files)

        # load pclouds
        points, centers = self.load_scans(pc_files)
        pose, betas, trans, flip = None, None, centers, True
        if pose_init is not None and pose_init[0] is not None:
            # load pose init
            pose, betas = self.load_mocap_data(pose_init)
            flip = False
        else:
            print("Warning: no pose file is provided to initialize body orientation, the fitting quality is not guaranteed!")

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, pose, betas, trans, flip)

        # load 3d joints
        joints_3d = self.load_j3d(j3d_files).to(self.device)

        # Optimize pose and shape
        iterations, steps_per_iter = 10, 20
        self.optimize_pose_shape(points, smpl, iterations, steps_per_iter, joints_3d)

        # save outputs
        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, pc_files, smpl, points, self.save_name_base)

    def get_loss_weights(self, phase=None):
        loss_weight = {
            'p2mf': lambda cst, it: 10. ** 2 * cst * (1 + it),
            'chamf': lambda cst, it: 3. ** 2 * cst / (1 + it),
            'beta': lambda cst, it: 10. ** 0 * cst / (1 + it),
            'body_prior': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'joints3d': lambda cst, it: 40 ** 2 * cst / (1 + it),
            'temporal': lambda cst, it: 0.15 ** 2 * cst / (1 + it),
        }
        if phase == 'all':
            # increase chamfer weight
            loss_weight['chamf'] = lambda cst, it: 30. ** 2 * cst / (1 + it)
        return loss_weight

    def optimize_pose_shape(self, pclouds, smpl, iterations, steps_per_iter, joints_3d=None):
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans,
                                      split_smpl.global_pose,
                                      split_smpl.top_betas],
                                     0.01, betas=(0.9, 0.999))
        weight_dict = self.get_loss_weights()
        steps_per_iter = 10
        iter_for_global = 5
        iter_for_kpts = 15  # use only keypoints to optimize pose
        iter_for_tune = 2  # very slow, only do a few iteractions
        iterations = 15  # iterations to optimize all losses

        phase = 'global'
        description = 'Optimizing SMPL global orientation'
        for it in tqdm(range(iterations + iter_for_kpts + iter_for_global + iter_for_tune)):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # steps_per_iter = 10
                description = 'Optimizing SMPL global orientation'
            elif it == iter_for_global:
                description = 'Optimizing all SMPL pose using keypoints'
                optimizer = torch.optim.Adam([split_smpl.trans,
                                              split_smpl.global_pose,
                                              split_smpl.body_pose,
                                              split_smpl.top_betas
                                              ],
                                             0.004, betas=(0.9, 0.999))
            elif it == iter_for_global + iter_for_kpts:
                # Now optimize full SMPL pose with point clouds
                print('Optimizing all SMPL pose')
                description = 'Optimizing all SMPL pose'
                # loop.set_description('Optimizing all SMPL pose')
                # smaller learning rate, just fine tune the hand etc.
                optimizer = torch.optim.Adam([split_smpl.trans,
                                              split_smpl.global_pose,
                                              split_smpl.body_pose],
                                             0.001, betas=(0.9, 0.999))
                phase = 'all'
                weight_dict = self.get_loss_weights(phase)
            else:
                pass
            loop.set_description(description)

            optimizer.zero_grad()

            for i in loop:
                loss_dict = self.forward_pose_shape(pclouds, split_smpl, joints_3d, phase)
                total_loss = self.backward_step(loss_dict, weight_dict, it)
                total_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(split_smpl, pclouds)
        self.copy_smpl_params(split_smpl, smpl)

    def forward_pose_shape(self, points_list, smpl, joints_3d=None, phase=None):
        loss_dict = dict()
        verts, _, _, _ = smpl()

        loss_dict['beta'] = torch.mean((smpl.betas) ** 2)
        prior = get_prior(self.model_root, smpl.gender)
        prior_loss = torch.mean(prior(smpl.pose[:, :SMPL_POSE_PRAMS_NUM]))
        loss_dict['body_prior'] = prior_loss
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss_dict['hand'] = torch.mean(hand_prior(smpl.pose))  # add hand prior if smplh is used

        J, face, hands = smpl.get_landmarks()

        # 3D joints loss
        joints = self.compose_smpl_joints(J, face, hands, joints_3d)
        j3d_loss = batch_3djoints_loss(joints_3d, joints)
        loss_dict['joints3d'] = j3d_loss

        if phase == 'all':
            # add chamfer distance loss
            chamf = batch_chamfer(points_list, verts, bidirectional=False)
            loss_dict['chamf'] = chamf

        elif phase == 'tune':
            # fine tune, use both chamfer and p2mesh
            pclouds = Pointclouds(points_list)
            # chamfer distance from smpl to pc
            chamf = batch_chamfer(points_list, verts, bidirectional=False)
            loss_dict['chamf'] = chamf

            # faces = []
            faces = smpl.faces.repeat(verts.shape[0], 1, 1)
            smpl_meshes = Meshes(verts, faces)
            s2m_face = point_mesh_face_distance(smpl_meshes, pclouds)
            loss_dict['p2mf'] = s2m_face

        return loss_dict

    def save_outputs(self, save_path, pc_files, smpl, pclouds, save_name='smpl'):
        th_smpl_meshes = self.smpl2meshes(smpl)
        mesh_paths, names = self.get_mesh_paths(save_name, save_path, pc_files)
        # save smpl mesh
        self.save_meshes(th_smpl_meshes, mesh_paths)
        # save original pc
        # self.save_pclouds(pclouds, [join(save_path, n) for n in names])
        # Save params
        self.save_smpl_params(names, save_path, smpl, save_name)
        return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy()

    def save_pclouds(self, points, save_paths, colors=None):
        for i, (p, sp) in enumerate(zip(points, save_paths)):
            if colors is not None:
                m = Mesh(p.cpu().numpy(), [], vc=colors[i])
            else:
                m = Mesh(p.cpu().numpy(), [])
            m.write_ply(sp)

    @staticmethod
    def load_scans(pcfiles, device='cuda:0'):
        "load pclouds, no face information"
        points, centers = [], []
        for file in pcfiles:
            pc = Mesh()
            pc.load_from_file(file)
            pc_th = torch.from_numpy(pc.v).float().to(device)
            center = torch.mean(pc_th, 0)
            points.append(pc_th)
            centers.append(center)
        return points, torch.stack(centers, 0)

    def load_mocap_data(self, pose_files):
        """
        load smpl pose detected by FrankMocap
        Args:
            pose_files: a list of json file containing the pose and betas detected by FrankMocap

        Returns:
            poses: (B, 72)
            betas: (B, 10)

        """
        poses, betas = [], []
        for file in pose_files:
            params = json.load(open(file))
            poses.append(torch.tensor(params['pose']))
            betas.append(torch.tensor(params['betas']))
        return torch.stack(poses), torch.stack(betas)

    def viz_fitting(self, smpl, pclouds, ind=0,
                    smpl_vc=np.array([0, 1, 0])):
        verts, _, _, _ = smpl()
        smpl_mesh = Mesh(v=verts[ind].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
        scan_mesh = Mesh(v=pclouds[ind].cpu().detach().numpy(),
                         f=[], vc=smpl_vc)
        self.mv.set_static_meshes([scan_mesh, smpl_mesh])


def main(args):
    fitter = SMPLHPCloudFitter(args.model_root, debug=args.display, hands=args.hands)
    fitter.fit([args.pc_path], [args.j3d_file], [args.pose_init], args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('pc_path', type=str, help='path to the point cloud')
    parser.add_argument('j3d_file', type=str, help='3d body joints file')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('pose_init', type=str, help='init smpl pose, if exist')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument('-gender', type=str, default='male')
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    # args = lambda: None
    # args.pc_path = "data/pc/person.ply"
    # args.j3d_file = "data/pc/3D_test.json"
    # args.pose_init = "data/pc/mocap.json"
    # args.display = True
    # args.gender = 'male'
    # args.save_path = 'data/pc'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)
        
