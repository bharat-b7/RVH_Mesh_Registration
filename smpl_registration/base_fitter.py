"""
base smpl fitter class to handle data io, load smpl, output saving etc. so that they can be easily reused later
this can be inherited for fitting smplh, smph+d to scan, kinect point clouds etc.

Author: Xianghui, 12, January 2022
"""
import torch
from os.path import join, split
from pytorch3d.structures import Meshes
from pytorch3d.io import save_ply, load_ply, load_obj
import pickle as pkl
import numpy as np
import json
from psbody.mesh import MeshViewer, Mesh
from lib.smpl.priors.th_hand_prior import mean_hand_pose
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl_paths import SmplPaths
from lib.smpl.wrapper_smplh import SMPLHPyTorchWrapperBatch
from lib.smpl.const import *
from lib.body_objectives import HAND_VISIBLE


class BaseFitter(object):
    def __init__(self, model_root, device='cuda:0', save_name='smpl', debug=False):
        self.model_root = model_root # root path to the smpl or smplh model
        self.debug = debug
        self.save_name = save_name # suffix of the output file
        self.device = device
        if debug:
            self.mv = MeshViewer()

    def fit(self, scans, pose_files, gender='male', save_path=None):
        raise NotImplemented

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        """
        optimize smpl pose and shape parameters together
        Args:
            th_scan_meshes:
            smpl:
            iterations:
            steps_per_iter:
            th_pose_3d:

        Returns:

        """
        raise NotImplemented

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        """
        Initially we want to only optimize the global rotation of SMPL. Next we optimize full pose.
        We optimize pose based on the 3D keypoints in th_pose_3d.
        Args:
            th_scan_meshes:
            smpl:
            iterations:
            steps_per_iter:
            th_pose_3d:
            prior_weight:

        Returns:

        """
        raise NotImplemented

    def init_smpl(self, batch_sz, gender, pose=None, betas=None, trans=None, flip=False):
        """
        initialize a smpl batch model
        Args:
            batch_sz:
            gender:
            flip: rotate smpl around z-axis by 180 degree

        Returns: batch smplh model

        """
        sp = SmplPaths(gender=gender)
        smpl_faces = sp.get_faces()
        th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(self.device)
        num_betas = 10
        prior = get_prior(self.model_root, gender=gender)
        pose_init = torch.zeros((batch_sz, SMPLH_POSE_PRAMS_NUM))
        hand_mean = mean_hand_pose(self.model_root)
        if pose is None:
            pose_init[:, 3:SMPLH_HANDPOSE_START] = prior.mean
            hand_init = torch.tensor(hand_mean, dtype=torch.float).to(self.device)
            pose_init[:, SMPLH_HANDPOSE_START:] = hand_init
            if flip:
                pose_init[:, 2] = np.pi
        else:
            pose_init[:, :SMPLH_HANDPOSE_START] = pose[:, :SMPLH_HANDPOSE_START]
            if pose.shape[1] == SMPLH_POSE_PRAMS_NUM:
                pose_init[:, SMPLH_HANDPOSE_START:] = pose[:, SMPLH_HANDPOSE_START:]
        beta_init = torch.zeros((batch_sz, num_betas)) if betas is None else betas
        trans_init = torch.zeros((batch_sz, 3)) if trans is None else trans
        betas, pose, trans = beta_init, pose_init, trans_init
        # Init SMPL, pose with mean smpl pose, as in ch.registration
        smpl = SMPLHPyTorchWrapperBatch(self.model_root, batch_sz, betas, pose, trans, faces=th_faces,
                                        num_betas=num_betas).to(self.device)
        return smpl

    @staticmethod
    def load_smpl_params(pkl_files):
        """
        load smpl params from file
        Args:
            pkl_files:

        Returns:

        """
        pose, betas, trans = [], [], []
        for spkl in pkl_files:
            smpl_dict = pkl.load(open(spkl, 'rb'), encoding='latin-1')
            p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
            pose.append(p) # smplh only allows 10 shape parameters
            # if len(b) == 10:
            #     temp = np.zeros((300,))
            #     temp[:10] = b
            #     b = temp.astype('float32')
            betas.append(b)
            trans.append(t)
        pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)
        return pose, betas, trans

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

    def save_outputs(self, save_path, scan_paths, smpl, th_scan_meshes, save_name='smpl'):
        th_smpl_meshes = self.smpl2meshes(smpl)
        mesh_paths, names = self.get_mesh_paths(save_name, save_path, scan_paths)
        self.save_meshes(th_smpl_meshes, mesh_paths)
        self.save_meshes(th_scan_meshes, [join(save_path, n) for n in names])
        # Save params
        self.save_smpl_params(names, save_path, smpl, save_name)
        return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy()

    def smpl2meshes(self, smpl):
        "convert smpl batch to pytorch3d meshes"
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        return th_smpl_meshes

    def get_mesh_paths(self, save_name, save_path, scan_paths):
        names = [split(s)[1] for s in scan_paths]
        # Save meshes
        mesh_paths = []
        for n in names:
            if n.endswith('.obj'):
                mesh_paths.append(join(save_path, n.replace('.obj', f'_{save_name}.ply')))
            else:
                mesh_paths.append(join(save_path, n.replace('.ply', f'_{save_name}.ply')))
        return mesh_paths, names

    def save_smpl_params(self, names, save_path, smpl, save_name):
        for p, b, t, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                              smpl.trans.cpu().detach().numpy(), names):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t}
            pkl.dump(smpl_dict, open(join(save_path, n.replace('.obj', f'_{save_name}.pkl')), 'wb'))

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def save_meshes(meshes, save_paths):
        print('Mesh saved at', save_paths[0])
        for m, s in zip(meshes, save_paths):
            save_ply(s, m.verts_list()[0].cpu(), m.faces_list()[0].cpu())

    def load_j3d(self, pose_files):
        """
        load 3d body keypoints
        Args:
            pose_files: json files containing the body keypoints location

        Returns: a list of body keypoints

        """
        th_no_right_hand_visible, th_no_left_hand_visible, th_pose_3d = [], [], []
        for pose_file in pose_files:
            with open(pose_file) as f:
                pose_3d = json.load(f)
                th_no_right_hand_visible.append(
                    np.max(np.array(pose_3d['hand_right_keypoints_3d']).reshape(-1, 4)[:, 3]) < HAND_VISIBLE)
                th_no_left_hand_visible.append(
                    np.max(np.array(pose_3d['hand_left_keypoints_3d']).reshape(-1, 4)[:, 3]) < HAND_VISIBLE)

                pose_3d['pose_keypoints_3d'] = torch.from_numpy(
                    np.array(pose_3d['pose_keypoints_3d']).astype(np.float32).reshape(-1, 4))
                pose_3d['face_keypoints_3d'] = torch.from_numpy(
                    np.array(pose_3d['face_keypoints_3d']).astype(np.float32).reshape(-1, 4))
                pose_3d['hand_right_keypoints_3d'] = torch.from_numpy(
                    np.array(pose_3d['hand_right_keypoints_3d']).astype(np.float32).reshape(-1, 4))
                pose_3d['hand_left_keypoints_3d'] = torch.from_numpy(
                    np.array(pose_3d['hand_left_keypoints_3d']).astype(np.float32).reshape(-1, 4))
            th_pose_3d.append(pose_3d)
        return th_pose_3d

    @staticmethod
    def load_scans(scans, device='cuda:0'):
        verts, faces, centers = [], [], []
        for scan in scans:
            print('scan path ...', scan)
            if scan.endswith('.ply'):
                v, f = load_ply(scan)
            else:
                v, f, _ = load_obj(scan)
                f = f[0]  # see pytorch3d doc
            verts.append(v)
            faces.append(f)
        th_scan_meshes = Meshes(verts, faces).to(device)
        return th_scan_meshes

    def viz_fitting(self, smpl, th_scan_meshes, ind=0,
                    smpl_vc=np.array([0, 1, 0])):
        verts, _, _, _ = smpl()
        smpl_mesh = Mesh(v=verts[ind].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
        scan_mesh = Mesh(v=th_scan_meshes.verts_list()[ind].cpu().detach().numpy(),
                         f=th_scan_meshes.faces_list()[ind].cpu().numpy(), vc=smpl_vc)
        self.mv.set_static_meshes([scan_mesh, smpl_mesh])

    def copy_smpl_params(self, split_smpl, smpl):
        smpl.pose.data[:, :3] = split_smpl.global_pose.data
        smpl.pose.data[:, 3:66] = split_smpl.body_pose.data
        smpl.pose.data[:, 66:] = split_smpl.hand_pose.data
        smpl.betas.data[:, :2] = split_smpl.top_betas.data
        smpl.betas.data[:, 2:] = split_smpl.other_betas.data

        smpl.trans.data = split_smpl.trans.data
