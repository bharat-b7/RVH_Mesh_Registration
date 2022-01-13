"""
wrapper for smplh model
Created by Xianghui, 12 January 2022.
"""
import os
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn

from os.path import join
from .smplpytorch.smplpytorch.native.webuser.serialization import ready_arguments
from .smplpytorch.smplpytorch.pytorch.tensutils import (
  th_posemap_axisang,
  th_with_zeros,
  th_pack, make_list, subtract_flat_id
)
from .joint_regressor import JointRegressor
from .const import *
from ..torch_functions import batch_sparse_dense_matmul


class SMPLHPyTorchWrapper(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']
    def __init__(self,
                 model_root="/BS/xxie2020/static00/mysmpl/smplh",
                 gender='neutral',
                 num_betas=10):
        """
        Args:
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male'
        """
        super(SMPLHPyTorchWrapper, self).__init__()

        self.gender = gender
        if gender == 'neutral':
            self.model_path = join(model_root, 'SMPLH_neutral.pkl')
        elif gender == 'female':
            self.model_path = join(model_root, 'SMPLH_female.pkl')
        elif gender == 'male':
            self.model_path = join(model_root, 'SMPLH_male.pkl')

        smpl_data = ready_arguments(self.model_path)
        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'][:, :, :num_betas].r))  # maximum 10 betas?
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents
        self.num_joints = len(parents)

    def forward(self,
                th_pose_axisang,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1, 3),
                th_offsets=None, scale=1.):
        """
        Args:
        th_pose_axisang (Tensor (batch_size x 152)): pose parameters in axis-angle representation
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        th_trans (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        th_offsets (Tensor (batch_size x 6890 x 3)): if provided, adds per-vertex offsets in t-pose
        """
        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat, hands=True)

        # Below does: v_shaped = v_template + shapedirs * betas
        # If shape parameters are not provided
        # if th_betas is None or bool(torch.norm(th_betas) == 0):
        # th_v_shaped = self.th_v_template + torch.matmul(
        #     self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
        # th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
        #     batch_size, 1, 1)
        # else:
        th_v_shaped = self.th_v_template + torch.matmul(self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
        th_j = torch.matmul(self.th_J_regressor, th_v_shaped)  # (B, 6890, 3)

        # Below does: v_posed = v_shaped + posedirs * pose_map
        naked = th_v_shaped + torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        if th_offsets is not None:
            th_v_posed = naked + th_offsets
        else:
            th_v_posed = naked
        # Final T pose with transformation done!

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)  # (B, 52, 3) to (B, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))  # convert to 4x4 matrix

        # Rotate each part
        for i in range(self.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                                          9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.kintree_parents)[i_val]  # get parent joint index
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)  # parent joint location?
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))  # (B, 4, 4)
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, self.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)
        for i in range(self.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2,
                            self.th_weights.transpose(0, 1))  # pose blend shape, convert joints pose to verts offsets?

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]

        # Scale
        th_verts = (th_verts) * scale
        th_jtr = (th_jtr) * scale

        # If translation is not provided
        th_jtr = th_jtr + th_trans.unsqueeze(1)
        th_verts = th_verts + th_trans.unsqueeze(1)

        # Vertices and joints in meters
        return th_verts, th_jtr, th_v_posed, naked


class SMPLHPyTorchWrapperBatch(Module):
    def __init__(self, model_root, batch_sz, betas=None,
                 pose=None, trans=None,
                 offsets=None, faces=None,
                 gender='male', num_betas=10):
        super(SMPLHPyTorchWrapperBatch, self).__init__()
        self.model_root = model_root

        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, num_betas))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(torch.tensor(betas))
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, SMPLH_POSE_PRAMS_NUM))
        else:
            assert pose.ndim == 2
            self.pose = nn.Parameter(torch.tensor(pose))
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(torch.tensor(trans))
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(torch.tensor(offsets))

        self.faces = faces
        self.gender = gender
        self.smpl = SMPLHPyTorchWrapper(self.model_root, gender=gender, num_betas=num_betas)

        # Landmarks, same as SMPL
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = JointRegressor.load_regressors(self.model_root,
                                                                                                    batch_size=batch_sz)

    def forward(self):
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smpl(self.pose,
                                  th_betas=self.betas,
                                  th_trans=self.trans,
                                  th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class SMPLHPyTorchWrapperBatchSplitParams(Module):
    def __init__(self, model_root, batch_sz,
                 top_betas=None,
                 other_betas=None,
                 global_pose=None,
                 body_pose=None,
                 hand_pose=None,
                 trans=None,
                 offsets=None,
                 faces=None, gender='male', num_betas=10):
        super(SMPLHPyTorchWrapperBatchSplitParams, self).__init__()
        self.model_root = model_root
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, TOP_BETA_NUM))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, num_betas-TOP_BETA_NUM))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, GLOBAL_POSE_NUM))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(batch_sz, BODY_POSE_NUM))
        else:
            assert body_pose.ndim == 2
            self.body_pose = nn.Parameter(body_pose)
        if hand_pose is None:
            self.hand_pose = nn.Parameter(torch.zeros(batch_sz, HAND_POSE_NUM))
        else:
            assert hand_pose.ndim == 2
            self.hand_pose = nn.Parameter(hand_pose)

        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)

        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.body_pose, self.hand_pose], axis=1)

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPLHPyTorchWrapper(self.model_root, gender=gender, num_betas=num_betas)
        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = JointRegressor.load_regressors(self.model_root,
                                                                                                    batch_size=batch_sz)

    def forward(self):
        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.body_pose, self.hand_pose], axis=1)

        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)

        # everytime forward is called, these two parameters are updated
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""
        verts, _, _, _ = self.smpl(self.pose,
                                  th_betas=self.betas,
                                  th_trans=self.trans,
                                  th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)
        return J, face, hands

    @staticmethod
    def from_smplh(smpl:SMPLHPyTorchWrapperBatch):
        """
        construct a smplh with split parameters from given smplh batch model
        Args:
            smpl:

        Returns: SMPLH batch model with split parameters

        """
        batch_sz = smpl.pose.shape[0]
        split_smpl = SMPLHPyTorchWrapperBatchSplitParams(smpl.model_root,
                                                         batch_sz,
                                                         trans=smpl.trans.data,
                                                         top_betas=smpl.betas.data[:, :TOP_BETA_NUM],
                                                         other_betas=smpl.betas.data[:, TOP_BETA_NUM:],
                                                          global_pose=smpl.pose.data[:, :GLOBAL_POSE_NUM],
                                                          body_pose=smpl.pose.data[:, GLOBAL_POSE_NUM:GLOBAL_POSE_NUM+BODY_POSE_NUM],
                                                          hand_pose=smpl.pose.data[:, GLOBAL_POSE_NUM+BODY_POSE_NUM:],
                                                          faces=smpl.faces, gender=smpl.gender)
        return split_smpl