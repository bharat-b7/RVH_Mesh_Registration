'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
Author: Bharat
Edit: Xiaohan,
Edit: Xianghui, April 9, 2021
'''
import torch
import torch.nn as nn
# from smpl_layer import SMPL_Layer
import sys
from .smplh_layer import SMPLH_Layer
from .body_objectives import torch_pose_obj_data
from .torch_functions import batch_sparse_dense_matmul

SMPLH_POSE_PARAM_NUM=152


# not optimizable
class batch_SMPLH(nn.Module):

    def __init__(self, gender='male', hands=False):
        super(batch_SMPLH, self).__init__()

        self.gender = gender
        self.hands = hands
        ## pytorch smpl
        self.smpl = SMPLH_Layer(center_idx=0,
                               gender=self.gender,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models',
                               num_betas=10,
                               hands=self.hands)
        self.faces = self.smpl.th_faces


    def forward(self, betas, pose, trans, scale):
        betas = betas.repeat(pose.shape[0], 1)
        verts, jtr, tposed, naked = self.smpl(pose,
                                              th_betas=betas,
                                              th_trans=trans,
                                              scale=scale)
        return verts, jtr, tposed, naked

class th_batch_SMPLH_Split(nn.Module):
    """
    SMPL-H model, split beta, pose parameters
    pose: global pose, body pose, and hand pose
    beta: top beta, and others
    """
    GLOBAL_POSE_NUM = 3
    BODY_POSE_NUM = 63
    HAND_POSE_BUM = 90
    TOP_BETA_NUM = 2
    def __init__(self, batch_sz,
                 top_betas=None,
                 other_betas=None,
                 global_pose=None,
                 body_pose=None,
                 hand_pose = None,
                 trans=None,
                 offsets=None,
                 faces=None, gender='male', num_betas=300):
        super(th_batch_SMPLH_Split, self).__init__()
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, self.TOP_BETA_NUM))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, num_betas-self.TOP_BETA_NUM))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, self.GLOBAL_POSE_NUM))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(batch_sz, self.BODY_POSE_NUM))
        else:
            assert body_pose.ndim == 2
            self.body_pose = nn.Parameter(body_pose)
        if hand_pose is None:
            self.hand_pose = nn.Parameter(torch.zeros(batch_sz, self.HAND_POSE_BUM))
        else:
            assert hand_pose.ndim == 2
            self.hand_pose = nn.Parameter(hand_pose)

        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)

        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.body_pose, self.hand_pose], axis=1)

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPLH_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                                model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models',
                                hands=True)
        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

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


# added by Xianghui, April 9, 2021
class th_batch_SMPLH(nn.Module):
    "SMPL-H model, batched"
    def __init__(self, batch_sz, betas=None, pose=None,
                 trans=None, offsets=None, faces=None,
                 gender='male', num_betas=300):
        super(th_batch_SMPLH, self).__init__()

        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, num_betas))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(torch.tensor(betas))
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, SMPLH_POSE_PARAM_NUM))
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
        self.smpl = SMPLH_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models',
                                hands=True)

        # Landmarks, same as SMPL
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)


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


class th_SMPLH(nn.Module):
    def __init__(self, betas=None, pose=None, trans=None, offsets=None, tailor=False):
        super(th_SMPLH, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:
            if tailor:
                self.offsets = torch.zeros(6890, 3).cuda()
            else:
                self.offsets = nn.Parameter(torch.zeros(6890, 3))
        else:
            if tailor:
                self.offsets = offsets.cuda() #todo:hack for tailornt, should be tensor
            else:
                self.offsets = nn.Parameter(offsets)
        # self.update_betas = nn.Parameter(torch.zeros(10,))
        # self.update_pose = nn.Parameter(torch.zeros(72,))
        # self.update_trans = nn.Parameter(torch.zeros(3,))

        ## pytorch smpl
        self.smpl = SMPLH_Layer(center_idx=0, gender=self.gender,
                          model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/common')

    def forward(self):
        verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]
