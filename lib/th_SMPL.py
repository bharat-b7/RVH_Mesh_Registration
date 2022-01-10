'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
If code works:
    Author: Bharat
else:
    Author: Anonymous
'''
import torch
import torch.nn as nn
from lib.smpl_layer import SMPL_Layer
from lib.body_objectives import torch_pose_obj_data
from lib.torch_functions import batch_sparse_dense_matmul


class th_batch_SMPL_split_params(nn.Module):
    """
    Alternate implementation of th_batch_SMPL that allows us to independently optimise:
     1. global_pose
     2. remaining other_pose
     3. top betas (primarly adjusts bone lengths)
     4. other betas
    """
    def __init__(self, batch_sz, top_betas=None, other_betas=None, global_pose=None, other_pose=None, trans=None,
                 offsets=None, faces=None, gender='male', num_betas=300):
        super(th_batch_SMPL_split_params, self).__init__()
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, 2))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, num_betas-2))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if other_pose is None:
            self.other_pose = nn.Parameter(torch.zeros(batch_sz, 69))
        else:
            assert other_pose.ndim == 2
            self.other_pose = nn.Parameter(other_pose)

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
        self.pose = torch.cat([self.global_pose, self.other_pose], axis=1)

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root='lib/smplpytorch/smplpytorch/native/models')

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

    def forward(self):
        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.other_pose], axis=1)
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


class th_batch_SMPL(nn.Module):
    def __init__(self, batch_sz, betas=None, pose=None, trans=None, offsets=None, faces=None, gender='male'):
        super(th_batch_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, 72))
        else:
            assert pose.ndim == 2
            self.pose = nn.Parameter(pose)
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

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                               model_root='lib/smplpytorch/smplpytorch/native/models')

        # Landmarks
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


class th_SMPL(nn.Module):
    def __init__(self, betas=None, pose=None, trans=None, offsets=None, gender='male'):
        super(th_SMPL, self).__init__()
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
            self.offsets = nn.Parameter(torch.zeros(6890,3))
        else:
            self.offsets = nn.Parameter(offsets)

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                          model_root='lib/smplpytorch/smplpytorch/native/models')

    def forward(self):
        verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]
