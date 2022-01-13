"""
wrapper for the body key point regressor from smpl/smplh model

Author: Xianghui
"""
import pickle as pkl
from os.path import join
import torch


class JointRegressor(object):
    def __init__(self, model_root):
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = self.load_regressors(model_root)

    @staticmethod
    def load_regressors(model_root, batch_size=1):
        body25_reg = pkl.load(open(join(model_root, 'regressors/body_25_openpose_joints.pkl'), 'rb'),
                              encoding="latin1").T
        face_reg = pkl.load(open(join(model_root, 'regressors/face_70_openpose_joints.pkl'), 'rb'), encoding="latin1").T
        hand_reg = pkl.load(open(join(model_root, 'regressors/hands_42_openpose_joints.pkl'), 'rb'),
                            encoding="latin1").T
        body25_reg_torch = torch.sparse_coo_tensor(body25_reg.nonzero(), body25_reg.data, body25_reg.shape)
        face_reg_torch = torch.sparse_coo_tensor(face_reg.nonzero(), face_reg.data, face_reg.shape)
        hand_reg_torch = torch.sparse_coo_tensor(hand_reg.nonzero(), hand_reg.data, hand_reg.shape)

        return torch.stack([body25_reg_torch] * batch_size), torch.stack([face_reg_torch] * batch_size), \
               torch.stack([hand_reg_torch] * batch_size)
