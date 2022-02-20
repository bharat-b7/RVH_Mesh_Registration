"""
hand prior for SMPL-H
Author: Xianghui, 12 January 2022
"""
import numpy as np
import torch
import pickle as pkl
from os.path import join


def grab_prior(root_path):
    lhand_data, rhand_data = load_grab_prior(root_path)

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec


def load_grab_prior(root_path):
    lhand_path = join(root_path, 'priors', 'lh_prior.pkl')
    rhand_path = join(root_path, 'priors', 'rh_prior.pkl')
    lhand_data = pkl.load(open(lhand_path, 'rb'))
    rhand_data = pkl.load(open(rhand_path, 'rb'))
    return lhand_data, rhand_data


def mean_hand_pose(root_path):
    "mean hand pose computed from grab dataset"
    lhand_data, rhand_data = load_grab_prior(root_path)
    lhand_mean = np.array(lhand_data['mean'])
    rhand_mean = np.array(rhand_data['mean'])
    mean_pose = np.concatenate([lhand_mean, rhand_mean])
    return mean_pose


class HandPrior:
    HAND_POSE_NUM=45
    def __init__(self, prior_path,
                 prefix=66,
                 device='cuda:0',
                 dtype=torch.float,
                 type='grab'):
        "prefix is the index from where hand pose starts, 66 for SMPL-H"
        self.prefix = prefix
        if type == 'grab':
            prior, lhand_prec, rhand_prec = grab_prior(prior_path)
            self.mean = torch.tensor(prior, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = torch.tensor(lhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            self.rhand_prec = torch.tensor(rhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
        else:
            raise NotImplemented("Only grab hand prior is supported!")

    def __call__(self, full_pose):
        "full_pose also include body poses, this function can be used to compute loss"
        temp = full_pose[:, self.prefix:] - self.mean
        if self.lhand_prec is None:
            return (temp*temp).sum(dim=1)
        else:
            lhand = torch.matmul(temp[:, :self.HAND_POSE_NUM], self.lhand_prec)
            rhand = torch.matmul(temp[:, self.HAND_POSE_NUM:], self.rhand_prec)
            temp2 = torch.cat([lhand, rhand], axis=1)
            return (temp2 * temp2).sum(dim=1)
