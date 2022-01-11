"""
hand prior for SMPL-H
"""
import numpy as np
import torch
import pickle as pkl
from os.path import join


def grab_prior(lhand_path = '/BS/xxie2020/static00/assets/grab/lh_prior.pkl',
    rhand_path = '/BS/xxie2020/static00/assets/grab/rh_prior.pkl'):
    lhand_data = pkl.load(open(lhand_path, 'rb'))
    rhand_data = pkl.load(open(rhand_path, 'rb'))

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec

def amass_prior(lhand_path = '/BS/xxie2020/static00/assets/BMLmovi/BMLmovi_blhand_prior.pkl',
    rhand_path = '/BS/xxie2020/static00/assets/BMLmovi/BMLmovi_brhand_prior.pkl'):
    lhand_data = pkl.load(open(lhand_path, 'rb'))
    rhand_data = pkl.load(open(rhand_path, 'rb'))

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec


def mean_hand_pose(model_root):
    # lhand_file = "/BS/XZ_project2/work/Human-Chair-Interaction/pose/datasets/lh_mean.npy",
    # rhand_file = "/BS/XZ_project2/work/Human-Chair-Interaction/pose/datasets/rh_mean.npy"
    lhand_file = join(model_root, "lh_mean.npy")
    rhand_file = join(model_root, "rh_mean.npy")
    lhand_mean = np.array(np.load(lhand_file))
    rhand_mean = np.array(np.load(rhand_file))
    mean_pose = np.concatenate([lhand_mean, rhand_mean])
    return mean_pose


class HandPrior:
    HAND_POSE_NUM=45

    def __init__(self, prefix=66, device='cuda:0', dtype=torch.float,
                 type='grab'):
        "prefix is the index from where hand pose starts, 66 for SMPL-H"
        self.prefix = prefix
        if type == 'mean':
            hand_pose = mean_hand_pose()
            self.mean = torch.tensor(hand_pose, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = None
            self.rhand_prec = None
        else:
            prior, lhand_prec, rhand_prec = grab_prior()
            # prior, lhand_prec, rhand_prec = amass_prior()  # use AMASS prior
            self.mean = torch.tensor(prior, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = torch.tensor(lhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            self.rhand_prec = torch.tensor(rhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)


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
