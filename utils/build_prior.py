"""
build hand prior from grab dataset, and smplh prior from AMASS dataset

Author: Xianghui Xie, April 12, 2021
"""
import pickle
import numpy as np
import glob
from os.path import join
import os
from tqdm import tqdm


def create_prior_from_samples(samples):
    from sklearn.covariance import GraphicalLassoCV
    from numpy import asarray, linalg
    model = GraphicalLassoCV(max_iter=500)
    model.fit(asarray(samples))
    return asarray(samples).mean(axis=0), linalg.cholesky(model.precision_)


def build_bmlmovi_prior(data_path, out_path, dataset="BMLmoviall"):
    """

    Args:
        data_path: path to amass registration, e.g.  .../amass/BMLmovi
        out_path:
        dataset: name of the dataset used, for output filename

    Returns:

    """
    os.makedirs(out_path, exist_ok=True)
    dataset_path = data_path
    seqs = os.listdir(dataset_path)

    body_poses = []
    lhand_poses = []
    rhand_poses = []
    # for seq in tqdm(seqs):
    for seq in seqs:
        files = glob.glob(join(dataset_path, seq, "*_poses.npz"))
        for file in tqdm(files):
            pose_data = np.load(file)['poses']
            # selected = np.random.choice(pose_data.shape[0], int(pose_data.shape[0]/10))
            body_poses.append(pose_data[:, 3:66])
            lhand_poses.append(pose_data[:, 66:66 + 45])
            rhand_poses.append(pose_data[:, 66 + 45:])

    print("in total {} sequences loaded".format(len(body_poses)))
    body_poses = np.concatenate(body_poses, axis=0)
    lhand_poses = np.concatenate(lhand_poses, axis=0)
    rhand_poses = np.concatenate(rhand_poses, axis=0)
    print("in total {} poses".format(body_poses.shape[0]))

    pose_mean, pose_precision = create_prior_from_samples(body_poses)
    data = {"mean": pose_mean, "precision": pose_precision}
    outfile = join(out_path, f"{dataset}_body_prior.pkl")
    pickle.dump(data, open(outfile, 'wb'))
    print(outfile, 'saved')

    pose_mean, pose_precision = create_prior_from_samples(lhand_poses)
    data = {"mean": pose_mean, "precision": pose_precision}
    outfile = join(out_path, f"{dataset}_blhand_prior.pkl")
    pickle.dump(data, open(outfile, 'wb'))
    print(outfile, 'saved')

    pose_mean, pose_precision = create_prior_from_samples(rhand_poses)
    data = {"mean": pose_mean, "precision": pose_precision}
    outfile = join(out_path, f"{dataset}_brhand_prior.pkl")
    pickle.dump(data, open(outfile, 'wb'))
    print(outfile, 'saved')


def build_grab_prior(data_path, out_path):
    """
    build hand prior from given grab dataset
    Args:
        data_path: path to the grab dataset, e.g. ../grab_unzip/data
        out_path: path to where the priors are stored

    Returns:

    """
    os.makedirs(out_path, exist_ok=True)
    lh_pose = []
    rh_pose = []
    for path in tqdm(glob.glob(join(data_path, '*.npz'))):
        data = np.load(path, allow_pickle=True)
        lh_pose.append(data['lhand'].item()['params']['fullpose'])  # (F, 45), where F is the number of frames
        rh_pose.append(data['rhand'].item()['params']['fullpose'])
    lh_pose = np.concatenate(lh_pose, axis=0)
    rh_pose = np.concatenate(rh_pose, axis=0)
    lh_mean, lh_precision = create_prior_from_samples(lh_pose)
    data = {'mean':lh_mean, 'precision':lh_precision}
    pickle.dump(data, open(join(out_path, 'lh_prior.pkl'), 'wb'))
    rh_mean, rh_precision = create_prior_from_samples(rh_pose)
    data = {'mean': rh_mean, 'precision': rh_precision}
    pickle.dump(data, open(join(out_path, 'rh_prior.pkl'), 'wb'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('data_path', help='path to the dataset')
    parser.add_argument('out', help='output file path')

    args = parser.parse_args()

    build_grab_prior(args.data_path, args.out)
    # build_bmlmovi_prior(args.data_path, args.out)