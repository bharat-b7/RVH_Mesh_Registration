"""
Code to test IPNet with a pointcloud as input. To run IPNet on entire dataset see generate.py
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""
import sys, os
sys.path.append(os.getcwd())
import torch
import models.ipnet_models as model
import numpy as np
import argparse
from utils.preprocess_scan import func
from utils.voxelized_pointcloud_sampling import voxelize
from models.generator import GeneratorIPNet, GeneratorIPNetMano, Generator
import trimesh
import os
from os.path import join, split, exists


def pc2vox(pc, res):
    """Convert PC to voxels for IPNet"""
    # datagen the pointcloud
    pc, scale, cent = func(pc)
    vox = voxelize(pc, res)
    return vox, scale, cent


def main(args):
    # Load PC
    pc = trimesh.load(args.pc)
    pc_vox, scale, cent = pc2vox(pc.vertices, args.res)
    pc_vox = np.reshape(pc_vox, (args.res,) * 3).astype('float32')

    # save scale file
    from utils.preprocess_scan import SCALE, new_cent
    np.save(join(args.out_path, 'cent.npy'), [scale / SCALE, (cent - new_cent)])

    # Load network
    if args.model == 'IPNet':
        net = model.IPNet(hidden_dim=args.decoder_hidden_dim, num_parts=14)
        gen = GeneratorIPNet(net, 0.5, exp_name=None, resolution=args.retrieval_res,
                             batch_points=args.batch_points)
    elif args.model == 'IPNetMano':
        net = model.IPNetMano(hidden_dim=args.decoder_hidden_dim, num_parts=7)
        gen = GeneratorIPNetMano(net, 0.5, exp_name=None, resolution=args.retrieval_res,
                                 batch_points=args.batch_points)
    elif args.model == 'IPNetSingleSurface':
        net = model.IPNetSingleSurface(hidden_dim=args.decoder_hidden_dim, num_parts=14)
        gen = Generator(net, 0.5, exp_name=None, resolution=args.retrieval_res,
                        batch_points=args.batch_points)
    else:
        print('Wow watch where u goin\' with that model')
        exit()

    # Load weights
    print('Loading weights from,', args.weights)
    checkpoint_ = torch.load(args.weights)
    net.load_state_dict(checkpoint_['model_state_dict'])

    # Run IPNet and Save
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    data = {'inputs': torch.tensor(pc_vox[np.newaxis])}    # add a batch dimension
    if args.model == 'IPNet':
        full, body, parts = gen.generate_meshs_all_parts(data)
        body.set_vertex_colors_from_weights(parts)
        body.write_ply(args.out_path + '/body.ply')
        np.save(args.out_path + '/parts.npy', parts)

    elif args.model == 'IPNetMano':
        full, parts = gen.generate_meshs_all_parts(data)
        np.save(args.out_path + '/parts.npy', parts)

    elif args.model == 'IPNetSingleSurface':
        full = gen.generate_mesh_all(data)

    full.write_ply(args.out_path + '/full.ply')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Model')
    # Path to PC mesh
    # parser.add_argument('pc', type=str)
    # # path to pretrained weights
    # parser.add_argument('weights', type=str)
    # # path to save result
    # parser.add_argument('out_path', type=str)
    # # the resolution of the input
    # parser.add_argument('-res', default=128, type=int)
    # # keep this fixed
    # parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
    # # number of points queried for to produce the result
    # parser.add_argument('-retrieval_res', default=256, type=int)
    # # number of points from the querey grid which are put into the batch at once
    # # parser.add_argument('-batch_points', default=300000, type=int)
    # parser.add_argument('-batch_points', default=10000, type=int)
    # # which model to use, e.g. "-m IPNet"
    # parser.add_argument('-m', '--model', default='IPNetSingleSurface', type=str)
    args = parser.parse_args()

    # args = lambda: None
    args.pc = 'assets/scan.obj'
    args.weights = '/BS/xxie2020/work/hoi3d/external/IPNet/experiments/IPNet_p5000_01_exp_id01/checkpoints/checkpoint_epoch_249.tar'
    args.out_path = 'test_data'
    args.res = 128
    args.decoder_hidden_dim = 256
    args.retrieval_res = 256
    args.batch_points = 10000
    args.model = 'IPNet'

    main(args)


"""
python test_IPNet.py <pc_path> <network_weights_path> <out_path> -m IPNetSingleSurface
"""