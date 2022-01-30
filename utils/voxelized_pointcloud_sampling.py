"""
Code taken from: https://github.com/jchibane/if-net/blob/master/data_processing/voxelized_pointcloud_sampling.py
Cite: Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion, CVPR 2020.
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import lib.libmesh.implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
import os
from os.path import join, split, exists
import argparse


def voxelize(pc, res, bounds=(-1., 1.), save_path=None):
    grid_points = iw.create_grid_points_from_bounds(bounds[0], bounds[1], res)
    occupancies = np.zeros(len(grid_points), dtype=np.int8)
    kdtree = KDTree(grid_points)
    _, idx = kdtree.query(pc)
    occupancies[idx] = 1

    if save_path is not None:
        compressed_occupancies = np.packbits(occupancies)
        if not exists(save_path):
            os.makedirs(save_path)
        np.savez(save_path, point_cloud=pc, compressed_occupancies=compressed_occupancies, bb_min=bounds[0],
                 bb_max=bounds[1], res=res)

    return occupancies


def voxelized_pointcloud_sampling(mesh_path, name, out_path, res, num_points, bounds=(-1., 1.), ext=''):
    if not exists(mesh_path):
        print('Mesh not found, ', mesh_path)
        return

    out_file = join(out_path, name + '_voxelized_point_cloud_res_{}_points_{}_{}.npz'.format(res, num_points, ext))
    if exists(out_file) and REDO == False:
        print('Already exists, ', split(out_file)[1])
        return

    mesh = trimesh.load(mesh_path)
    point_cloud = mesh.sample(num_points)
    _ = voxelize(point_cloud, res, bounds, save_path=out_path)

    print('Done, ', split(out_file)[1])


REDO = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxelized sampling'
    )
    parser.add_argument(
        'mesh_path',
        type=str)
    parser.add_argument(
        'out_path',
        type=str)
    parser.add_argument(
        '--ext',
        type=str, default='')
    parser.add_argument(
        '--REDO', dest='REDO', action='store_true')
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--num_points', type=int, default=1024)
    args = parser.parse_args()

    bb_min = -1
    bb_max = 1.

    REDO = args.REDO
    name = split(args.mesh_path)[1][:-4]
    voxelized_pointcloud_sampling(args.mesh_path, name, args.out_path, args.res, args.num_points,
                                  bounds=(bb_min, bb_max), ext=args.ext)
