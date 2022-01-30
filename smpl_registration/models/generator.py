"""
Generator to produce meshes (using marching cubes) after training IP-Net.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""
from lib.libmesh.implicit_waterproofing import create_grid_points_from_bounds
import mcubes
from psbody.mesh import Mesh, MeshViewer
import torch
import os
from glob import glob
import numpy as np

MIN, MAX = -1., 1.


class Generator(object):
    def __init__(self, model, threshold, exp_name, checkpoint=None, device=torch.device("cuda"), resolution=16,
                 batch_points=1000000):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(exp_name)
        if exp_name is not None:
            self.load_checkpoint(checkpoint)
        else:
            print('Not loading weights, it is assumed that the provided network has correct weights.')
        self.batch_points = batch_points

    @staticmethod
    def generate_grid(min, max, res):
        """Create a uniform grid for testing"""
        grid_points = create_grid_points_from_bounds(MIN, MAX, res)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = max + min
        b = max - min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b
        return grid_coords

    def generate_grid_torch(self):
        grid_coords = self.generate_grid(MIN, MAX, self.resolution)
        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_coords), 3))
        grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
        return grid_points_split

    def generate_mesh(self, logits):
        """Convert voxelized continous occupancy to mesh"""
        # logits = torch.cat(logits, dim=0)
        logits = np.reshape(logits.numpy(), (self.resolution,) * 3)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)

        mesh = Mesh(v=vertices, f=triangles)
        return mesh

    def normalize(self, vertices, max, min):
        step = (max - min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [min, min, min]
        return vertices

    def generate_mesh_all(self, data):
        """Predict occupancy and create a mesh"""
        grid_points_split = self.generate_grid_torch()
        inputs = data['inputs'].to(self.device)

        logits_list = {'out': []}
        for points in grid_points_split:
            points.to(self.device)
            with torch.no_grad():
                logits = self.model(points, inputs)
            logits_list['out'].append(logits['out'].squeeze(0).detach().cpu())

        # import ipdb; ipdb.set_trace()

        logits_list['out'] = torch.cat(logits_list['out'], dim=1)
        full = self.generate_mesh(logits_list['out'])
        full.v = self.normalize(full.v, MAX, MIN)
        return full

    def get_last_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        return checkpoints[-1]

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoint = self.get_last_checkpoint()
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint_ = torch.load(path)
        self.model.load_state_dict(checkpoint_['model_state_dict'])


class GeneratorIPNet(Generator):

    def generate_parts(self, logits_part, vertices):
        """Predict part labels"""
        logits = np.reshape(logits_part.numpy(), (-1, self.resolution, self.resolution, self.resolution))
        part_grid = np.argmax(logits, axis=0)

        # pick nearest joint label to a vertex
        r_verts = np.round(vertices).astype('int32')
        parts = part_grid[r_verts[:, 0], r_verts[:, 1], r_verts[:, 2]]
        # import ipdb; ipdb.set_trace()
        return parts

    @staticmethod
    def replace_infs(x):
        x[x == float("Inf")] = x[x != float("Inf")].max()
        x[x == float("-Inf")] = x[x != float("-Inf")].min()
        return x

    def generate_meshs_all_parts(self, data):
        """
        Marching cubes requies binary logits whereas IPNet produces a multi-label output.
        We need to convert categorical labels to binary.
        """
        grid_points_split = self.generate_grid_torch()
        inputs = data['inputs'].to(self.device)

        logits_list = {'out': [], 'parts': []}
        for points in grid_points_split:
            points.to(self.device)
            with torch.no_grad():
                logits = self.model(points, inputs)
            logits_list['out'].append(logits['out'].squeeze(0).detach().cpu())
            logits_list['parts'].append(logits['parts'].squeeze(0).detach().cpu())

        logits_list['parts'] = torch.cat(logits_list['parts'], dim=1)
        logits = torch.cat(logits_list['out'], dim=1)

        softmax_logits = torch.softmax(logits, dim=0)
        full_logit = -1. * torch.log(1. / (softmax_logits[1, :] + softmax_logits[2, :]) - 1)
        body_logit = -1. * torch.log(1. / softmax_logits[2, :] - 1)

        # replace infs
        full_logit = self.replace_infs(full_logit)
        body_logit = self.replace_infs(body_logit)

        full = self.generate_mesh(full_logit)
        body = self.generate_mesh(body_logit)
        parts = self.generate_parts(logits_list['parts'], body.v.copy())
        full.v = self.normalize(full.v, MAX, MIN)
        body.v = self.normalize(body.v, MAX, MIN)

        return full, body, parts


class GeneratorIPNetMano(GeneratorIPNet):
    def generate_meshs_all_parts(self, data):
        grid_points_split = self.generate_grid_torch()
        inputs = data['inputs'].to(self.device)

        logits_list = {'out': [], 'parts': []}
        for points in grid_points_split:
            points.to(self.device)
            # import ipdb; ipdb.set_trace()
            with torch.no_grad():
                logits = self.model(points, inputs)
            logits_list['out'].append(logits['out'].squeeze(0).detach().cpu())
            logits_list['parts'].append(logits['parts'].squeeze(0).detach().cpu())

        logits_list['out'] = torch.cat(logits_list['out'], dim=1)
        logits_list['parts'] = torch.cat(logits_list['parts'], dim=1)
        full = self.generate_mesh(logits_list['out'])
        parts = self.generate_parts(logits_list['parts'], full.v.copy())
        full.v = self.normalize(full.v, MAX, MIN)
        return full, parts

