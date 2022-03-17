"""
Code to fit SMPL (pose, shape) to IPNet predictions using pytorch, kaolin.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.

this code does:
1. load ipnet model, and generate meshes
2. fit smplh+d to ipnet predictions
"""
import torch
import trimesh
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from os.path import join, basename, splitext
from psbody.mesh import Mesh
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
import pickle as pkl
from tqdm import tqdm
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from models.generator import GeneratorIPNet, GeneratorIPNetMano, Generator
import models.ipnet_models as model
from fit_SMPLHD import SMPLDFitter
from utils.preprocess_scan import func
from utils.preprocess_scan import SCALE, new_cent
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from utils.voxelized_pointcloud_sampling import voxelize

NUM_PARTS = 14


def pc2vox(pc, res):
    """Convert PC to voxels for IPNet"""
    # datagen the pointcloud
    pc, scale, cent = func(pc)
    vox = voxelize(pc, res)
    return vox, scale, cent


class SMPLHIPNetFitter(SMPLDFitter):
    def __init__(self, args):
        """
        initialize model, data etc.
        Args:
            args: command line arguments
        """
        super(SMPLHIPNetFitter, self).__init__(args.model_root, debug=args.display, hands=args.hands)
        # Load network
        net = model.IPNet(hidden_dim=args.decoder_hidden_dim, num_parts=14)
        gen = GeneratorIPNet(net, 0.5, exp_name=None, resolution=args.retrieval_res,
                             batch_points=args.batch_points)
        # Load weights
        print('Loading weights from,', args.weights)
        checkpoint_ = torch.load(args.weights)
        net.load_state_dict(checkpoint_['model_state_dict'])

        self.net = net
        self.generator = gen
        self.res = args.res # voxel input resolution

        self.save_name_base = 'smplh' if self.hands else 'smpl'

    def fit(self, scans, pose_files, smpl_pkl, gender='male', save_path=None):
        assert len(scans) == 1, 'currently only support batch size 1!'
        batch_sz = len(scans)
        # Load PC
        pc = trimesh.load(scans[0])
        pc_vox, scale, cent = pc2vox(pc.vertices, self.res)
        pc_vox = np.reshape(pc_vox, (self.res,) * 3).astype('float32')

        # save scale file
        from utils.preprocess_scan import SCALE, new_cent
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        scan_name = splitext(basename(scans[0]))[0]
        np.save(join(save_path, f'{scan_name}_cent.npy'), [scale / SCALE, (cent - new_cent)])

        # Run IPNet and save intermediate results
        data = {'inputs': torch.tensor(pc_vox[np.newaxis])}  # add a batch dimension
        full, body, parts = self.generator.generate_meshs_all_parts(data)
        body.set_vertex_colors_from_weights(parts)
        body.write_ply(join(save_path, f'{scan_name}_body.ply'))
        np.save(join(save_path, f'{scan_name}_parts.npy'), parts)
        full.write_ply(join(save_path, f'{scan_name}_full.ply'))

        # for debug: load predictions from file
        # parts = np.load(join(save_path, f'{scan_name}_parts.npy'))
        # full, body = Mesh(), Mesh()
        # full.load_from_file(join(save_path, f'{scan_name}_full.ply'))
        # body.load_from_file(join(save_path, f'{scan_name}_body.ply'))

        # scale and move back to original center
        full.v = (full.v + cent - new_cent) * scale / SCALE
        body.v = (body.v + cent - new_cent) * scale / SCALE

        # now fit smplh to predictions
        # init smpl
        smpl = self.init_smpl(batch_sz, gender)
        smpl_part_labels = self.load_smpl_parts(batch_sz)
        th_scan_meshes = self.meshes2torch([body])

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 3, 2, 30, 30

        # Optimize pose only
        parts_th = torch.from_numpy(parts).to(self.device).unsqueeze(0)
        self.optimize_pose_only(th_scan_meshes, smpl, iterations, steps_per_iter, parts_th, smpl_part_labels)

        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, parts_th, smpl_part_labels)
        # save smpl outputs
        self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name=f'{self.save_name_base}-ipnet')

        # optimize offsets using outer surface prediction
        th_scan_meshes = self.meshes2torch([full])
        self.optimize_offsets(th_scan_meshes, smpl, 6, 10)

        # save smpld outputs
        return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name=f'{self.save_name_base}-ipnet')

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, scan_parts, smpl_parts):
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                     betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        iter_for_global = 1
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            # global COUNT
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step(th_scan_meshes, split_smpl, scan_parts, smpl_parts)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)
                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes, scan_parts=scan_parts)
        self.copy_smpl_params(smpl, split_smpl)
        print('** Optimised smplh pose **')
    
    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, scan_parts, smpl_parts):
        """
            Optimize SMPL.
            :param display: if not None, pass index of the scan in th_scan_meshes to visualize.
            """
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # global COUNT
        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step(th_scan_meshes, smpl, scan_parts, smpl_parts)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes, scan_parts=scan_parts)

        print('** Optimised smplh pose and shape **')
    
    def viz_fitting(self, smpl, th_scan_meshes, ind=0,
                    smpl_vc=np.array([0, 1, 0]), **kwargs):
        verts, _, _, _ = smpl()
        smpl_mesh = Mesh(v=verts[ind].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
        scan_mesh = Mesh(v=th_scan_meshes.verts_list()[ind].cpu().detach().numpy(),
                         f=th_scan_meshes.faces_list()[ind].cpu().numpy(), vc=smpl_vc)
        if 'scan_parts' in kwargs:
            scan_part_labels = kwargs.get('scan_parts')
            scan_mesh.set_vertex_colors_from_weights(scan_part_labels[ind].cpu().detach().numpy())
        self.mv.set_static_meshes([smpl_mesh, scan_mesh])

    def forward_step(self, th_scan_meshes, smpl, scan_part_labels, smpl_part_labels):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
            """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender, precomputed=True)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        loss = dict()
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose))  # add hand prior if smplh is used

        loss['part'] = []
        scan_verts = th_scan_meshes.verts_list()
        for n, (sc_v, sc_l) in enumerate(zip(scan_verts, scan_part_labels)):
            tot = 0
            for i in range(NUM_PARTS):  # we currently use 14 parts
                if i not in sc_l:
                    continue
                ind = torch.where(sc_l == i)[0]
                sc_part_points = sc_v[ind].unsqueeze(0)
                sm_part_points = verts[n][torch.where(smpl_part_labels[n] == i)[0]].unsqueeze(0)
                dist, _ = chamfer_distance(sc_part_points, sm_part_points)
                tot += dist
            loss['part'].append(tot / NUM_PARTS)
        loss['part'] = torch.stack(loss['part']).mean()
        # for k, v in loss.items():
        #     print(k, v.shape)
        return loss

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: 100 ** 2 * cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'part': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

    def meshes2torch(self, meshes, device='cuda:0'):
        """
        convert a list of psbody meshes to pytorch3d mesh
        """
        verts, faces = [], []
        for m in meshes:
            verts.append(torch.from_numpy(m.v).float())
            faces.append(torch.from_numpy(m.f.astype(int)))
        py3d_mesh = Meshes(verts, faces).to(device)
        return py3d_mesh

    @staticmethod
    def load_smpl_parts(batch_size, device='cuda:0'):
        part_labels = pkl.load(open('assets/smpl_parts_dense.pkl', 'rb'))
        labels = np.zeros((6890,), dtype='int32')
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n
        labels = torch.tensor(labels).unsqueeze(0).to(device)
        smpl_part_labels = torch.cat([labels] * batch_size, axis=0)
        return smpl_part_labels


def main(args):
    fitter = SMPLHIPNetFitter(args)
    fitter.fit([args.scan_path], None, None, args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-w', '--weights', help='ip-net model checkpoint path')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument('-gender', type=str, default='male') # can be female
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    parser.add_argument('-res', default=128, type=int)
    # keep this fixed
    parser.add_argument('-h_dim', '--decoder_hidden_dim', default=256, type=int)
    # number of points queried for to produce the result
    parser.add_argument('-retrieval_res', default=256, type=int)
    # number of points from the querey grid which are put into the batch at once
    parser.add_argument('-batch_points', default=10000, type=int)
    args = parser.parse_args()

    # args.scan_path = 'assets/scan.obj'
    # args.save_path = 'test_data'
    # args.res = 128
    # args.decoder_hidden_dim = 256
    # args.retrieval_res = 256
    # args.batch_points = 10000
    # args.display = True
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)