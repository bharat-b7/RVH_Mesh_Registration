"""
Code to fit SMPL (pose, shape) to IPNet predictions using pytorch, kaolin.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""
import os
from os.path import split, join, exists
import sys
import ipdb
import json
import torch
import numpy as np
import pickle as pkl
import kaolin as kal
from kaolin.rep import TriangleMesh as tm
# from kaolin.metrics.mesh import point_to_surface, laplacian_loss  # , chamfer_distance
# from kaolin.conversions import trianglemesh_to_sdf
# from kaolin.rep import SDF as sdf
from psbody.mesh import Mesh, MeshViewer, MeshViewers
from tqdm import tqdm

from fit_SMPL import save_meshes, backward_step
from fit_SMPLD import optimize_offsets
# from fit_SMPLD import forward_step as forward_step_offsets
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.th_SMPL import th_batch_SMPL, th_batch_SMPL_split_params
from lib.mesh_distance import chamfer_distance, batch_point_to_surface
# from lib.body_objectives import batch_get_pose_obj

NUM_PARTS = 14  # number of parts that the smpl is segmented into.
COUNT = 0   # frame number
DEVICE = 'cuda'


def get_loss_weights():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                   'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                   'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                   'lap': lambda cst, it: cst / (1 + it),
                   'part': lambda cst, it: 10. ** 2 * cst / (1 + it)
                   }
    return loss_weight


def forward_step(th_scan_meshes, smpl, scan_part_labels, smpl_part_labels):
    """
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """
    # Get pose prior
    prior = get_prior(smpl.gender, precomputed=True)

    # forward
    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=smpl.faces) for v in verts]

    scan_verts = [sm.vertices for sm in th_scan_meshes]
    smpl_verts = [sm.vertices for sm in th_smpl_meshes]

    # losses
    loss = dict()
    loss['s2m'] = batch_point_to_surface(scan_verts, th_smpl_meshes)
    loss['m2s'] = batch_point_to_surface(smpl_verts, th_scan_meshes)
    loss['betas'] = torch.mean(smpl.betas ** 2, axis=1)
    loss['pose_pr'] = prior(smpl.pose)

    loss['part'] = []
    for n, (sc_v, sc_l) in enumerate(zip(scan_verts, scan_part_labels)):
        tot = 0
        for i in range(NUM_PARTS):  # we currently use 14 parts
            if i not in sc_l:
                continue
            ind = torch.where(sc_l == i)[0]
            sc_part_points = sc_v[ind].unsqueeze(0)
            sm_part_points = smpl_verts[n][torch.where(smpl_part_labels[n] == i)[0]].unsqueeze(0)
            dist = chamfer_distance(sc_part_points, sm_part_points, w1=1., w2=1.)
            tot += dist
        loss['part'].append(tot / NUM_PARTS)
    loss['part'] = torch.stack(loss['part'])
    return loss


def optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                        display=None):
    """
    Optimize SMPL.
    :param display: if not None, pass index of the scan in th_scan_meshes to visualize.
    """
    # Optimizer
    optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    # Display
    if display is not None:
        assert int(display) < len(th_scan_meshes)
        mv = MeshViewer()

    # global COUNT
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, smpl, scan_part_labels, smpl_part_labels)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

            if display is not None:
                verts, _, _, _ = smpl()
                smpl_mesh = Mesh(v=verts[display].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
                scan_mesh = Mesh(v=th_scan_meshes[display].vertices.cpu().detach().numpy(),
                                 f=th_scan_meshes[display].faces.cpu().numpy(), vc=np.array([0, 1, 0]))
                scan_mesh.set_vertex_colors_from_weights(scan_part_labels[display].cpu().detach().numpy())
                mv.set_static_meshes([scan_mesh, smpl_mesh])

                # col = np.load('DO_NOT_RELEASE/vitruvian_cols.npy')
                # smpl_mesh.set_vertex_colors(col)
                # mv.set_dynamic_meshes([smpl_mesh])
                # mv.save_snapshot('/BS/bharat-2/work/temp/show_optimization/snap_{}.png'.format(str(COUNT).zfill(5)))
                # COUNT += 1

    print('** Optimised smpl pose and shape **')


def optimize_pose_only(th_scan_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                       display=None):
    """
    Initially we want to only optimize the global rotation of SMPL. Next we optimize full pose.
    We optimize pose based on the 3D keypoints in th_pose_3d.
    :param  th_pose_3d: array containing the 3D keypoints.
    """

    batch_sz = smpl.pose.shape[0]
    split_smpl = th_batch_SMPL_split_params(batch_sz, top_betas=smpl.betas.data[:, :2],
                                            other_betas=smpl.betas.data[:, 2:],
                                            global_pose=smpl.pose.data[:, :3], other_pose=smpl.pose.data[:, 3:],
                                            faces=smpl.faces, gender=smpl.gender).to(DEVICE)
    optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                 betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    if display is not None:
        assert int(display) < len(th_scan_meshes)
        # mvs = MeshViewers((1,1))
        mv = MeshViewer(keepalive=True)

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
                                          split_smpl.other_pose], 0.02, betas=(0.9, 0.999))
        else:
            loop.set_description('Optimizing SMPL pose only')

        # global COUNT
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, split_smpl, scan_part_labels, smpl_part_labels)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

            if display is not None:
                verts, _, _, _ = split_smpl()
                smpl_mesh = Mesh(v=verts[display].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
                scan_mesh = Mesh(v=th_scan_meshes[display].vertices.cpu().detach().numpy(),
                                 f=th_scan_meshes[display].faces.cpu().numpy(), vc=np.array([0, 1, 0]))
                scan_mesh.set_vertex_colors_from_weights(scan_part_labels[display].cpu().detach().numpy())
                mv.set_static_meshes([smpl_mesh, scan_mesh])

                # col = np.load('DO_NOT_RELEASE/vitruvian_cols.npy')
                # smpl_mesh.set_vertex_colors(col)
                # mv.set_dynamic_meshes([smpl_mesh])
                # mv.save_snapshot('/BS/bharat-2/work/temp/show_optimization/snap_{}.png'.format(str(COUNT).zfill(5)))
                # COUNT+=1


    # Put back pose, shape and trans into original smpl
    smpl.pose.data = split_smpl.pose.data
    smpl.betas.data = split_smpl.betas.data
    smpl.trans.data = split_smpl.trans.data

    print('** Optimised smpl pose **')


def fit_SMPL(scans, scan_labels, gender='male', save_path=None, scale_file=None, display=None):
    """
    :param save_path:
    :param scans: list of scan paths
    :param pose_files:
    :return:
    """
    # Get SMPL faces
    sp = SmplPaths(gender=gender)
    smpl_faces = sp.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(DEVICE)

    # Load SMPL parts
    part_labels = pkl.load(open('assets/smpl_parts_dense.pkl', 'rb'))
    labels = np.zeros((6890,), dtype='int32')
    for n, k in enumerate(part_labels):
        labels[part_labels[k]] = n
    labels = torch.tensor(labels).unsqueeze(0).to(DEVICE)

    # Load scan parts
    scan_part_labels = []
    for sc_l in scan_labels:
        temp = torch.tensor(np.load(sc_l).astype('int32')).to(DEVICE)
        scan_part_labels.append(temp)

    # Batch size
    batch_sz = len(scans)

    # Set optimization hyper parameters
    iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 3, 2, 30, 30

    prior = get_prior(gender=gender, precomputed=True)
    pose_init = torch.zeros((batch_sz, 72))
    pose_init[:, 3:] = prior.mean
    betas, pose, trans = torch.zeros((batch_sz, 300)), pose_init, torch.zeros((batch_sz, 3))

    # Init SMPL, pose with mean smpl pose, as in ch.registration
    smpl = th_batch_SMPL(batch_sz, betas, pose, trans, faces=th_faces).to(DEVICE)
    smpl_part_labels = torch.cat([labels] * batch_sz, axis=0)

    th_scan_meshes, centers = [], []
    for scan in scans:
        print('scan path ...', scan)
        temp = Mesh(filename=scan)
        th_scan = tm.from_tensors(torch.tensor(temp.v.astype('float32'), requires_grad=False, device=DEVICE),
                                  torch.tensor(temp.f.astype('int32'), requires_grad=False, device=DEVICE).long())
        th_scan_meshes.append(th_scan)

    if scale_file is not None:
        for n, sc in enumerate(scale_file):
            dat = np.load(sc, allow_pickle=True)
            th_scan_meshes[n].vertices += torch.tensor(dat[1]).to(DEVICE)
            th_scan_meshes[n].vertices *= torch.tensor(dat[0]).to(DEVICE)

    # Optimize pose first
    optimize_pose_only(th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, scan_part_labels, smpl_part_labels,
                       display=None if display is None else 0)

    # Optimize pose and shape
    optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, scan_part_labels, smpl_part_labels,
                        display=None if display is None else 0)

    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v, faces=smpl.faces) for v in verts]

    if save_path is not None:
        if not exists(save_path):
            os.makedirs(save_path)

        names = [split(s)[1] for s in scans]

        # Save meshes
        save_meshes(th_smpl_meshes, [join(save_path, n.replace('.ply', '_smpl.obj')) for n in names])
        save_meshes(th_scan_meshes, [join(save_path, n.replace('.ply', '.obj')) for n in names])

        # Save params
        for p, b, t, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                              smpl.trans.cpu().detach().numpy(), names):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t}
            pkl.dump(smpl_dict, open(join(save_path, n.replace('.ply', '_smpl.pkl')), 'wb'))

        return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy()


def fit_SMPLD(scans, smpl_pkl, gender='male', save_path=None, scale_file=None):
    # Get SMPL faces
    sp = SmplPaths(gender=gender)
    smpl_faces = sp.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()

    # Batch size
    batch_sz = len(scans)

    # Init SMPL
    pose, betas, trans = [], [], []
    for spkl in smpl_pkl:
        smpl_dict = pkl.load(open(spkl, 'rb'))
        p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
        pose.append(p)
        if len(b) == 10:
            temp = np.zeros((300,))
            temp[:10] = b
            b = temp.astype('float32')
        betas.append(b)
        trans.append(t)
    pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)

    betas, pose, trans = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans)
    smpl = th_batch_SMPL(batch_sz, betas, pose, trans, faces=th_faces).cuda()

    verts, _, _, _ = smpl()
    init_smpl_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                        faces=smpl.faces) for v in verts]

    # Load scans
    th_scan_meshes = []
    for scan in scans:
        print('scan path ...', scan)
        temp = Mesh(filename=scan)
        th_scan = tm.from_tensors(torch.tensor(temp.v.astype('float32'), requires_grad=False, device=DEVICE),
                                  torch.tensor(temp.f.astype('int32'), requires_grad=False, device=DEVICE).long())
        th_scan_meshes.append(th_scan)

    if scale_file is not None:
        for n, sc in enumerate(scale_file):
            dat = np.load(sc, allow_pickle=True)
            th_scan_meshes[n].vertices += torch.tensor(dat[1]).to(DEVICE)
            th_scan_meshes[n].vertices *= torch.tensor(dat[0]).to(DEVICE)

    # Optimize
    optimize_offsets(th_scan_meshes, smpl, init_smpl_meshes, 5, 10)
    print('Done')

    verts, _, _, _ = smpl()
    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=smpl.faces) for v in verts]

    if save_path is not None:
        if not exists(save_path):
            os.makedirs(save_path)

        names = [split(s)[1] for s in scans]

        # Save meshes
        save_meshes(th_smpl_meshes, [join(save_path, n.replace('.ply', '_smpld.obj')) for n in names])
        save_meshes(th_scan_meshes, [join(save_path, n.replace('.ply', '.obj')) for n in names])
        # Save params
        for p, b, t, d, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                                 smpl.trans.cpu().detach().numpy(), smpl.offsets.cpu().detach().numpy(), names):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'offsets': d}
            pkl.dump(smpl_dict, open(join(save_path, n.replace('.ply', '_smpld.pkl')), 'wb'))

    return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), \
           smpl.trans.cpu().detach().numpy(), smpl.offsets.cpu().detach().numpy()


if __name__ == "__main__":
    import argparse
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # parser = argparse.ArgumentParser(description='Run Model')
    # parser.add_argument('inner_path', type=str)  # predicted by IPNet
    # parser.add_argument('outer_path', type=str)  # predicted by IPNet
    # parser.add_argument('inner_labels', type=str)  # predicted by IPNet
    # parser.add_argument('scale_file', type=str, default=None)  # obtained from utils/process_scan.py
    # parser.add_argument('save_path', type=str)
    # parser.add_argument('-gender', type=str, default='male')  # can be female/ male/ neutral
    # parser.add_argument('--display', default=None)
    # args = parser.parse_args()

    # args = lambda: None
    # args.inner_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir/body.ply'
    # args.outer_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir/full.ply'
    # args.inner_labels = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir/parts.npy'
    # args.scale_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir/cent.npy'
    # args.display = True
    # args.save_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir'
    # args.gender = 'male'

    args = lambda: None
    args.inner_path = 'DO_NOT_RELEASE/out_dir_3/May06_xiaohan_basketball_catch/t0001.000/body.ply'
    args.outer_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir_2/lingPurple/full.ply'
    args.inner_labels = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir_2/lingPurple/parts.npy'
    args.scale_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir_2/lingPurple/cent.npy'
    args.display = 0
    args.save_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/out_dir_2/lingPurple/'
    args.gender = 'female'

    _, _, _ = fit_SMPL([args.inner_path], scan_labels=[args.inner_labels], display=args.display, save_path=args.save_path,
                       scale_file=[args.scale_file], gender=args.gender)

    names = [split(s)[1] for s in [args.inner_path]]
    smpl_pkl = [join(args.save_path, n.replace('.ply', '_smpl.pkl')) for n in names]

    _, _, _, _ = fit_SMPLD([args.outer_path], smpl_pkl=smpl_pkl, save_path=args.save_path,
                           scale_file=[args.scale_file], gender=args.gender)
