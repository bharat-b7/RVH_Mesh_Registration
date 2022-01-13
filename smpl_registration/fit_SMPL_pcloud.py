"""
fit SMPL to person point cloud captured by multiple kinects

"""
import sys, os
sys.path.append(os.getcwd())
from tqdm import tqdm
import torch
import numpy as np
from psbody.mesh import Mesh, MeshViewer
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Pointclouds, Meshes

from fit_smpl.data_loader import prepare_smpl_pcfit_data, get_ready_pcfiles, get_undone_pcfile
from calib.seq_utils import SeqInfo
from fit_smpl.constants import *
from fit_obj.opt_utils import backward_step, copy_smpl_params, display_smpl_mesh
from fit_obj.object_losses import kpts_loss
from lib.th_smpl_prior import get_prior
from lib.th_hand_prior import HandPrior
from fit_smpl.lib.body_objectives import batch_chamfer, reproject_loss_batchcam, smpl_collision_loss, batch_3djoints_loss
from fit_smpl.utils import save_smplfit_results

def get_loss_weights(phase=None):
    loss_weight = {
        'p2mf': lambda cst, it: 10. ** 2 * cst * (1 + it),
        'chamf':lambda cst, it: 0.8 ** 2 * cst / (1 + it),
        'beta':lambda cst, it: 10. ** 0 * cst / (1 + it),
        'body_prior':lambda cst, it: 10. ** -5 * cst / (1 + it),
        'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
        'joints_2d':lambda cst, it: 0.03 ** 2 * cst / (1 + it),
        'virtual_joints2d':lambda cst, it: 0.03 ** 2 * cst / (1 + it),
        'collision':lambda cst, it: 6.0 ** 2 * cst / (1 + it),
        'hand_prior':lambda cst, it: 50. ** -2 * cst / (1 + it),
        # 'joints3d': lambda cst, it: 10 ** 2 * cst / (1 + it),
        'joints3d': lambda cst, it: 40 ** 2 * cst / (1 + it),
        # 'temporal': lambda cst, it: 0.2 ** 2 * cst / (1 + it),
        'temporal': lambda cst, it: 0.15 ** 2 * cst / (1 + it),
        # 'temporal': lambda cst, it: 0.07 ** 2 * cst / (1 + it),
        # TODO: add temporal constraints
    }
    if phase=='all':
        # increase chamfer weight
        loss_weight['chamf'] = lambda cst, it: 2.5 ** 2 * cst / (1 + it)
    return loss_weight


def forward_step(data_dict, smpl, temporal=False, phase='global'):
    loss_dict = dict()
    verts, _, _, _ = smpl()


    # do not optimize beta for pc fitting
    # if data_dict['beta_init'] is None:
    #     loss_dict['beta'] = torch.mean((smpl.betas) ** 2)
    # else:
    #     loss_dict['beta'] = torch.mean((smpl.betas - data_dict['betas_init']) ** 2)  # average over all batches

    prior = get_prior(smpl.gender)
    prior_loss = torch.mean(prior(smpl.pose[:, :SMPL_POSE_PRAMS_NUM]))
    # if data_dict['pose_init'] is not None:
    #     prior_loss += torch.mean((smpl.pose[:, 3:HANDPOSE_START] - data_dict['pose_init'][:, 3:HANDPOSE_START]) ** 2)
    loss_dict['body_prior'] = prior_loss
    hand_prior = HandPrior(type='grab')
    loss_dict['hand'] = torch.mean(hand_prior(smpl.pose))

    J, face, hands = smpl.get_landmarks()

    # 3D joints loss
    j3d_loss = batch_3djoints_loss(data_dict['joints3d'], J)
    loss_dict['joints3d'] = j3d_loss

    loss_dict['collision'] = torch.mean(smpl_collision_loss(smpl, data_dict["penetration_dist"], data_dict["search_tree"]))

    # temporal all time
    verts_diff = (verts[1:] - verts[:-1]) ** 2
    loss_dict['temporal'] = torch.mean(torch.sum(verts_diff, dim=(1, 2)))

    if phase == 'all':
        pclouds = Pointclouds(data_dict['pc'])
        # add chamfer distance loss
        # chamf, _ = chamfer_distance(pclouds, verts)
        chamf = batch_chamfer(data_dict['pc'], verts, bidirectional=False)
        loss_dict['chamf'] = chamf

        # not helpful if the point cloud is noisy
        # faces = smpl.faces.repeat(verts.shape[0], 1, 1)
        # smpl_meshes = Meshes(verts, faces)
        # s2m_face = point_mesh_face_distance(smpl_meshes,
        #                                     pclouds)  # pytorch3d does not check which device the data stored in!
        # loss_dict['p2mf'] = s2m_face

    elif phase == 'tune':
        # fine tune, use both chamfer and p2mesh
        pclouds = Pointclouds(data_dict['pc'])
        # add chamfer distance loss
        # chamf, _ = chamfer_distance(pclouds, verts)
        chamf = batch_chamfer(data_dict['pc'], verts, bidirectional=False)
        loss_dict['chamf'] = chamf

        # faces = []
        faces = smpl.faces.repeat(verts.shape[0], 1, 1)
        smpl_meshes = Meshes(verts, faces)
        s2m_face = point_mesh_face_distance(smpl_meshes,
                                            pclouds)  # pytorch3d does not check which device the data stored in!
        loss_dict['p2mf'] = s2m_face

    return loss_dict


def optimize_pose(data_dict, smpl, iterations,
                  steps_per_iter, display=None):
    from fit_smpl.lib.th_SMPLH import th_batch_SMPLH_Split
    batch_sz = smpl.pose.shape[0]
    split_smpl = th_batch_SMPLH_Split(batch_sz, top_betas=smpl.betas.data[:, :2],
                                      other_betas=smpl.betas.data[:, 2:],
                                      global_pose=smpl.pose.data[:, :3],
                                      body_pose=smpl.pose.data[:, 3:HANDPOSE_START],
                                      hand_pose=smpl.pose.data[:, HANDPOSE_START:],
                                      faces=smpl.faces, gender=smpl.gender,
                                      trans=smpl.trans, num_betas=NUM_BETAS).cuda()
    optimizer = torch.optim.Adam([split_smpl.trans,
                                  split_smpl.global_pose],
                                 0.01, betas=(0.9, 0.999))
    weight_dict = get_loss_weights()

    mv = MeshViewer() if display is not None else None

    steps_per_iter = 10
    iter_for_global = 5
    # iter_for_kpts = 20 # use only keypoints to optimize pose
    # iter_for_tune = 2 # very slow, only do a few iteractions
    # iterations = 20 # iterations to optimize all losses

    # new set of iterations
    iter_for_kpts = 15  # use only keypoints to optimize pose
    iter_for_tune = 2  # very slow, only do a few iteractions
    iterations = 15  # iterations to optimize all losses

    # for steps_per_iter = 10, not working
    # iter_for_global = 5
    # iter_for_kpts = 6  # use only keypoints to optimize pose
    # iter_for_tune = 2  # very slow, only do a few iteractions
    phase = 'global'
    description = 'Optimizing SMPL global orientation'
    for it in tqdm(range(iterations+iter_for_kpts+iter_for_global+iter_for_tune)):
        loop = tqdm(range(steps_per_iter))
        if it < iter_for_global:
            # steps_per_iter = 10
            description = 'Optimizing SMPL global orientation'
            # print('Optimizing SMPL global orientation')
            # loop.set_description('Optimizing SMPL global orientation')
        elif it==iter_for_global:
            # loop.set_description('Optimizing all SMPL pose using keypoints')
            description= 'Optimizing all SMPL pose using keypoints'
            optimizer = torch.optim.Adam([split_smpl.trans,
                                          split_smpl.global_pose,
                                          split_smpl.body_pose,
                                          # split_smpl.top_betas
                                          ],
                                         0.004, betas=(0.9, 0.999))
            # steps_per_iter = 30
        elif it==iter_for_global+iter_for_kpts:
            # Now optimize full SMPL pose with point clouds
            print('Optimizing all SMPL pose')
            description= 'Optimizing all SMPL pose'
            # loop.set_description('Optimizing all SMPL pose')
            # smaller learning rate, just fine tune the hand etc.
            optimizer = torch.optim.Adam([split_smpl.trans,
                                          split_smpl.global_pose,
                                          split_smpl.body_pose],
                                         0.001, betas=(0.9, 0.999))
            phase = 'all'
            weight_dict = get_loss_weights(phase)
        # elif it == iter_for_global+iterations+iter_for_kpts:
        #     # loop.set_description('Fine tune SMPL pose')
        #     phase = 'tune'
        #     description = 'Fine tune SMPL pose'
        else:
            pass
        loop.set_description(description)

        optimizer.zero_grad()

        for i in loop:
            loss_dict = forward_step(data_dict, split_smpl, phase=phase)
            total_loss = backward_step(loss_dict, weight_dict, it)
            total_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

            if mv is not None:
                display_smpl_mesh(mv, split_smpl, data_dict['pc'])

    # modified for SMPL-H, here SMPL-H has more pose parameters than smpl
    smpl = copy_smpl_params(split_smpl, smpl)

    print('** Optimised smpl pose **')


def fit_SMPLH_mocap(args):
    pc_files = get_ready_pcfiles(args.seq_folder, 'person', use_vcam=False)
    if not args.redo:
        pc_files = get_undone_pcfile(pc_files, args.save_name)

    seq_end = len(pc_files) if args.end is None else args.end
    pc_files = pc_files[args.start:seq_end]
    print("Fitting SMPL to {}, batch size {}, frame {}-{}, save name {}, downsample: {}".format(args.seq_folder,
                                                                                                args.batch_size,
                                                                                args.start,
                                                                 seq_end, args.save_name, args.down_sample))
    if len(pc_files) == 0:
        print('all done.')
        return
    print("Starting frame: {}".format(pc_files[0]))

    seq_info = SeqInfo(args.seq_folder)
    i = 0
    while i < len(pc_files):
        pc_files_batch = pc_files[i:i+args.batch_size]
        data_dict, smplh = prepare_smpl_pcfit_data(pc_files_batch,
                                                   seq_info.get_intrinsic(),
                                                   seq_info.get_config(),
                                                   seq_info.get_gender(),
                                                   mocap=True,
                                                   newdim=True,
                                                   yaxis_down=True,
                                                   vcam=False,
                                                   j3d=True,
                                                   beta_file=seq_info.beta_init(),
                                                   down_sample=args.down_sample,
                                                   kinect_count=seq_info.kinect_count())
        # first optimize global orientation and translation, then fine tune pose
        # iterations, steps_per_iter = 20, 10
        iterations, steps_per_iter = 5, 30

        # sanity check: the y-axis is pointing down
        assert data_dict['config'][1, 0, 0] == 1.0, 'the configuration is incorrect!'

        optimize_pose(data_dict, smplh, iterations, steps_per_iter, args.display)

        # save results
        from fit_obj.utils import create_result_meshpaths
        mesh_paths = create_result_meshpaths(pc_files_batch, name=args.save_name,
                                             ext='ply')

        # TODO: compute reprojection error score and save results
        with torch.no_grad():
            verts, joints, _, _ = smplh()
            # use bidirectional chamfer distance as metric
            # mesh2scandist = batch_chamfer(data_dict['pc'], verts, reduction=None)
            pclouds = Pointclouds(data_dict['pc'])
            chamf, _ = chamfer_distance(pclouds, verts, batch_reduction=None)
            scores = chamf.cpu().detach().numpy().flatten()

            # Save meshes
            save_smplfit_results(verts, smplh.faces, mesh_paths, data_dict['pc'], None, scores, smplh, save_mesh=True)

        i += args.batch_size

    print("All done.")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', "--seq_folder")
    parser.add_argument('-sn', '--save_name', help='subfolder name to save fitted mesh and parameters',
                        required=True)
    parser.add_argument('--display', default=None, action='store_true')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=None)
    # parser.add_argument('-f', '--frame', type=int, help='start frame index')
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-ds', '--down_sample', help='down sample pc to speed up, specify the voxel size', default=0.012, type=float)
    parser.add_argument('-redo', default=False, action='store_true')

    args = parser.parse_args()
    fit_SMPLH_mocap(args)