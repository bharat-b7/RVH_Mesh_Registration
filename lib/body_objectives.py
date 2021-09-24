"""
Objectives used in single mesh registrations
Original author: Garvita
Edited by: Bharat
"""
import pickle as pkl
from os.path import join

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras
from torch.nn.functional import mse_loss
# from psbody.smpl import load_model
# from psbody.smpl.serialization import backwards_compatibility_replacements

# from lib.smpl_paths import ROOT
from lib.torch_functions import batch_sparse_dense_matmul
# from lib.mesh_distance import point_to_surface_vec, batch_point_to_surface_vec_signed

# HAND_VISIBLE = 0.2

# part2num = {
#     'global': 0, 'leftThigh': 1, 'rightThigh': 2, 'spine': 3, 'leftCalf': 4, 'rightCalf': 5, 'spine1': 6, 'leftFoot': 7,
#     'rightFoot': 8, 'spine2': 9, 'leftToes': 10, 'rightToes': 11, 'neck': 12, 'leftShoulder': 13, 'rightShoulder': 14,
#     'head': 15, 'leftUpperArm': 16, 'rightUpperArm': 17, 'leftForeArm': 18, 'rightForeArm': 19, 'leftHand': 20,
#     'rightHand': 21, 'leftFingers': 22, 'rightFingers': 23
# }


# def get_prior_weight(no_right_hand_batch, no_left_hand_batch):
#     pr_w = np.ones((len(no_right_hand_batch), 69)).astype('float32')
#
#     # for (no_right_hand, no_left_hand) in zip(no_right_hand_batch,no_left_hand_batch )
#     for i in range(len(no_right_hand_batch)):
#         if no_right_hand_batch[i]:
#             pr_w[i, (part2num['rightFingers'] - 1) * 3:part2num['rightFingers'] * 3] = 1e5
#             pr_w[i, (part2num['rightHand'] - 1) * 3:part2num['rightHand'] * 3] = 1e3
#
#         if no_left_hand_batch[i]:
#             pr_w[i, (part2num['leftFingers'] - 1) * 3:part2num['leftFingers'] * 3] = 1e5
#             pr_w[i, (part2num['leftHand'] - 1) * 3:part2num['leftHand'] * 3] = 1e3
#
#         pr_w[i, (part2num['rightToes'] - 1) * 3:part2num['rightToes'] * 3] = 1e3
#         pr_w[i, (part2num['leftToes'] - 1) * 3:part2num['leftToes'] * 3] = 1e3
#     # pr_w = np.ones((len(no_right_hand_batch), 69)).astype('float32')
#     return torch.from_numpy(pr_w)


def torch_pose_obj_data(model_root, batch_size=1):
    """
    Keypoint operators on SMPL verts.
    """
    body25_reg = pkl.load(open(join(model_root, 'regressors/body_25_openpose_joints.pkl'), 'rb'), encoding="latin1").T
    face_reg = pkl.load(open(join(model_root, 'regressors/face_70_openpose_joints.pkl'), 'rb'), encoding="latin1").T
    hand_reg = pkl.load(open(join(model_root, 'regressors/hands_42_openpose_joints.pkl'), 'rb'), encoding="latin1").T
    body25_reg_torch = torch.sparse_coo_tensor(body25_reg.nonzero(), body25_reg.data, body25_reg.shape)
    face_reg_torch = torch.sparse_coo_tensor(face_reg.nonzero(), face_reg.data, face_reg.shape)
    hand_reg_torch = torch.sparse_coo_tensor(hand_reg.nonzero(), hand_reg.data, hand_reg.shape)

    return torch.stack([body25_reg_torch]*batch_size), torch.stack([face_reg_torch]*batch_size),\
           torch.stack([hand_reg_torch]*batch_size)


# def batch_get_pose_obj(th_pose_3d, smpl, init_pose=False):
#     """
#     Comapre landmarks/keypoints ontained from the existing SMPL against those observed on the scan.
#     Naive implementation as batching currently implies just looping.
#     """
#     batch_size = len(th_pose_3d)
#     verts, _, _, _ = smpl.forward()
#     J, face, hands = smpl.get_landmarks()
#
#     device = th_pose_3d.device
#     J_observed = torch.stack([th_pose_3d[i]['pose_keypoints_3d'] for i in range(batch_size)]).to(device)
#     face_observed = torch.stack([th_pose_3d[i]['face_keypoints_3d'] for i in range(batch_size)]).to(device)
#
#     # Bharat: Why do we need to loop? Shouldn't we structure th_pose_3d as [key][batch, ...] as opposed to current [batch][key]?
#     # This would allow us to remove the loop here.
#     hands_observed = torch.stack(
#         [torch.cat((th_pose_3d[i]['hand_left_keypoints_3d'], th_pose_3d[i]['hand_right_keypoints_3d']), dim=0) for i in
#         range(batch_size)]).to(device)
#
#     idx_mask = hands_observed[:, :, 3] < HAND_VISIBLE
#     hands_observed[:, :, :3][idx_mask] = 0.
#
#     if init_pose:
#         pose_init_idx = torch.LongTensor([0, 2, 5, 8, 11])
#         return (((J[:, pose_init_idx, : ] - J_observed[:, pose_init_idx, : 3])
#          *J_observed[:, pose_init_idx,  3].unsqueeze(-1)) ** 2).mean()
#     else:
#         return ( (((J - J_observed[:, :, :3]) *J_observed[:, :, 3].unsqueeze(-1))**2).mean() +
#                    (((face - face_observed[:, :,: 3]) *face_observed[:, :,  3].unsqueeze(-1))**2).mean() +
#                    (((hands - hands_observed[:, :, :3]) *hands_observed[:, :,  3].unsqueeze(-1))**2).mean()).unsqueeze(0)/3
#         # return (((J - J_observed[:, :, :3]) *J_observed[:, :, 3].unsqueeze(-1))**2).mean().unsqueeze(0)   #only joints


# def get_pose_obj(pose_3d, smpl):
#     # Bharat: Why do we have torch, chumpy and numpy in the same function. It seems all the ops can be done in torch.
#
#     model_root = smpl.model_root
#     body25_reg_torch, face_reg_torch, hand_reg_torch = torch_pose_obj_data(model_root)
#     ipdb.set_trace()
#     verts, _, _, _ = smpl.forward()
#     J = batch_sparse_dense_matmul(body25_reg_torch, verts)
#     face = batch_sparse_dense_matmul(face_reg_torch, verts)
#     hands = batch_sparse_dense_matmul(hand_reg_torch, verts)
#
#     J_observed = pose_3d['pose_keypoints_3d']
#     face_observed = pose_3d['face_keypoints_3d']
#     hands_observed = np.vstack((pose_3d['hand_left_keypoints_3d'], pose_3d['hand_right_keypoints_3d']))
#
#     hands_observed[:, 3][hands_observed[:, 3] < HAND_VISIBLE] = 0.
#
#     #
#     return ch.vstack((
#         (J - J_observed[:, :3]) * J_observed[:, 3].reshape((-1, 1)),
#         (face - face_observed[:, :3]) * face_observed[:, 3].reshape((-1, 1)),
#         (hands - hands_observed[:, :3]) * hands_observed[:, 3].reshape((-1, 1)),
#         ))


body_keypoints_weights = torch.tensor([
    0.6, 0.6,
    1.0, 1.0, 1.0,  # right arm
    1.0, 1.0, 1.0,  # left arm
    0.6, 0.6,
    1.0, 1.0,  # right knee and leg
    1.0, 1.0,  # left knee and leg
    0.8,  # left foot
    1.0, 1.0, 1.0, 1.0,  # head
    0.8, 0.8, 0.8,  # left foot
    0.8, 0.8, 0.8  # right foot
], dtype=torch.float).reshape((25, 1))
bodyhand_keypoints_weights = 0.6 * torch.ones((67, 1), dtype=torch.float)
bodyhand_keypoints_weights[:25, 0] = body_keypoints_weights[:, 0]
joint_weights = 0.6 * torch.ones((137, 1), dtype=torch.float)  # for all body joints
joint_weights[:25, 0] = body_keypoints_weights[:, 0]


def project_points(kinect_pose, color_mat, depth2color, points):
    """project 3d points to 2d color image
    color_mat: (3, 4)
    kinect_pose: (3, 4)
    d2c: (3, 4)
    """
    smpl_joints_local = torch.matmul(points - kinect_pose[:, 3], kinect_pose[:, :3].T)
    smpl_joints_color = torch.matmul(smpl_joints_local, depth2color[:, :3].T) + depth2color[:, 3]

    cx, cy = color_mat[0, 2], color_mat[1, 2]
    fx, fy = color_mat[0, 0], color_mat[1, 1]
    smpl_joints_proj_x = cx + fx * torch.div(smpl_joints_color[:, :, 0], smpl_joints_color[:, :, 2])
    smpl_joints_proj_y = cy + fy * torch.div(smpl_joints_color[:, :, 1], smpl_joints_color[:, :, 2])
    smpl_joints_proj_xy = torch.stack((smpl_joints_proj_x, smpl_joints_proj_y), axis=-1)
    return smpl_joints_proj_xy


def batch_reprojection_loss_kinect(img_bodyjoints, smpl_bodyjoints, color_mats, depth2colors, kinect_poses):
    """
    Reprojection loss between 2D joints and SMPL joints projected  using calibrated Kinect cameras.

    Parameters:
        img_bodyjoints: (B, N, 3*C), body joints detected by openpose from kinect images, C is the number of cameras
        smpl_bodyjoints: (B, N, 3) SMPL body joints
        color_mats: (3, 3, 3), intrinsics of 3 kinects' color camera
        depth2colors: (3, 3, 4), depth to color transform of each kinect
        kinect_poses: (3, 3, 4), poses of 3 kinects

    Returns
        loss value
        smpl_bodyjoints projected to image planes using cameras
    """
    device = img_bodyjoints.device

    kinect_count = color_mats.shape[0]
    if img_bodyjoints.shape[1] == 25:
        weights = body_keypoints_weights.to(device)
    elif img_bodyjoints.shape[1] == 67:
        weights = bodyhand_keypoints_weights.to(device)
    else:
        weights = joint_weights.to(device)
    sum_loss = img_bodyjoints.new_zeros(img_bodyjoints.shape[0])
    smpl_joints_projected = []
    for k in range(kinect_count):
        smpl_joints_proj_xy = project_points(kinect_poses[k, :, :], color_mats[k, :, :], depth2colors[k, :, :], smpl_bodyjoints)
        smpl_joints_projected.append(smpl_joints_proj_xy[0])  # return the first image for visualization

        loss = mse_loss(img_bodyjoints[:,:, k*3:k*3+2], smpl_joints_proj_xy, reduction='none')
        mse_sum = torch.sum(loss, axis=-1)
        mse_weighted = torch.matmul(mse_sum*img_bodyjoints[:, :, k*3+2], weights)  # mask invalid predictions out
        sum_loss = sum_loss + mse_weighted
    return sum_loss/float(kinect_count), smpl_joints_projected


def batch_reprojection_loss_vcam(img_bodyjoints, smpl_bodyjoints, cameras: PerspectiveCameras, image_size):
    """
    Reprojection loss between 2D joints and SMPL joints projected  using virtual Pytorch3D perspective cameras.

    Parameters:
        img_bodyjoints: (B, N, 3*C), body joints detected by openpose from rendered images, C is the number of cameras
        smpl_bodyjoints: (B, N, 3), smpl body joints
        cameras:
        image_size:

    Returns:
        loss value
        smpl_bodyjoints projected to image planes using cameras
    """
    device = img_bodyjoints.device

    if img_bodyjoints.shape[1] == 25:
        weights = body_keypoints_weights.to(device)
    else:
        weights = joint_weights.to(device)

    sum_loss = img_bodyjoints.new_zeros(img_bodyjoints.shape[0])
    smpl_joints_projected = []
    for k, cam in enumerate(cameras):
        smpl_joints_proj_xy = cam.transform_points_screen(smpl_bodyjoints, image_size)[:, :, :2]
        smpl_joints_projected.append(smpl_joints_proj_xy)

        loss = mse_loss(img_bodyjoints[:, :, k * 3:k * 3 + 2], smpl_joints_proj_xy, reduction='none')
        mse_sum = torch.sum(loss, axis=-1)
        mse_weighted = torch.matmul(mse_sum * img_bodyjoints[:, :, k*3+2], weights)  # mask invalid predictions out
        sum_loss = sum_loss + mse_weighted
    return sum_loss / float(len(cameras)), smpl_joints_projected


# def smpl_collision_loss(smpl, pen_distance, search_tree, device='cuda'):
#     "inter-penetration loss"
#     verts, _, _, _ = smpl()
#     face_tensor = smpl.faces # (F, 3)
#     bs, nv = verts.shape[:2]
#     nf, _ = face_tensor.shape[:2]
#     faces_idx = face_tensor + \
#                 (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]
#     triangles = verts.view([-1, 3])[faces_idx]  # (B, F, 3, 3)
#
#     with torch.no_grad():
#         # to find out the collision index
#         collision_idxs = search_tree(triangles)
#
#     pen_loss = pen_distance(triangles, collision_idxs)
#     return pen_loss


# def batch_3djoints_loss(pc_bodyjoints, smpl_bodyjoints):
#     """
#     mse loss between 3d keyjoints in pc and SMPL body joints, weighted
#
#     pc_bodyjoints: (B, N, 4), the forth column is prediction score
#     smpl_bodyjoints: (B, N, 3)
#     """
#     if pc_bodyjoints.shape[1] == 25:
#         weights = body_keypoints_weights
#     else:
#         weights = joint_weights
#     loss = mse_loss(pc_bodyjoints[:, :, :3], smpl_bodyjoints, reduction='none')
#     mse_sum = torch.sum(loss, axis=-1) * pc_bodyjoints[:, :, 3]
#     mse_weighted = torch.matmul(mse_sum, weights)
#     return torch.mean(mse_weighted)


# def batch_3djoints_loss(pc_bodyjoints, smpl_bodyjoints, kinect_poses):
#     """
#     3d joints loss between SMPL and pc
#     pc_bodyjoints: (B, N, 9), in world coordinate system
#     """
#     from chamferdist import ChamferDistance
#     chamferDist = ChamferDistance()
#
#     kinect_count = kinect_poses.shape[0]
#     sum_loss = pc_bodyjoints.new_zeros(pc_bodyjoints.shape[0])
#     smpl_joints_projected = []
#     for k in range(kinect_count):
#         smpl_joints_local = torch.matmul(smpl_bodyjoints - kinect_poses[k, :, 3], kinect_poses[k, :, :3].T)
#         chamf = chamferDist(smpl_joints_local, pc_bodyjoints[:, :, k*3:(k+1)*3])
#
#         sum_loss = sum_loss + chamf
#     return sum_loss / float(kinect_count), smpl_joints_projected
