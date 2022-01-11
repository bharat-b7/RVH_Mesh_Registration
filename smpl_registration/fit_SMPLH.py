"""
fit SMPLH to the scan

this script has not been tested
"""
import os
from os.path import split, join, exists
import sys
import ipdb
import json
import torch
import numpy as np
import pickle as pkl

from psbody.mesh import Mesh, MeshViewer, MeshViewers
from tqdm import tqdm

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.io import load_ply, save_ply

from lib.smpl_paths import SmplPaths
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.th_SMPLH import th_batch_SMPLH, th_batch_SMPLH_Split
from lib.body_objectives import batch_get_pose_obj, get_prior_weight, HAND_VISIBLE
from lib.smpl.priors.th_hand_prior import mean_hand_pose


def plot_points(pts, cols=None):
	from psbody.mesh.sphere import Sphere
	temp = Sphere(np.zeros((3)), 1.).to_mesh()
	meshes = [Mesh(vc='SteelBlue' if cols is None else cols[n], f=temp.f, v=temp.v * 10. ** -2 + p) for n, p in
	          enumerate(pts)]
	return meshes


def get_loss_weights():
	"""Set loss weights"""

	loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
	               'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
	               'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
	               'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
	               'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
	               'lap': lambda cst, it: cst / (1 + it),
	               'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
	               }
	return loss_weight


def save_meshes(meshes, save_paths):
	print('Mesh saved at', save_paths[0])
	for m, s in zip(meshes, save_paths):
		save_ply(s, m.verts_list()[0].cpu(), m.faces_list()[0].cpu())


def forward_step(th_scan_meshes, smpl, th_pose_3d=None):
	"""
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """
	# Get pose prior
	prior = get_prior(smpl.gender)

	# forward
	verts, _, _, _ = smpl()
	th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

	# losses
	loss = dict()
	loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
	loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
	loss['betas'] = torch.mean(smpl.betas ** 2, axis=1)
	loss['pose_pr'] = prior(smpl.pose)
	if th_pose_3d is not None:
		loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl)
	return loss


def forward_step_pose_only(smpl, th_pose_3d, prior_weight):
	"""
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """
	# Get pose prior
	prior = get_prior(smpl.gender)

	# losses
	loss = dict()
	loss['pose_pr'] = prior(smpl.pose, prior_weight)
	loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
	return loss


def backward_step(loss_dict, weight_dict, it):
	w_loss = dict()
	for k in loss_dict:
		w_loss[k] = weight_dict[k](loss_dict[k], it)

	tot_loss = list(w_loss.values())
	tot_loss = torch.stack(tot_loss).sum()
	return tot_loss


def optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None, display=None):
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

	for it in range(iterations):
		loop = tqdm(range(steps_per_iter))
		loop.set_description('Optimizing SMPL')
		for i in loop:
			optimizer.zero_grad()
			# Get losses for a forward pass
			loss_dict = forward_step(th_scan_meshes, smpl, th_pose_3d)
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
				scan_mesh = Mesh(v=th_scan_meshes.verts_list()[display].cpu().detach().numpy(),
				                 f=th_scan_meshes.faces_list()[display].cpu().numpy(), vc=np.array([0, 1, 0]))
				mv.set_static_meshes([scan_mesh, smpl_mesh])

	print('** Optimised smpl pose and shape **')


def optimize_pose_only(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d, prior_weight, display=None):
	"""
    Initially we want to only optimize the global rotation of SMPL. Next we optimize full pose.
    We optimize pose based on the 3D keypoints in th_pose_3d.
    :param  th_pose_3d: array containing the 3D keypoints.
    :param prior_weight: weights corresponding to joints depending on visibility of the joint in the 3D scan.
                         eg: hand could be inside pocket.
    """

	batch_sz = smpl.pose.shape[0]
	split_smpl = th_batch_SMPLH_Split(batch_sz, top_betas=smpl.betas.data[:, :2],
	                                        other_betas=smpl.betas.data[:, 2:],
	                                        global_pose=smpl.pose.data[:, :3],
									  		body_pose=smpl.pose.data[:, 3:66],
									  		hand_pose=smpl.pose.data[:, 66:],
	                                        faces=smpl.faces, gender=smpl.gender).cuda()
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
			                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
		else:
			loop.set_description('Optimizing SMPL pose only')

		for i in loop:
			optimizer.zero_grad()
			# Get losses for a forward pass
			loss_dict = forward_step_pose_only(split_smpl, th_pose_3d, prior_weight)
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
				scan_mesh = Mesh(v=th_scan_meshes.verts_list()[display].cpu().detach().numpy(),
				                 f=th_scan_meshes.faces_list()[display].cpu().numpy(), vc=np.array([0, 1, 0]))

				mv.set_dynamic_meshes([smpl_mesh, scan_mesh])

	# Put back pose, shape and trans into original smpl
	smpl.pose.data = split_smpl.pose.data
	smpl.betas.data = split_smpl.betas.data
	smpl.trans.data = split_smpl.trans.data

	print('** Optimised smpl pose **')


def fit_SMPLH(scans, pose_files=None, gender='male', save_path=None, display=None):
	"""
    :param save_path:
    :param scans: list of scan paths
    :param pose_files:
    :return:
    """
	# Get SMPL faces
	sp = SmplPaths(gender=gender)
	smpl_faces = sp.get_faces()
	th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()
	num_betas = 10

	# Batch size
	batch_sz = len(scans)

	# Set optimization hyper parameters
	iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 3, 2, 30, 30

	if False:
		"""Test by loading GT SMPL params"""
		betas, pose, trans = torch.tensor(GT_SMPL['betas'].astype('float32')).unsqueeze(0), torch.tensor(
			GT_SMPL['pose'].astype('float32')).unsqueeze(0), torch.zeros((batch_sz, 3))
	else:
		prior = get_prior(gender=gender)
		pose_init = torch.zeros((batch_sz, 156))
		pose_init[:, 3:66] = prior.mean
		hand_mean = mean_hand_pose()
		hand_init = torch.tensor(hand_mean, dtype=torch.float).cuda()
		pose_init[:, 66:] = hand_init
		betas, pose, trans = torch.zeros((batch_sz, num_betas)), pose_init, torch.zeros((batch_sz, 3))

	# Init SMPL, pose with mean smpl pose, as in ch.registration
	smpl = th_batch_SMPLH(batch_sz, betas, pose, trans, faces=th_faces, num_betas=num_betas).cuda()

	# Load scans and center them. Once smpl is registered, move it accordingly.
	# Do not forget to change the location of 3D joints/ landmarks accordingly.
	verts, faces, centers = [], [], []
	for scan in scans:
		print('scan path ...', scan)
		v, f = load_ply(scan)
		verts.append(v)
		faces.append(f)

	th_scan_meshes = Meshes(verts, faces).to('cuda')

	# Load pose information if pose file is given
	# Bharat: Shouldn't we structure th_pose_3d as [key][batch, ...] as opposed to current [batch][key]? See batch_get_pose_obj() in body_objectives.py
	th_pose_3d = None
	if pose_files is not None:
		th_no_right_hand_visible, th_no_left_hand_visible, th_pose_3d = [], [], []
		for pose_file in pose_files:
			with open(pose_file) as f:
				pose_3d = json.load(f)
				th_no_right_hand_visible.append(
					np.max(np.array(pose_3d['hand_right_keypoints_3d']).reshape(-1, 4)[:, 3]) < HAND_VISIBLE)
				th_no_left_hand_visible.append(
					np.max(np.array(pose_3d['hand_left_keypoints_3d']).reshape(-1, 4)[:, 3]) < HAND_VISIBLE)

				pose_3d['pose_keypoints_3d'] = torch.from_numpy(
					np.array(pose_3d['pose_keypoints_3d']).astype(np.float32).reshape(-1, 4))
				pose_3d['face_keypoints_3d'] = torch.from_numpy(
					np.array(pose_3d['face_keypoints_3d']).astype(np.float32).reshape(-1, 4))
				pose_3d['hand_right_keypoints_3d'] = torch.from_numpy(
					np.array(pose_3d['hand_right_keypoints_3d']).astype(np.float32).reshape(-1, 4))
				pose_3d['hand_left_keypoints_3d'] = torch.from_numpy(
					np.array(pose_3d['hand_left_keypoints_3d']).astype(np.float32).reshape(-1, 4))
			th_pose_3d.append(pose_3d)

		prior_weight = get_prior_weight(th_no_right_hand_visible, th_no_left_hand_visible).cuda()

		# Optimize pose first
		optimize_pose_only(th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, th_pose_3d, prior_weight,
		                   display=None if display is None else 0)

	# Optimize pose and shape
	optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d,
	                    display=None if display is None else 0)

	verts, _, _, _ = smpl()
	th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

	if save_path is not None:
		if not exists(save_path):
			os.makedirs(save_path)

		names = [split(s)[1] for s in scans]

		# Save meshes
		save_meshes(th_smpl_meshes, [join(save_path, n.replace('.ply', '_smpl.ply')) for n in names])
		save_meshes(th_scan_meshes, [join(save_path, n) for n in names])

		# Save params
		for p, b, t, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
		                      smpl.trans.cpu().detach().numpy(), names):
			smpl_dict = {'pose': p, 'betas': b, 'trans': t}
			pkl.dump(smpl_dict, open(join(save_path, n.replace('.obj', '_smpl.pkl')), 'wb'))

		return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str)
    parser.add_argument('pose_file', type=str)
    parser.add_argument('pose_init', type=str, help='init smpl pose, if exist')
    parser.add_argument('save_path', type=str)
    parser.add_argument('-gender', type=str, default='male') # can be female
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()

    # args = lambda: None
    # args.scan_path = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/rp_alison_posed_017_30k.obj'
    # args.pose_file = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/pose3d/rp_alison_posed_017_30k.json'
    # args.display = False
    # args.save_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data'
    # args.gender = 'female'

    _, _, _ = fit_SMPLH([args.scan_path], pose_files = [args.pose_file], display=args.display, save_path=[args.save_path], gender=args.gender)
