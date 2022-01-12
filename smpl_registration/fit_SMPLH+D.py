"""
fit SMPLH+offset to scans

created by Xianghui, 12 January 2022
"""
import sys, os
sys.path.append(os.getcwd())
from os.path import split, join, exists
import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from lib.mesh_laplacian import mesh_laplacian_smoothing
from smpl_registration.fit_SMPLH import SMPLHFitter


class SMPLDFitter(SMPLHFitter):
    def __init__(self, model_root, save_name='smpld', debug=False):
        super(SMPLDFitter, self).__init__(model_root, save_name, debug)

    def fit(self, scans, pose_files, smpl_pkl, gender='male', save_path=None):
        if smpl_pkl is None or smpl_pkl[0] is None:
            print('SMPL not specified, fitting SMPL now')
            pose, betas, trans = super(SMPLDFitter, self).fit(scans, pose_files, gender, save_path)
        else:
            # load from fitting results
            pose, betas, trans = self.load_smpl_params(smpl_pkl)

        betas, pose, trans = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans)
        # Batch size
        batch_sz = len(scans)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, pose, betas, trans)

        # Load scans and center them. Once smpl is registered, move it accordingly.
        # Do not forget to change the location of 3D joints/ landmarks accordingly.
        th_scan_meshes = self.load_scans(scans)

        # optimize offsets
        self.optimize_offsets(th_scan_meshes, smpl, 6, 10)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smpld')

    def forward_step_offset(self, th_scan_meshes, smpl, init_smpl_lap):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
        """
        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        lap_new = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # reference: https://github.com/NVIDIAGameWorks/kaolin/blob/v0.1/kaolin/metrics/mesh.py#L155
        loss['lap'] = torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))
        loss['offsets'] = torch.mean(torch.mean(smpl.offsets ** 2, axis=1))
        return loss

    def optimize_offsets(self, th_scan_meshes, smpl, iterations, steps_per_iter):
        # Optimizer
        optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.trans, smpl.betas], 0.005, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # precompute initial laplacian of the smpl meshes
        bz = smpl.offsets.shape[0]
        verts, _, _, _ = smpl()
        verts_list = [verts[i].clone().detach() for i in range(bz)]
        faces_list = [torch.tensor(smpl.faces) for i in range(bz)]
        init_smpl = Meshes(verts_list, faces_list)
        init_lap = mesh_laplacian_smoothing(init_smpl)

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL+D')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_offset(th_scan_meshes, smpl, init_lap)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Lx100. Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item() * 100)
                loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: 100**2*cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

def main(args):
    fitter = SMPLDFitter(args.model_root, debug=args.display)
    fitter.fit([args.scan_path], [args.pose_file], [args.smpl_pkl], args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    # parser.add_argument('scan_path', type=str)
    # parser.add_argument('pose_file', type=str)
    # parser.add_argument('pose_init', type=str, help='init smpl pose, if exist')
    # parser.add_argument('save_path', type=str) # save path for all scans
    # parser.add_argument('-gender', type=str, default='male') # can be female
    parser.add_argument('-smpl_pkl', type=str, default=None)  # In case SMPL fit is already available
    # parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-mr', '--model_root', default="/BS/xxie2020/static00/mysmpl/smplh")
    args = parser.parse_args()

    # args = lambda: None
    args.scan_path = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/rp_alison_posed_017_30k.obj'
    args.pose_file = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/pose3d/rp_alison_posed_017_30k.json'
    args.display = True
    args.save_path = '/BS/xxie-2/work/MPI_MeshRegistration/test_data'
    args.gender = 'female'
    args.smpl_pkl = '/BS/xxie-2/work/MPI_MeshRegistration/test_data/rp_alison_posed_017_30k_smpl.pkl'

    main(args)