"""
This file was taken from: https://github.com/bharat-b7/MultiGarmentNetwork
Author: Bharat
"""

import pickle as pkl

import numpy as np
import scipy.sparse as sp

from lib.geometry import get_hres
from lib.smpl.smplpytorch.smplpytorch.native.webuser.serialization import backwards_compatibility_replacements, load_model


class SMPLNaiveWrapper:
    def __init__(self, model_root, assets_root, gender='neutral'):
        # self.project_dir = project_dir
        self.model_root = model_root
        self.assets_root = assets_root
        # experiments name
        # self.exp_name = exp_name
        self.gender = gender
        # self.garment = garment

    def get_smpl_file(self):
        if self.gender == 'neutral':
            return list(self.model_root.glob("**/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"))[0]
        else:
            return list(self.model_root.glob(f"**/lrotmin/lbs_tj10smooth6_0fixed_normalized/{self.gender}/model.pkl"))[0]

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m

    def get_hres_smpl_model_data(self):
        dd = pkl.load(open(self.get_smpl_file()), encoding='latin-1')
        backwards_compatibility_replacements(dd)

        hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'bs_style': dd['bs_style'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def get_hres_smpl(self):
        smpl_m = load_model(self.get_hres_smpl_model_data())
        smpl_m.gender = self.gender
        return smpl_m

    def get_vt_ft(self):
        vt, ft = pkl.load((self.assets_root / "smpl_vt_ft.pkl").open('rb'), encoding='latin-1')
        return vt, ft

    def get_vt_ft_hres(self):
        vt, ft = self.get_vt_ft()
        vt, ft, _ = get_hres(np.hstack((vt, np.ones((vt.shape[0], 1)))), ft)
        return vt[:, :2], ft

    def get_template_file(self):
        fname = self.assets_root / 'template' /'template.obj'
        return fname

    def get_template(self):
        from psbody.mesh import Mesh
        return Mesh(filename=self.get_template_file())

    def get_faces(self):
        fname = self.assets_root / 'template' / 'faces.npy'
        return np.load(fname)

    def get_bmap(self):
        fname = self.assets_root / 'template' / 'bmap.npy'
        return np.load(fname)

    def get_fmap(self):
        fname = self.assets_root / 'template' / 'fmap.npy'
        return np.load(fname)

    def get_bmap_hres(self):
        fname = self.assets_root / 'template' / 'bmap_hres.npy'
        return np.load(fname)

    def get_fmap_hres(self):
        fname = self.assets_root / 'template' / 'fmap_hres.npy'
        return np.load(fname)

    def get_mesh(self, verts):
        from psbody.mesh import Mesh
        return Mesh(v=verts, f=self.get_faces())
