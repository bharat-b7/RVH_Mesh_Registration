"""
This file was taken from: https://github.com/bharat-b7/MultiGarmentNetwork
Author: Bharat
"""

import numpy as np
from os.path import join
from lib.smpl.smplpytorch.smplpytorch.native.webuser.serialization import load_model


class SMPLNaiveWrapper:
    def __init__(self, model_root, gender='neutral'):
        self.model_root = model_root
        self.gender = gender

    def get_smpl_file(self):
        return join(self.model_root, f"SMPL_{self.gender}.pkl")

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m

    def get_faces(self):
        fname = self.model_root / "template" / "faces.npy"
        return np.load(fname)

    def get_mesh(self, verts):
        from psbody.mesh import Mesh
        return Mesh(v=verts, f=self.get_faces())
