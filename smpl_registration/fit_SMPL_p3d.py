"""
Code to fit SMPL (pose, shape) to scan using pytorch, kaolin.
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

from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.th_SMPL import th_batch_SMPL, th_batch_SMPL_split_params
from lib.body_objectives import batch_get_pose_obj, torch_pose_obj_data, get_prior_weight, HAND_VISIBLE
from lib.mesh_distance import point_to_surface_vec, batch_point_to_surface_vec_signed, batch_point_to_surface