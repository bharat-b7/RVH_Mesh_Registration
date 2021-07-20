"""
Script to render scan from multiple views.
"""

import torch, cv2
import pickle as pkl
import torch.nn.functional as F
from skimage.io import imread