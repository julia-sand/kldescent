"""
Stores a function for a normalising flow neural network
"""

import os
import math

import torch
from torch import nn

from ..utils.parser import du_network_params,rho_network_params,fetch_checkpoint_path
