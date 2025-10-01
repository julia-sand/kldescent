"""
Stores a function for a normalising flow neural network
"""

import os
import math

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.distributions as dist

class CouplingLayer(dist.Transform):
    def __init__(self, input_dim, hidden_dim, mask, base_transform_type='affine'):
        super().__init__()
        self.input_dim = input_dim
        # Ensure mask is a binary tensor (0s and 1s)
        self.register_buffer('mask', mask) 

        # Define the network 's_t_network' that computes scale and translation params
        self.s_t_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2) # Output scale and shift for all dims
        )
        
        self.bijective = True
        # For affine: domain = real, codomain = real
        self.domain = dist.constraints.real_vector
        self.codomain = dist.constraints.real_vector
        
        # Optional: Use built-in transforms like AffineTransform 
        # if base_transform_type == 'affine' ... (implementation detail)

    def _call(self, x):
        """ Applies the transform: x -> z """
        x_masked = x * self.mask
        s_t_params = self.s_t_network(x_masked)
        # Split the output into scale (s) and translation (t)
        # Ensure scale is positive, e.g., using tanh for stability + scaling
        s = torch.tanh(s_t_params[..., :self.input_dim]) 
        t = s_t_params[..., self.input_dim:]

        # Apply transformation only to unmasked elements
        # z = x_masked + (x_unmasked * exp(s) + t) * (1 - mask) 
        z = self.mask * x + (1 - self.mask) * (x * torch.exp(s) + t)
        return z

    def _inverse(self, z):
        """ Applies the inverse transform: z -> x """
        z_masked = z * self.mask
        s_t_params = self.s_t_network(z_masked)
        s = torch.tanh(s_t_params[..., :self.input_dim])
        t = s_t_params[..., self.input_dim:]

        # Apply inverse transformation only to unmasked elements
        # x = z_masked + ((z_unmasked - t) * exp(-s)) * (1 - mask)
        x = self.mask * z + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x

    def log_abs_det_jacobian(self, x, z):
        """ Computes log |det J(x)| """
        x_masked = x * self.mask
        s_t_params = self.s_t_network(x_masked)
        s = torch.tanh(s_t_params[..., :self.input_dim])

        # Log determinant is the sum of 's' for the transformed dimensions
        log_det_jacobian = (1 - self.mask) * s 
        # Sum over the transformed dimensions
        return log_det_jacobian.sum(-1) 

# Example Usage:
input_dim = 2
hidden_dim = 64
num_flows = 5

# Define base distribution (e.g., standard Normal)
base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))

# Create masks (alternating between steps is common)
masks = []
for i in range(num_flows):
    mask = torch.zeros(input_dim)
    mask[i % 2::2] = 1 # Simple alternating mask
    # Or more sophisticated masking strategies exist
    masks.append(mask)
    
# Create a sequence of transforms (coupling layers)
transforms = []
for i in range(num_flows):
    # Alternate which part is identity vs transformed
    current_mask = masks[i] if i % 2 == 0 else (1 - masks[i]) 
    transforms.append(CouplingLayer(input_dim, hidden_dim, current_mask))
    # Optional: Add permutation/activation normalization layers between flows

# Build the transformed distribution
flow_dist = dist.TransformedDistribution(base_dist, transforms)

# --- Training ---
optimizer = torch.optim.Adam(flow_dist.parameters(), lr=1e-4)
 
for data_batch in dataloader:
    optimizer.zero_grad()
    # Calculate log probability of the data batch
    log_prob = flow_dist.log_prob(data_batch) # Shape: [batch_size]

    # Maximize log-likelihood -> Minimize negative log-likelihood
    loss = -log_prob.mean() 
     
    loss.backward()
    optimizer.step()

