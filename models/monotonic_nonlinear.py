'''
Efficient and simplified implementation for the monotonic transformation of logits.
This implementation is a rewritten version with improved computational efficiency and clarity.

This module provides a lightweight and computation-efficient wrapper around a monotonic neural network
to transform logits while preserving their intra-order. Such transformations are useful 
for calibration tasks or enforcing order-preserving properties in model outputs.

Original code refers to:
 - https://github.com/AmirooR/IntraOrderPreservingCalibration
 - https://github.com/AWehenkel/UMNN

Author: Wenjian Huang
'''
import torch
import torch.nn as nn
from models.UMNN import MonotonicNN_v1


class MonotonicModel_v1(nn.Module):
    """
    A PyTorch module that performs a monotonic transformation on input logits using a learned
    monotonic neural network. The logits are normalized before transformation.

    Args:
        num_hiddens (int): Number of hidden units in the monotonic neural network.
        nb_steps (int): Number of steps used for numerical integration in the UMNN.
        **kdwags: Additional keyword arguments (currently unused, included for flexibility).
    """    
    def __init__(self, num_hiddens, nb_steps, **kdwags):
        super(MonotonicModel_v1, self).__init__()
        self.model_monotonic = MonotonicNN_v1(num_hiddens, nb_steps=nb_steps)

        # Parameters for normalization (not trainable)
        self.meanlogit = nn.parameter.Parameter(data=torch.tensor([0.0]),requires_grad=False)
        self.logitstd = nn.parameter.Parameter(data=torch.tensor([1.0]),requires_grad=False)

    def forward(self, logits):
        """
        Forward pass through the monotonic transformation model.

        Steps:
        1. Normalize logits with log-sum-exp for numerical stability.
        2. Standardize the logits using pre-defined (non-trainable) mean and std.
        3. Flatten and apply the monotonic neural network transformation.
        4. Reshape the output back to the original logits shape.

        Args:
            logits (Tensor): A tensor of raw logits with shape (batch_size, num_classes).

        Returns:
            Tensor: Monotonically transformed logits with the same shape as input.
        """        
        # Normalize logits using log-sum-exp for numerical stability
        logits = logits - torch.logsumexp(logits,dim=1,keepdim=True)

        # Standardize using fixed mean and std
        logits = (logits - self.meanlogit) / self.logitstd

        # Flatten logits for monotonic transformation
        flat_out = logits.contiguous().view(-1).unsqueeze(1)

        # Apply monotonic transformation and reshape back
        mono_out = self.model_monotonic(flat_out).contiguous().view(logits.shape)
        return mono_out


