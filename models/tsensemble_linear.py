"""
Implements an ensemble of temperature scaling transformations for logit calibration.

This module applies multiple scalar temperature scaling transformations to the input logits,
and aggregates them using a learnable weighted average. It serves as a logit linear 
transformation ensemble approach for predictive probability calibration in classification tasks.

Author: Wenjian Huang
"""

from torch import nn
import torch
import math

class EnsembleTempScal(nn.Module):
    def __init__(self,scalarnum):
        """
        Initializes the ensemble temperature scaling module.
        
        Args:
            scalarnum (int): The number of learnable scalar temperatures to ensemble.
        """        
        super().__init__()
        # Learnable temperature parameters (inverse temperature values); shape: (1, 1, scalarnum)
        self.para = torch.nn.Parameter(torch.empty(1,1,scalarnum), requires_grad=True)
        nn.init.kaiming_uniform_(self.para, a=math.sqrt(5))
        
        # Learnable weights for combining the ensemble outputs; shape: (1, 1, scalarnum)
        self.weight = torch.nn.Parameter(torch.empty(1,1,scalarnum), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        """
        Applies the ensemble temperature scaling to input logits.
        
        Args:
            x (torch.Tensor): Uncalibrated logits tensor of shape (batch_size, num_classes).
        
        Returns:
            torch.Tensor: Calibrated class probabilities of shape (batch_size, num_classes).
        """
        # Normalize logits using log-softmax to stabilize computation        
        x = x - torch.logsumexp(x,dim=1,keepdim=True)
        
        # Ensure positive temperatures using absolute values
        inv_temp = torch.abs(self.para) # shape: (1, 1, scalarnum)

        # Apply each temperature scaling transformation to the logits
        x_calib_list = x[...,None] * inv_temp # shape: (batch_size, num_classes, scalarnum)
        
        norm_weight = torch.abs(self.weight) / torch.sum(torch.abs(self.weight))

        # Weighted sum of softmax-calibrated probabilities across the ensemble
        calprobs = (torch.softmax(x_calib_list,dim=1) * norm_weight).sum(2)
        
        return calprobs # shape: (batch_size, num_classes)






