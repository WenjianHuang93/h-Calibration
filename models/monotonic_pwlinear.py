"""
This module implements a learnable piecewise linear transformation over logit values.
The mapping is designed to be a monotonic, piecewise linear function that preserves 
both the order and relative accuracy of the original logits (intra-logit preservation).

Author: Wenjian Huang
"""

import torch
import torch.nn as nn

class PWLinearModel(nn.Module):
    """
    A wrapper model that applies a learnable monotonic piecewise linear transformation
    to normalized logits. The normalization is done via log-softmax to ensure numerical
    stability and invariance to logit scale.

    Args:
        segment (int): Number of linear segments for the piecewise transformation.
    """    
    def __init__(self, segment, **kdwags):
        super(PWLinearModel, self).__init__()
        self.model_monotonic = PiecewiseLinear(segment)

    def forward(self, logits):
        """
        Forward pass through the model.

        Args:
            logits (Tensor): Input tensor of shape (batch_size, num_classes).

        Returns:
            Tensor: Transformed logits of the same shape.
        """
        # Normalize logits using log-softmax        
        logits = logits - torch.logsumexp(logits,dim=1,keepdim=True)

        # Flatten and transform
        flat_out = logits.contiguous().view(-1).unsqueeze(1)
        mono_out = self.model_monotonic(flat_out).contiguous().view(logits.shape)

        return mono_out


class PiecewiseLinear(nn.Module):
    """
    Implements a learnable, monotonic piecewise linear transformation over a fixed input range.

    This transformation is defined by dividing the input space into equal-width segments
    and assigning a separate learnable slope to each segment.

    Args:
        segments (int): Number of segments for the piecewise function.
    """ 
    def __init__(self, segments, device=None, dtype=None):
        super(PiecewiseLinear, self).__init__()
        
        self.logitrange = 100 # Defines the total range for input values
        self.delta = nn.parameter.Parameter(   data=torch.ones(1)/segments * self.logitrange,  requires_grad=False)
        
        # Predefine segment boundaries (non-learnable)
        steps = torch.linspace(start=0,end=self.logitrange,steps=segments+1)
        self.steps = nn.parameter.Parameter(   data=steps,  requires_grad=False)

        # Learnable slope parameters for each segment
        self.slops = nn.parameter.Parameter(   data=torch.ones(segments+1),  requires_grad=True)

    def forward(self, input):
        """
        Applies the piecewise linear function to the input tensor.

        Args:
            input (Tensor): Input tensor.
        
        Returns:
            Tensor: Transformed tensor.
        """        
        # Compute weighted sum over active slopes for each input location
        V1 = ( (-input >= self.steps[None,:]) * torch.abs(self.slops[None,:])  ).sum(1) - 1
        V1 = V1 * self.delta
        V2 = ( ( (-input+self.delta)  >= self.steps[None,:]) * torch.abs(self.slops[None,:])  ).sum(1)  - 1
        V2 = V2 * self.delta

        # Linear interpolation between V1 and V2
        Final = V1 + (V2 - V1)/self.delta * ( -input.flatten()  -  V1  )

        return -Final 


