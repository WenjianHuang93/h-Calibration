'''
Temperature Scaling Calibration Module
Author: Wenjian Huang
'''

import torch
import torch.nn as nn

class TempScaling(nn.Module):
    """
    A PyTorch module that applies temperature scaling to model logits for calibration purposes.

    Temperature scaling is a post-processing technique used to calibrate the confidence
    of classification models. This implementation learns the inverse of the temperature
    as a trainable parameter to maintain numerical stability and enforce positivity.

    Reference:
        Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    """    
    def __init__(self):
        """
        Initializes the temperature scaling module.

        The temperature is represented as the inverse of a learnable parameter to
        ensure that the scaling factor remains positive during training.
        """        
        super(TempScaling, self).__init__()
        self.temperature_inv = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        
    def forward(self, logits):
        """
        Applies temperature scaling to the input logits.

        Args:
            logits (Tensor): The unnormalized model outputs (logits), typically of shape (batch_size, num_classes).

        Returns:
            Tensor: Scaled logits with the same shape as input.
        """
        # Ensure temperature is positive by applying absolute value        
        return logits * torch.abs(self.temperature_inv)
    
