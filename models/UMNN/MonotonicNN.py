"""
This module provides a computationally optimized and simplified implementation
of a monotonic transformation network, based on the original work:

- https://github.com/AmirooR/IntraOrderPreservingCalibration
- https://github.com/AWehenkel/UMNN

Modified and refactored by: Wenjian Huang
"""


import torch
import torch.nn as nn
from .ParallelNeuralIntegral import integrate_v1, compute_cc_weights_v1


class IntegrandNN_v1(nn.Module):
    """
    Defines a neural network to serve as the integrand for monotonic function construction.

    Args:
        in_d (int): Input dimensionality.
        hidden_layers (list): List containing the number of units in each hidden layer.
    """    
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN_v1, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        # Remove the last ReLU activation for the output layer
        self.net.pop()
        self.net = nn.Sequential(*self.net)
        self.act = nn.ELU()

    def forward(self, x):
        # Ensure positive integrand to preserve monotonicity
        return self.act(self.net(x)) + 1.0
    

class MonotonicNN_v1(nn.Module):
    """
    Monotonic Neural Network using numerical integration to ensure output monotonicity.

    Args:
        hidden_layers (list): List of hidden layer sizes for the integrand network.
        nb_steps (int): Number of discretization steps for integration (default: 50).
    """    
    def __init__(self, hidden_layers, nb_steps=50):
        super(MonotonicNN_v1, self).__init__()
        self.integrand = IntegrandNN_v1(1, hidden_layers)
        self.nb_steps = nb_steps
        self.cc_weights, self.steps = compute_cc_weights_v1(nb_steps)
        self.norm = nn.BatchNorm1d(1)
        self.scaling = nn.parameter.Parameter(   data=torch.Tensor([1.0]),  requires_grad=True)


    def forward(self, x):
        """
        Computes a monotonic transformation of input `x` via numerical integration.
        
        Args:
            x (Tensor): 1D tensor input.
        
        Returns:
            Tensor: Monotonically increasing transformation of `x`.
        """
        x0 = torch.zeros(x.shape).to(x.device)
        
        integrated = integrate_v1(x0, self.nb_steps, (x - x0)/self.nb_steps, self.integrand, self.cc_weights, self.steps)
        
        return integrated








