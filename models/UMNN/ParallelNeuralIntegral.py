"""
This module provides an optimized and simplified implementation of a monotonic transformation network,
leveraging numerical integration techniques for efficient computation of the following works

- https://github.com/AmirooR/IntraOrderPreservingCalibration
- https://github.com/AWehenkel/UMNN

Modified and refactored by: Wenjian Huang
"""

import torch
import numpy as np
import math


def compute_cc_weights_v1(nb_steps):
    """
    
    Rectified version by Wenjian Huang for efficiently computing
    the Clenshaw-Curtis weights and corresponding evaluation nodes.
    
    This method is based on numerical integration theory and references the following sources:
    [1] https://people.math.ethz.ch/~waldvoge/Papers/fejer.pdf
    [2] https://www.math.ucdavis.edu/~bremer/classes/fall2018/MAT128a/lecture19.pdf (p.7)
    [3] "On the Method for Numerical Integration of Clenshaw and Curtis"
    [4] "Methods of Numerical Integration" by Philip J. Davis et al. (p.86)

    Args:
        nb_steps (int): Number of discretization steps for the quadrature.

    Returns:
        cc_weights (torch.Tensor): Clenshaw-Curtis weights of shape (1,).
        steps (torch.Tensor): Evaluation nodes (cosine-spaced) of shape (nb_steps + 1, 1).
    """

    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W[-1] = W[-1] / 2
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate_v1(x0, nb_steps, step_sizes, integrand, cc_weights, steps):
    """
    Performs numerical integration using the Clenshaw-Curtis quadrature method.

    This function estimates the definite integral of a given function `integrand`
    over an interval defined by the input tensor `x0` and the number of steps.

    Args:
        x0 (torch.Tensor): Starting point of integration with shape (batch_size, input_dim).
        nb_steps (int): Number of Clenshaw-Curtis discretization steps.
        step_sizes (torch.Tensor): Step sizes for integration, same shape as `x0`.
        integrand (Callable): Function to be integrated, accepts input of shape (N, input_dim).
        cc_weights (torch.Tensor): Precomputed Clenshaw-Curtis weights.
        steps (torch.Tensor): Precomputed Clenshaw-Curtis evaluation nodes.

    Returns:
        torch.Tensor: Estimated integral for each input in the batch.
    """
    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps*step_sizes
    x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
    xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)

    steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
    X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
    X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])

    dzs = integrand(X_steps)
    dzs = dzs.view(xT_t.shape[0], nb_steps+1, -1)
    dzs = dzs*cc_weights.unsqueeze(0).expand(dzs.shape)
    z_est = dzs.sum(1)
    return z_est*(xT - x0)/2