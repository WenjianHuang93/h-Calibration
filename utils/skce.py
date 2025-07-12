"""
This module is a Python reimplementation of the SKCE (Squared Kernel Calibration Error)
metric originally developed in Julia, based on the repository:

    https://github.com/devmotion/pycalibration

The current implementation uses the default kernel combination (CRBF & CWhiteKernel) and has 
been tested to match the results of the original Julia version with negligible numerical difference.

If you prefer not to install the Julia-based package, you can directly replace the `_SKCE` class 
in `utils/utils.py` file with this implementation. It provides an equivalent 
interface and output.

Author: Hui Wang, Wenjian Huang
Affiliation: SUSTech
"""


from scipy.spatial.distance import pdist, squareform
from torch import nn
import numpy as np
from torch.nn import functional as F


# Abstract base class for a kernel function
class Kernel:
    def __call__(self, x, y):
        raise NotImplementedError("Kernel computation must be implemented.")

# Main SKCE module
class _SKCE(nn.Module):
  
  # RBF (Radial Basis Function) kernel
  class CRBF(Kernel):
      def __init__(self, scale = 1.0):
          self.scale = scale

      def __call__(self, predictions,targets):        
          # Compute pairwise Euclidean distances
          dists = squareform(pdist(predictions, metric='euclidean'))
          # Apply the RBF kernel
          return np.exp(-dists / self.scale)
  
  # Constant white noise kernel (always returns 0)
  class CWhiteKernel(Kernel):
      def __init__(self, noise_level=1.0):
          self.noise_level = noise_level

      def __call__(self, g_X_i, g_X_j):
          return 0.0

  # Tensor product of two kernels (currently uses only the first)
  class KernelTensorProduct(Kernel):
      def __init__(self, kernel1: Kernel, kernel2: Kernel):
          self.kernel1 = kernel1
          self.kernel2 = kernel2
      # Combine kernels if needed; currently only uses kernel1
      def __call__(self, predictions, targets):
          return self.kernel1(predictions, targets)
      
  # Core SKCE implementation
  class SKCE:
      def __init__(self, kernel: Kernel, unbiased: bool = True, blocksize: int = None):
          self.kernel = kernel
          self.unbiased = unbiased
          self.blocksize = blocksize

      def __call__(self, predictions: np.ndarray, targets: np.ndarray):
          # Choose block estimation or standard estimation
          if self.blocksize is not None and (isinstance(self.blocksize, int) or callable(self.blocksize)):
              return self.block_estimate(predictions, targets)

          if self.unbiased:
              return self.unbiasedskce(predictions, targets)
          else:
              return self.biasedskce(predictions, targets)

      def biasedskce(self, predictions: np.ndarray, targets: np.ndarray):
          n = self.check_nsamples(predictions, targets, 1)
          
          # One-hot encode targets (assuming 1-based labels)
          one_hot_targets = np.eye(predictions.shape[1])[targets - 1]  # targets 从 1 开始
          kernel_matrix = self.kernel(predictions,targets)
          
          # Compute pairwise inner product (PIP) matrix
          term1 = (targets[:, None] == targets[None, :]).astype(float)
          term2 = np.dot(one_hot_targets, predictions.T)
          term3 = np.dot(predictions, one_hot_targets.T)
          term4 = np.dot(predictions, predictions.T)

          pip_matrix = term1 - term2 - term3 + term4

          # Element-wise product with the kernel matrix
          composite_kernel_matrix = kernel_matrix * pip_matrix

          return np.mean(composite_kernel_matrix)

      def unbiasedskce(self, predictions: np.ndarray, targets: np.ndarray):
          n = self.check_nsamples(predictions, targets, 2)

          one_hot_targets = np.eye(predictions.shape[1])[targets - 1]  # targets 从 1 开始
          kernel_matrix = self.kernel(predictions,targets)

          term1 = (targets[:, None] == targets[None, :]).astype(float)
          term2 = np.dot(one_hot_targets, predictions.T)
          term3 = np.dot(predictions, one_hot_targets.T)
          term4 = np.dot(predictions, predictions.T)

          pip_matrix = term1 - term2 - term3 + term4
          composite_kernel_matrix = kernel_matrix * pip_matrix
          
          # Extract upper triangle (excluding diagonal) for unbiased estimate
          triu_indices = np.triu_indices(composite_kernel_matrix.shape[0], k=1)
          upper_triangle_values = composite_kernel_matrix[triu_indices]

          return np.mean(upper_triangle_values)

      @staticmethod
      def check_nsamples(predictions: np.ndarray, targets: np.ndarray, min_samples: int = 1) -> int:
          # Sanity checks for sample size and shape
          n = predictions.shape[0]
          if targets.shape[0] != n:
              raise ValueError(f"Number of predictions ({n}) and targets ({targets.shape[0]}) must match.")
          if n < min_samples:
              raise ValueError(f"At least {min_samples} samples are required, but got {n}.")
          return n

  def __init__(self, scale=1.0, unbiased=True):
    super(_SKCE, self).__init__()
    # Define kernel as tensor product of CRBF and White kernel
    kernel = self.KernelTensorProduct(self.CRBF(scale), self.CWhiteKernel())
    # Initialize SKCE estimator
    self.skce = self.SKCE(kernel, unbiased=unbiased)

  def forward(self, logits, labels):
    # Convert logits to probabilities
    softmaxes = F.softmax(logits, dim=1)
    predictions = softmaxes.detach().cpu().numpy().astype("float64")
    # Flatten labels and convert to 1-based class index
    outcomes = labels.flatten().detach().cpu().numpy().astype("int64") + 1
    # Compute SKCE value
    skce_value = self.skce(predictions, outcomes)
    return skce_value
  