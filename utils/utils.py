"""
This module contains utility functions and evaluation metrics commonly used in deep learning experiments,
including:

- Logging setup and configuration
- Model saving and loading (checkpoints)
- Random seed fixing for reproducibility
- Early stopping mechanism
- Running average calculator
- A comprehensive collection of calibration and evaluation metrics (ECE variants, Brier score, SKCE, KDE-based metrics, etc.)

The calibration metric implementations are adapted from existing open-source projects (see comment in corresponding script), 
but have been thoroughly optimized for improved computational efficiency, cleaner structure, and ease of integration—without affecting their correctness.

Author: Wenjian Huang
"""

import json
import logging
import os
import numpy as np
import torch
from datetime import datetime
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import pdist, squareform

from scipy.stats import binned_statistic_dd

import abc
import scipy.cluster.vq
import scipy
import os

from concurrent.futures import ThreadPoolExecutor

try:
    from KDEpy import FFTKDE
except ImportError:
    Warning('kdepy==1.1.0 not installed; only relevant for KDE CE')


_RNG_SEED = None



def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed.
    Args:
        seed (int):
    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.
    Example:
        Fix random seed in both tensorpack and tensorflow.
    .. code-block:: python
            import utils
            seed = 42
            utils.fix_rng_seed(seed)
            torch.manual_seed(seed)
            if config.cuda: torch.cuda.manual_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


class RunningAverage():
    """
    Maintains the running average of a scalar quantity.

    Example:
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        print(loss_avg())  # Outputs: 3.0
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path, append=False):
    """
    Save a dictionary of numeric values to a JSON file.

    Args:
        d (dict): Dictionary containing float-castable values.
        json_path (str): Path to the JSON output file.
        append (bool): If True, append to existing file.
    """
    def to_float(d):
      for k,v in d.items():
        if type(v) is dict:
          d[k] = to_float(v)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
          d[k] = float(v)
        elif isinstance(v, list):
          d[k] = [float(x) for x in v]
      return d
    
    if append==False:
      with open(json_path, 'w') as f:
          # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
          d = to_float(d) #{k: float(v) for k, v in d.items()}
          json.dump(d, f, indent=2)
    elif append==True:
      if os.path.exists(json_path):
        with open(json_path,'r') as f:
          ori = json.load(f)
          ori.update(d)
      else:
        ori = d
      with open(json_path, 'w') as f:
          # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
          ori = to_float(ori) #{k: float(v) for k, v in d.items()}
          json.dump(ori, f, indent=2)


def save_checkpoint(state, savepath, name=None):
    """
    Save model state and other training parameters to disk.

    Args:
        state (dict): Contains model state_dict, optimizer state, epoch, etc.
        savepath (str): Directory to save the checkpoint.
        name (str, optional): Name suffix for the checkpoint file.
    """
    filepath = os.path.join(savepath, 'selector_{}.pth.tar'.format(name))
    if not os.path.exists(savepath):
        print("Checkpoint Directory does not exist! Making directory {}".format(savepath))
        os.mkdir(savepath)
        
    torch.save(state, filepath)


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Load model and optionally optimizer states from a checkpoint file.

    Args:
        checkpoint (str): Path to the checkpoint file.
        model (nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.

    Returns:
        dict: The loaded checkpoint.
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint,map_location ='cpu')
    
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        # remove "_orig_mod." prefix
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value    

    model.load_state_dict(new_state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}, current_best {}'.format(self.counter, self.patience, -self.best_score))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, self.val_loss_min))
        # torch.save(model.state_dict(), 'checkpoint.pt') # do not have to save model in overall project path
        self.val_loss_min = val_loss


class _ECELoss(nn.Module):
  """
  Calculates the Expected Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs into equally-sized interval bins.
  In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

  We then return a weighted average of the gaps, based on the number
  of samples in each bin

  See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
  "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
  2015.
  """
  def __init__(self, n_bins=15, save_bins=False, save_path=None):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_ECELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]
    self.save_bins = save_bins
    self.save_path = save_path

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    if self.save_bins:
      bin_data = {'bin_lowers': [], 'bin_uppers': [], 'props': [], 'accs': [], 'confs': []}
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        if self.save_bins:
          bin_data['bin_lowers'].append(bin_lower.item())
          bin_data['bin_uppers'].append(bin_upper.item())
          bin_data['props'].append(prop_in_bin.item())
          bin_data['accs'].append(accuracy_in_bin.item())
          bin_data['confs'].append(avg_confidence_in_bin.item())
    if self.save_bins:
      save_dict_to_json(bin_data, self.save_path)
    return ece
  

class _AECELoss(nn.Module):
  """
  Computes the Average Expected Calibration Error (ACE).

  This metric estimates calibration error using fixed-width bins, while dynamically ignoring bins 
  with insufficient sample count. It is adapted and simplified from the NetCal library.

  Args:
      n_bins (int): Number of bins used to partition the confidence interval [0, 1].
      sample_threshold (int): Minimum number of samples required in a bin to include it in the computation.
  """  
  def __init__(self, n_bins=15, sample_threshold = 1):
     super().__init__()
     self.sample_threshold = sample_threshold
     self.bins = n_bins

  def prepare(self,X,y):
      """
      Prepares the input data by:
      - Clipping probabilities for numerical stability
      - Extracting top-1 confidence scores and correctness indicators
      - Generating bin boundaries for histogram-based binning

      Args:
          X (np.ndarray): Probability predictions (N, C), after softmax.
          y (np.ndarray): True labels (N,)

      Returns:
          tuple: (confidence values, correctness indicators, bin boundaries)
      """      
      epsilon = np.finfo(X.dtype).eps
      X = np.clip(X, epsilon, 1. - epsilon)
      matched = np.argmax(X, axis=1) == y
      X = np.max(X, axis=1, keepdims=False)
      bin_bounds = [np.linspace(0.0, 1.0, self.bins + 1)]
      return X, matched, bin_bounds

  def binning(self, bin_bounds, samples, *values):
      """
      Bins the given values using fixed bin boundaries, computing mean statistics per bin.
      Ignores bins with fewer samples than the threshold.

      Args:
          bin_bounds (list): List of bin edge arrays for each dimension.
          samples (np.ndarray): Input values used for bin assignment.
          *values (np.ndarray): Values to compute statistics for (e.g., accuracy, confidence).

      Returns:
          tuple:
              - List of histograms (one per input value)
              - Histogram of sample counts per bin
      """
      num_samples_hist, _ = np.histogramdd(samples, bins=bin_bounds)
      binning_schemes = []

      for val in values:
          hist, _, _ = binned_statistic_dd(samples, val, statistic='mean', bins=bin_bounds, binned_statistic_result=None)
          hist[num_samples_hist < self.sample_threshold] = np.nan
          hist = np.nan_to_num(hist, nan=0.0)
          binning_schemes.append(hist)
      return tuple(binning_schemes), num_samples_hist

  def process(self,acc_hist,conf_hist,num_samples_hist):
      """
      Calculates the final ACE value based on binned statistics.

      Args:
          acc_hist (np.ndarray): Bin-wise average accuracy.
          conf_hist (np.ndarray): Bin-wise average confidence.
          num_samples_hist (np.ndarray): Number of samples in each bin.

      Returns:
          float: The scalar AECE value.
      """      
      deviation_map = np.abs(acc_hist - conf_hist)
      reduced_deviation_map = np.sum(deviation_map, axis=0)
      non_empty_bins = np.count_nonzero(num_samples_hist, axis=0)
      bin_map = np.divide(
          reduced_deviation_map, non_empty_bins,
          out=np.zeros_like(reduced_deviation_map),
          where=non_empty_bins != 0)
      miscalibration = np.sum(bin_map / np.count_nonzero(np.sum(num_samples_hist, axis=0)))
      return miscalibration
  
  def measure(self,X, y):
      """
      Full ACE computation pipeline from softmax outputs and labels.

      Args:
          X (np.ndarray): Softmax probabilities (N, C).
          y (np.ndarray): True labels (N,)

      Returns:
          float: AECE score.
      """      
      batch_X, batch_matched, bounds = self.prepare(X, y)
      histograms, num_samples_hist = self.binning(bounds, batch_X, batch_matched, batch_X)
      miscalibration = self.process(*histograms, num_samples_hist=num_samples_hist)
      return float(miscalibration)

  def forward(self, logits, labels):
      """
      PyTorch forward function.

      Args:
          logits (torch.Tensor): Logits output from model (N, C).
          labels (torch.Tensor): Ground truth labels (N,)

      Returns:
          torch.Tensor: Scalar AECE value as a float.
      """  
      return self.measure(nn.functional.softmax(logits, dim=1).detach().cpu().numpy(), labels.cpu().numpy())


class _DebiasedECELoss(nn.Module):
  """
  Computes calibration error with optional bias correction, marginalization, and adaptive binning.

  This orginal code is from:
  https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py
  as presented in the pape "Verified Uncertainty Calibration"
  I optimized the code for faster efficient calculation.

  It corresponds to several calibration error variants described in "Measuring Calibration in Deep Learning",
  Specific correspondences:
  - Static Calibration Error → debiased=False, marginal=True, adaptive=False
  - Adaptive Calibration Error → debiased=False, marginal=True, adaptive=True
    (Note: Adaptive Calibration Error sounds like `_AdaptiveECELoss` but is different, not to be confused)

  Args:
      n_bins (int): Number of bins used in the binning procedure.
      debiased (bool): Whether to apply bias correction using Gaussian resampling.
      marginal (bool): If True, compute marginal calibration error across all classes.
      adaptive (bool): If True, use equal-mass binning; otherwise use equal-width.
      power (float): Lp-norm used in calibration error (e.g., 1 or 2).
      nclasses (int): Number of classes (used in marginal mode).
  """  
  
  def __init__(self, n_bins=15, debiased = True, marginal=False, adaptive=True, power = 1.0, nclasses=1000):
     super(_DebiasedECELoss,self).__init__()
     self.n_bins = n_bins
     self.adaptive = adaptive
     self.marginal = marginal
     self.debiased = debiased
     self.power = power
     self.FasterApproximator = False   
  
  @staticmethod
  def difference_mean(data) -> float:
      """Returns average pred_prob - average label."""
      data = np.array(data)
      ave_pred_prob = np.mean(data[:, 0])
      ave_label = np.mean(data[:, 1])
      return ave_pred_prob - ave_label
  
  @staticmethod
  def get_bin_probs(binned_data):
      bin_sizes = list(map(len, binned_data))
      num_data = sum(bin_sizes)
      bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
      eps = 1e-6
      assert(abs(sum(bin_probs) - 1.0) < eps)
      return list(bin_probs)
    
  @staticmethod
  def fast_bin(data, bins):
      prob_label = np.array(data)
      bin_indices = np.searchsorted(bins, prob_label[:, 0]) # 每个prob属于哪个bin
      bin_sort_indices = np.argsort(bin_indices) 
      sorted_bins = bin_indices[bin_sort_indices] 
      splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
      binned_data = np.split(prob_label[bin_sort_indices], splits)
      return binned_data
  
  @staticmethod
  def plugin_ce(binned_data, power=2) -> float:
      """
      Computes the standard (plugin) calibration error using empirical bin statistics.

      Args:
          binned_data (list of arrays): Each element is a bin containing (prob, label) pairs.
          power (int): The power used for the Lp-norm.

      Returns:
          float: Estimated calibration error.
      """      
      def bin_error(data):
          if len(data) == 0:
              return 0.0
          return abs(_DebiasedECELoss.difference_mean(data)) ** power
      bin_probs = _DebiasedECELoss.get_bin_probs(binned_data)
      bin_errors = list(map(bin_error, binned_data))
      return np.dot(bin_probs, bin_errors) ** (1.0 / power)
  
  @staticmethod
  def plugin_ce_numpy(binned_data, power=2) -> float:
      return ( np.abs(binned_data[:,:,0].mean(1) - binned_data[:,:,1].mean(1)) ** power ).mean(0) ** (1.0/power)
    
  @staticmethod
  def split(sequence, parts: int):
      assert parts <= len(sequence)
      array_splits = np.array_split(sequence, parts)
      splits = [list(l) for l in array_splits]
      assert len(splits) == parts
      return splits

  @staticmethod
  def normal_debiased_ce(binned_data, power=1, resamples=1000, num_bins=10) -> float:
      """
      Compute the bias-corrected calibration error using a normal approximation.

      - When `power=1`, a closed-form approximation of the absolute moment of a Gaussian
        is used to avoid costly sampling.
      - When `power!=1`, Gaussian resampling is used with Monte Carlo estimation.

      The original implementation is slow; this version is rewritten for efficiency
      using vectorized NumPy and closed-form expressions wherever possible.

      Args:
          binned_data (List[np.ndarray]): Each element is a bin of shape (N_bin, 2), containing (prob, label) pairs.
          power (int): The Lp norm to use (typically 1 or 2).
          resamples (int): Number of samples used for Gaussian resampling when power != 1.
          num_bins (int): Number of bins used for the computation.

      Returns:
          float: Bias-corrected calibration error.
      """

      bin_sizes = np.array(list(map(len, binned_data)))
      if np.min(bin_sizes) <= 1:
          raise ValueError('Every bin must have at least 2 points for debiased estimator. '
                          'Try adding the argument debias=False to your function call.')
      
      label_means = np.array(list(map(lambda l: np.mean(l[:,1]), binned_data)))
      label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
      
      model_vals = np.array(list(map(lambda l: np.mean(l[:,0]), binned_data)))

      assert(label_means.shape == (len(binned_data),))
      assert(model_vals.shape == (len(binned_data),))
      ce = _DebiasedECELoss.plugin_ce(binned_data, power=power)
      bin_probs = _DebiasedECELoss.get_bin_probs(binned_data)
    
      # the orginal source code is too slow. the following is the rewritten fast version.
      if power==1:
        # Use closed-form approximation of the absolute moment of a Gaussian
        distributed_mean = label_means - model_vals
        distributed_std = label_stddev
        
        # Compute expectation of absolute deviation for Gaussian
        nonzeroid = ~np.equal(distributed_std, 0.0)
        diffs = np.abs(label_means - model_vals)        
        diffs[nonzeroid] = distributed_std[nonzeroid] * 2**(1/2) * (1/np.sqrt(np.pi)) * scipy.special.hyp1f1(-1/2,1/2,- distributed_mean[nonzeroid]**2/(2*distributed_std[nonzeroid]**2))
        mean_resampled = diffs.mean()
        # Note: I use this closed-form replaces the original, much slower sampling-based method.
      else:
        # Gaussian resampling when power != 1
        label_samples = np.random.normal(loc=np.tile(label_means,(resamples,1)), scale=np.tile(label_stddev,(resamples,1))) # shape: (resamples, num_bins)
        diffs = np.power(np.abs(label_samples - model_vals[None,:]), power)
        cur_ce = np.power( diffs @ np.array(bin_probs), 1.0 / power)
        resampled_ces = cur_ce
        mean_resampled = np.mean(resampled_ces)            
      
      # Apply bias correction
      bias_corrected_ce = 2 * ce - mean_resampled
      
      return bias_corrected_ce
  

  @staticmethod
  def normal_debiased_ce_numpy(binned_data, power=1, resamples=1000, num_bins=10) -> float:
      bin_sizes = binned_data.shape[1]
      label_means = binned_data[:,:,1].mean(1)
      label_stddev = np.sqrt(label_means * (1 - label_means) / bin_sizes)
      model_vals = binned_data[:,:,0].mean(1)

      ce = _DebiasedECELoss.plugin_ce_numpy(binned_data, power=power)

      if power==1:
        distributed_mean = label_means - model_vals
        distributed_std = label_stddev
        # Compute expectation of absolute deviation for Gaussian
        nonzeroid = ~np.equal(distributed_std, 0.0)
        diffs = np.abs(label_means - model_vals)        
        diffs[nonzeroid] = distributed_std[nonzeroid] * 2**(1/2) * (1/np.sqrt(np.pi)) * scipy.special.hyp1f1(-1/2,1/2,- distributed_mean[nonzeroid]**2/(2*distributed_std[nonzeroid]**2))
        mean_resampled = diffs.mean()
        # this is the re-written speed up version for the following code when power=1 based on the properties of gaussian distribution
      else:
        label_samples = np.random.normal(loc=np.tile(label_means,(resamples,1)), scale=np.tile(label_stddev,(resamples,1)))
        diffs = np.power(np.abs(label_samples - model_vals[None,:]), power)
        cur_ce = np.power( diffs.mean(), 1.0 / power)
        resampled_ces = cur_ce
        mean_resampled = np.mean(resampled_ces)           

      # Apply bias correction 
      bias_corrected_ce = 2 * ce - mean_resampled
      
      return bias_corrected_ce

  @staticmethod
  def ce_1d(probs, labels, adaptive, power, debiased, n_bins, FasterApproximator):
    assert probs.shape == labels.shape
    assert len(probs.shape) == 1
    
    if adaptive == True:
        if FasterApproximator:
        # use more faster estimator by discarding a few samples for equal-size bins
          Sortind = np.argsort(probs)
          probs, labels = probs[Sortind], labels[Sortind]
          IgnoreN = len(probs)-int(len(probs)/n_bins)*n_bins
          probs, labels = probs[IgnoreN:], labels[IgnoreN:]
          probs, labels = probs.reshape(n_bins,int(len(probs)/n_bins)), labels.reshape(n_bins,int(len(probs)/n_bins))
          data_binned = np.concatenate((probs[...,None], labels[...,None]),axis=-1)

        else:
          data = list(zip(probs, labels))
          Sortind = np.argsort(probs)
          Boundary = [np.int(np.round( len(probs)/n_bins * i)) for i in range(n_bins+1)]
          data_binned = [np.array(data)[Sortind[Boundary[i]:Boundary[i+1]]] for i in range(n_bins) ]

    elif adaptive == False:
        data = list(zip(probs, labels))
        bins = list(np.linspace(0,1.0,n_bins+1)[1:])
        data_binned = _DebiasedECELoss.fast_bin(data, bins)

    p = power
    if debiased == True:
      if FasterApproximator:
        return _DebiasedECELoss.normal_debiased_ce_numpy(data_binned, power=p, num_bins=n_bins)         
      else:
        bin_sizes = np.array(list(map(len, data_binned)))
        if np.min(bin_sizes) <= 1:
            raise ValueError('Every bin must have at least 2 points for debiased estimator. ')
        return _DebiasedECELoss.normal_debiased_ce(data_binned, power=p, num_bins=n_bins)
    
    elif debiased == False:
      if FasterApproximator:
          return _DebiasedECELoss.plugin_ce_numpy(data_binned, power=p)
      else:
          return _DebiasedECELoss.plugin_ce(data_binned, power=p)


  def get_labels_one_hot(self, labels, k):
      assert np.min(labels) >= 0
      assert np.max(labels) <= k - 1
      num_labels = labels.shape[0]
      labels_one_hot = np.zeros((num_labels, k))
      labels_one_hot[np.arange(num_labels), labels] = 1
      return labels_one_hot

  def forward(self, logits, labels):
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    if self.marginal:

      labels_one_hot = self.get_labels_one_hot(labels, k=probs.shape[1])
      assert probs.shape == labels_one_hot.shape
      
      # the orginal source code is too slow and let us speed it up. the following is the rewritten version.
      with ThreadPoolExecutor(max_workers=None) as executor:
          futures = [executor.submit(self.ce_1d, probs[:, class_idx], labels_one_hot[:, class_idx], self.adaptive, self.power, self.debiased, self.n_bins, self.FasterApproximator) for class_idx in range(probs.shape[1])]
          marginal_ces = [future.result() for future in futures]
      error = np.mean(marginal_ces)
    
    else:
      if np.min(labels) < 0 or np.max(labels) > probs.shape[1] - 1:
          raise ValueError('labels should be between 0 and num_classes - 1.')

      preds = np.argmax(probs,axis=1)
      correct = (preds == labels).astype(probs.dtype)
      confidences = np.max(probs,axis=1)
      error = self.ce_1d(confidences, correct, self.adaptive, self.power, self.debiased, self.n_bins, self.FasterApproximator)
    
    return error


class _AdaptiveECELoss(nn.Module):
    """
    Computes Adaptive Expected Calibration Error (ECE), as noted in:
    "Calibrating Deep Neural Networks using Focal Loss" 

    This metric partitions confidence scores into bins containing equal numbers of samples
    (adaptive/equal-mass binning), and computes the weighted average of absolute 
    differences between confidence and accuracy in each bin.

    Reference implementation adapted from:
    https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py

    Args:
        n_bins (int): Number of adaptive bins used to estimate calibration error.
    """    
    def __init__(self, n_bins=15):
        super(_AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class _CwECELoss(nn.Module):
  """
  Calculates the Class-wise Expected Calibration Error of a model.

  The input to this loss is the logits of a model, NOT the softmax scores.

  This divides the confidence outputs of each class j into equally-sized
  interval bins. In each bin, we compute the confidence gap:

  bin_gap = | avg_confidence_in_bin_j - accuracy_in_bin_j |

  We then return a weighted average of the gaps, based on the number
  of samples in each bin
  """
  def __init__(self, n_bins=15, avg=True):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_CwECELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]
    self.avg = avg

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    cw_ece = torch.zeros(1, device=logits.device)
    for j in range(num_classes):
      confidences_j = softmaxes[:,j]
      ece_j = torch.zeros(1, device=logits.device)
      for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
        in_bin = confidences_j.gt(bin_lower.item()) * confidences_j.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
          accuracy_j_in_bin = labels[in_bin].eq(j).float().mean()
          avg_confidence_j_in_bin = confidences_j[in_bin].mean()
          ece_j += torch.abs(avg_confidence_j_in_bin - accuracy_j_in_bin) * prop_in_bin
      cw_ece += ece_j
    if self.avg:
      return cw_ece/num_classes
    else:
      return cw_ece


class _BrierLoss(nn.Module):
  """
  Computes the Brier Score for probabilistic classification.

  The Brier Score is defined as the mean squared error between the predicted 
  probability distribution (after softmax) and the one-hot encoded ground truth labels.

  This implementation operates on logits (before softmax), and internally applies softmax.

  Reference:
  - The Brier Score formulation used here is consistent with that in
    "Better Uncertainty Calibration via Proper Scores for Classification and Beyond",
    where the Root Brier Score (RBS) is defined as simply the square root of the Brier Score.
  """
  
  def __init__(self):
    super(_BrierLoss, self).__init__()

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    labels_onehot = torch.zeros_like(softmaxes)
    labels_onehot.scatter_(1, labels[...,None], 1)
    diff = (labels_onehot - softmaxes)
    diff = diff*diff
    return diff.sum(1).mean()


class _sKDECE(nn.Module):
  """
  Computes Kernel Density Estimate-based Calibration Error (sKDECE) for top-label confidence.

  This implementation estimates ECE using a continuous confidence space via kernel density estimation (KDE),
  instead of traditional histogram binning. It measures the expected deviation between confidence and accuracy
  across the entire prediction distribution.

  Original implementation from:
  - "Better Uncertainty Calibration via Proper Scores for Classification and Beyond"
    https://github.com/MLO-lab/better_uncertainty_calibration/blob/75bed1b5b3fa41b1e298df11f4c6c23c71180860/errors.py
  - Further adopted from:
    https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/util_evaluation.py

  Notes:
  - This version includes several computational optimizations for speed and efficiency compared to the above original sources.
  - Focuses on top-label confidence error estimation.

  Args:
      order (float): The L^p norm order (default: 2.0).
      p_int (np.ndarray or None): Optional integration sample set. If None, will use predictions.
  """

  def __init__(self,order=2.0,p_int=None):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_sKDECE, self).__init__()
    self.order = order
    self.p_int = p_int

  def mirror_1d(self, d, xmin=None, xmax=None):
      """
      Applies reflection-based boundary conditions to avoid KDE bias near edges (0 or 1).

      This ensures the KDE mass does not fall outside the [0,1] interval.

      Args:
          d (np.ndarray): The 1D data array to mirror.
          xmin (float): Lower bound of domain.
          xmax (float): Upper bound of domain.

      Returns:
          np.ndarray: Mirrored data array.
      """      
      if xmin is not None and xmax is not None:
          xmed = (xmin+xmax)/2
          return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
      elif xmin is not None:
          return np.concatenate((2*xmin-d, d))
      elif xmax is not None:
          return np.concatenate((d, 2*xmax-d))
      else:
          return d
  

  def forward(self, logits, labels):
    """
    Computes the KDE-based Expected Calibration Error (ECE) for top-label confidence.

    Key steps:
    - Applies softmax and binarizes labels.
    - Uses KDE (with triweight kernel) to estimate the confidence density of correct and all predictions.
    - Mirrors confidence values at boundaries [0, 1] to avoid KDE edge artifacts.
    - Uses a ratio of densities to compute estimated accuracy at each confidence level.
    - Computes the weighted average L^p error between predicted confidence and estimated accuracy.

    Args:
        logits (Tensor): Model outputs (N, C).
        labels (Tensor): Ground-truth class labels (N,).

    Returns:
        float: KDE-based calibration error.
    """    
    order = self.order
    p_int = self.p_int    

    p = torch.softmax(logits, dim=1)
    p, labels = p.detach().cpu().numpy(),labels.detach().cpu().numpy(),

    # Convert labels to one-hot
    label = label_binarize(np.array(labels), classes=range(p.shape[1]))

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)
    
    # Clip for numerical stability
    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)

    x_int = np.linspace(-0.6, 1.6, num=2**14)

    N = p.shape[0]
    
    # Compute top-label correctness and confidence
    label_index = np.argmax(label,axis=1)
    with torch.no_grad():
        if p.shape[1] !=2:
            # source code is too slow so we accelerate it by rewriting the code
            pred_label = np.argmax(p,axis=1)
            label_binary = (pred_label == label_index)[:,None].astype('float64')
            p_b = (np.max(p,axis=1)/np.sum(p,axis=1))[:,None]
        else:
            p_b = (p/np.sum(p,1)[:,None])[:,1]
            label_binary = label_index

    method = 'triweight'
    
    # KDE on correct prediction confidences
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1))
    kbw = np.std(dconf_1)*(N*2)**-0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = self.mirror_1d(dconf_1,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    # pp1: KDE on correct prediction confidences

    p_int = p_int/np.sum(p_int,1)[:,None]

    # the original source code is too slow to run so we accelerate it by rewriting the code
    if p_int.shape[1]!=2:
       pred_b_int = np.max(p_int,axis=1)[:,None]
    else:
       pred_b_int = p_int[:,1][:,None]
    
    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = self.mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1
    # pp2: KDE on all sample maximum predicted probabilities

    # Estimate reliability and integral
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)

    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    
    # the original source code is too slow to run so we accelerate it by rewriting the code
    Indicator = np.maximum(pp1,pp2)>1e-6

    accu = np.minimum(perc*pp1[Indicator]/pp2[Indicator],1.0)
    integral[Indicator] = (np.abs(x_int[Indicator]-accu)**order*pp2[Indicator])
    integral[np.isnan(integral)] = 0.0

    reliability[Indicator] = accu
    reliability[np.isnan(reliability)] = 0.0
    
    # Fill missing values using previous valid ones
    Indicator[0] = True
    index_in_Indicator = np.cumsum(Indicator)[~Indicator] - 1
    integral[~Indicator]  = integral[Indicator][index_in_Indicator]
    Indicator[0] = False


    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])



class _SKCE(nn.Module):
  """
  Computes the Scaled Kernel Calibration Error (SKCE).

  SKCE is a kernel-based calibration metric introduced for evaluating the 
  quality of predictive uncertainty in probabilistic classifiers.

  This implementation uses the `pycalibration` package and requires a working Julia backend.
  The kernel used is a product of an Exponential Kernel and a White Kernel.

  NOTE:
  - See the "README" section of the project for installation and setup details,
    including proper Julia environment and pycalibration bindings.

  Reference:
  - Widmann et al., "Calibration tests in multi-class classification: A unifying framework"
  """
  def __init__(self):
    """
    Initializes the SKCE kernel metric using the `pycalibration` Julia wrapper.
    """
    super(_SKCE, self).__init__()

    from julia import Julia
    jl = Julia(compiled_modules=False)
    from pycalibration import ca    
    self.skce = ca.SKCE(ca.tensor(ca.ExponentialKernel(), ca.WhiteKernel()))

  def forward(self, logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    softmaxes = softmaxes.detach().to('cpu').numpy()
    predictions = [ softmaxes[sample,:].astype('float64') for sample in range(softmaxes.shape[0])]
    outcomes = labels.flatten().detach().to('cpu').numpy() + 1
    SKCE = self.skce(predictions, outcomes)

    return SKCE
  

class _ECEKDE(nn.Module):
  """
  Computes the kernel-based estimator for the Canonical Calibration Error (ECE-KDE).

  This estimator is adopted from:
  - "A Consistent and Differentiable Lp Canonical Calibration Error Estimator"
    https://github.com/tpopordanoska/ece-kde/blob/main/ece_kde.py

  Key Notes:
  - Unlike top-label KDE methods (e.g., _sKDECE), this method estimates calibration error
    in the full multiclass probability simplex.
  - Original implementation has been **modified and accelerated** here for efficiency.

  Args:
      order (float): Lp norm order, typically 1 or 2.
      bandwidths (float or Tensor): Kernel bandwidth(s) used for Dirichlet KDE.
  """
  def __init__(self,order=2.0, bandwidths = None):
    """
    n_bins (int): number of confidence interval bins
    """
    super(_ECEKDE, self).__init__()
    self.order = order 
    self.banwidths=bandwidths
    self.UseCPU = False
   

  def dirichlet_kernel(self, z, bandwidth=0.1):
      """
      Computes the log Dirichlet kernel matrix (slow version for reference).

      Args:
          z (Tensor): Input probabilities, shape (N, C)
          bandwidth (float): Bandwidth for smoothing

      Returns:
          Tensor: Log-density matrix of shape (N, N)
      """      
      alphas = z / bandwidth + 1 # of shape samples * probs
      log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
      log_num = torch.matmul(torch.log(z), (alphas-1).T) # (samples * classes)  matmul (classes * samples)
      log_dir_pdf = log_num - log_beta # of shape sampels * samples
      return log_dir_pdf # log value of the kernel
  
  
  def dirichlet_kernels(self, z, bandwidths):
      """
      Vectorized and accelerated computation of Dirichlet kernel log-densities (modified speed up version).

      Args:
          z (Tensor): Input probabilities, shape (N, C)
          bandwidths (Tensor): Bandwidths, shape (B,)

      Returns:
          Tensor: Log-kernel matrix, shape (N, N, B)
      """      
      alphas = z[:,:,None] / bandwidths[None,None,:] + 1 # of shape samples * classes * bandwidths
      log_num = torch.einsum('ij,jkl->ikl',torch.log(z),(alphas-1).permute(1,0,2)) # of shape samples * samples * bandwidths

      log_beta = (torch.sum(torch.lgamma(alphas), dim=1) - torch.lgamma(torch.sum(alphas, dim=1))) # of shape samples * bandwidths
      log_dir_pdf = log_num - log_beta[None,:,:]
      return log_dir_pdf  

  def check_input(self, f, bandwidth, mc_type):
      assert not torch.any(torch.isnan(f))
      assert len(f.shape) == 2
      assert bandwidth > 0
      assert torch.min(f) >= 0
      assert torch.max(f) <= 1


  def get_kernel(self, f, bandwidth, device):
      # if num_classes == 1
      if f.shape[1] == 1:
          raise NotImplementedError("noncanonical case ignored")
      else:
          log_kern = self.dirichlet_kernel(f, bandwidth).squeeze()
      # Trick: -inf on the diagonal
      return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)

  def get_kernel_efficient(self, f, bandwidth, device):
      # if num_classes == 1
      if f.shape[1] == 1:
          raise NotImplementedError("noncanonical case ignored")
      else:
          log_kern = self.dirichlet_kernel(f, bandwidth).squeeze()
      # Trick: -inf on the diagonal
      log_kern.fill_diagonal_(torch.finfo(torch.float).min)
      return log_kern

  def get_kernels(self, f, bandwidths, device):
      log_kern = self.dirichlet_kernels(f, bandwidths).squeeze()
      # Trick: -inf on the diagonal
      # log_kern of shape samples * samples * bandwidths
      return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)[:,:,None]
  
  def get_ratio_canonical_log_efficient_v3(self, f, y, bandwidth, p, device='cpu'):
      log_kern = self.get_kernel_efficient(f, bandwidth, device)
      log_den = log_kern - torch.logsumexp(log_kern, dim=1, keepdim=True)
      kern = torch.exp(log_den)
      y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(kern.dtype)
      ratio = torch.matmul(kern, y_onehot)
      ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)
      return torch.mean(ratio)

  def get_ece_kde(self, f, y, bandwidth, p, device, mc_type='canonical'):
      """
      Calculate an estimate of Lp calibration error.
      :param f: The vector containing the probability scores, shape [num_samples, num_classes]
      :param y: The vector containing the labels, shape [num_samples]
      :param bandwidth: The bandwidth of the kernel
      :param p: The p-norm. Typically, p=1 or p=2
      :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
      :param device: The device type: 'cpu' or 'cuda'
      :return: An estimate of Lp calibration error
      """
      device = f.device
      self.check_input(f, bandwidth, mc_type)
      if f.shape[1] == 1:
          raise NotImplementedError("noncanonical case not supported")
      else:
          if mc_type == 'canonical':
              return self.get_ratio_canonical_log_efficient_v3(f, y, bandwidth, p, device)

  def forward(self, logits, labels):  
    order = self.order

    if self.UseCPU:
      logits, labels = logits.detach().cpu(), labels.detach().cpu()
    else:
      logits, labels = logits.detach(), labels.detach()
    device = logits.device
    
    f = torch.softmax(logits, dim=1)
    f = torch.clip(f,min=1e-25,max=1.0-1e-25)
    f = f.to(torch.float64)   # more accurate version, but can cost more memory
    y = labels
    
    bandwidth = self.banwidths
    error = self.get_ece_kde(f, y, bandwidth, p=order, device=device, mc_type='canonical')
    
    return error
  

class sKSCE(nn.Module):
  """
  Computes the Kolmogorov-Smirnov Calibration Error (KS-CE) from:
  "Better Uncertainty Calibration via Proper Scores for Classification and Beyond"
  - https://github.com/kartikgupta-at-anu/spline-calibration/blob/master/cal_metrics/KS.py
  
  Original KS-based calibration error is published in:
  《Calibration of Neural Networks using Splines》

  NOTE:
  - The paper "Better Uncertainty Calibration via Proper Scores for Classification and Beyond" modified KSCE by using squared differences and taking the square root
  - However, the original definition (from "Calibration of Neural Networks using Splines") is based on absolute differences
  - This implementation follows the original and omits both squaring and root-taking
  
  Args:
      n_bins (int): Not used directly, but included for compatibility.
      debiased (bool): Placeholder, not used in this implementation.
      marginal (bool): Placeholder, not used.
      adaptive (bool): Placeholder, not used.
      power (float): Placeholder, not used.
  """
  def __init__(self, n_bins=15, debiased = True, marginal=False, adaptive=True, power = 1.0 ):
     super(sKSCE,self).__init__()
     self.n_bins = n_bins
     self.adaptive = adaptive
     self.marginal = marginal
     self.debiased = debiased
     self.power = power

  def ensure_numpy(self, a):
      if not isinstance(a, np.ndarray): a = a.numpy()
      return a

  def get_top_results(self, scores, labels, nn=-1, inclusive=False, return_topn_classid=False) :
      
      if nn==-1:         
        #  nn should be negative, -1 means top, -2 means second top, etc
        # Get the position of the n-th largest value in each row
        topn = np.argmax(scores,axis=1)
        nthscore = np.max(scores,axis=1)
        labs = (topn == labels).astype('float64')
      else:
         # the above is the adoptd source code, which is too slow let us speed it up
        topn = [np.argpartition(score, nn)[nn] for score in scores]
        nthscore = [score[n] for score, n in zip (scores, topn)]
        labs = [1.0 if int(label) == int(n) else 0.0 for label, n in zip(labels, topn)]         

      # Change to tensor
      tscores = np.array (nthscore)
      tacc = np.array(labs)

      if return_topn_classid:
          return tscores, tacc, topn
      else:
          return tscores, tacc
        
  def forward(self, logits, labels):
    """
    Computes the KS Calibration Error based on cumulative distribution mismatch.

    Steps:
    - Converts logits to softmax scores
    - Extracts top-1 predicted probabilities and correct/incorrect flags
    - Sorts samples by confidence
    - Computes cumulative accuracy and cumulative confidence
    - Returns the maximum absolute difference between them (i.e., KS statistic)

    Args:
        logits (Tensor): Raw model outputs, shape (N, C)
        labels (Tensor): Ground truth labels, shape (N,)

    Returns:
        float: Kolmogorov-Smirnov Calibration Error
    """
    
    # Convert logits to softmax probabilities
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Get top-1 confidence and correctness indicators
    scores, labels = self.get_top_results(scores, labels)

    # Ensure input arrays are numpy arrays
    scores = self.ensure_numpy (scores)
    labels = self.ensure_numpy (labels)

    # Sort by confidence
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Compute cumulative distributions
    n = scores.shape[0]
    integrated_scores = np.cumsum(scores) / n
    integrated_accuracy = np.cumsum(labels) / n

    # Compute Kolmogorov-Smirnov statistic (max absolute difference)
    KS_error_max = np.amax(np.absolute(integrated_scores - integrated_accuracy))

    return KS_error_max    


class BinMethod(abc.ABC):
  """
  Abstract base class for binning strategies used in calibration metrics.
  Each subclass should implement how to assign confidence scores to bins.
  """
  def __init__(self, num_bins):
    self.num_bins = num_bins

  @abc.abstractmethod
  def compute_bin_indices(self, scores):
    """Assign a bin index for each score.
    Args:
      scores: np.array of shape (num_examples, num_classes) containing the
        model's confidence scores
    Returns:
      bin_indices: np.array of shape (num_examples, num_classes) containing the
        bin assignment for each score
    """
    pass


class BinEqualWidth(BinMethod):
  """
  Binning strategy that divides [0, 1] into equal-width intervals.
  """
  def compute_bin_indices(self, scores):
    """
    Assign bin indices to each score.

    Args:
        scores (np.ndarray): shape (N, K), confidence scores.

    Returns:
        bin_indices (np.ndarray): shape (N, K), bin assignment for each score.
    """
    edges = np.linspace(0.0, 1.0, self.num_bins + 1)
    bin_indices = np.digitize(scores, edges, right=False)
    # np.digitze uses one-indexed bins, switch to using 0-indexed
    bin_indices = bin_indices - 1
    # Put examples with score equal to 1.0 in the last bin.
    bin_indices = np.where(scores == 1.0, self.num_bins - 1, bin_indices)
    return bin_indices


class BinEqualExamples(BinMethod):
  """
  Binning strategy that attempts to ensure each bin contains
  roughly the same number of examples (equal-mass binning).
  """

  def compute_bin_indices(self, scores):
    """Assign a bin index for each score assumes equal num examples per bin.
    Args:
      scores: np.ndarray of shape [N, K] containing the model's confidence
    Returns:
      bin_indices: np.ndarray of shape [N, K] containing the bin assignment for
        each score
    """
    num_examples = scores.shape[0]
    num_classes = scores.shape[1]

    bin_indices = np.zeros((num_examples, num_classes), dtype=int)
    for k in range(num_classes):
      sort_ix = np.argsort(scores[:, k])
      bin_indices[:, k][sort_ix] = np.minimum(
          self.num_bins - 1,
          np.floor((np.arange(num_examples) / num_examples) *
                   self.num_bins)).astype(int)
    return bin_indices


class CalibrationMetric(nn.Module):
  """
  General-purpose calibration error computation module.

  Supports:
  - Multiple binning strategies (equal-width, equal-mass)
  - Top-label or marginal multiclass calibration
  - Lp-norm based error types (e.g. ECE)
  - Monotonic bin sweep heuristics
  """  
  def __init__(self,
               ce_type="quant",
               num_bins=15,
               bin_method="equal_examples",
               norm=2,
               multiclass_setting="top_label"):
    super(CalibrationMetric,self).__init__()
    """Initialize calibration metric class.
    Args:
      ce_type: str describing the type of calibration error to compute.
        em_ece_bin implements equal mass ECE_bin
        ew_ece_bin implements equal width ECE_bin
        em_ece_sweep implements equal mass ECE_sweep
        ew_ece_sweep implements equal width ECE_sweep
      num_bins: int for number of bins.
      bin_method: string for binning technique to use. Must be either
        "equal_width", "equal_examples" or "".
      norm: integer for norm to use to compute the calibration error. Norm
        should be > 0.
      multiclass_setting: string specifying the type of multiclass calibration
        error to compute. Must be "top_label" or "marginal". If "top_label",
        computes the calibration error of the top class. If "marginal", computes
        the marginal calibration error.
    """
    if bin_method not in ["equal_width", "equal_examples", ""]:
      raise NotImplementedError("Bin method not supported.")
    if multiclass_setting not in ["top_label", "marginal"]:
      raise NotImplementedError(
          f"Multiclass setting {multiclass_setting} not supported.")
    if bin_method == "equal_width" or ce_type.startswith("ew"):
      self.bin_method = BinEqualWidth(num_bins)
    elif bin_method == "equal_examples" or ce_type.startswith("em"):
      self.bin_method = BinEqualExamples(num_bins)
    elif bin_method == "None":
      self.bin_method = None
    else:
      raise NotImplementedError(f"Bin method {bin_method} not supported.")

    self.ce_type = ce_type
    self.norm = norm
    self.num_bins = num_bins
    self.multiclass_setting = multiclass_setting
    self.configuration_str = "{}_bins:{}_{}_norm:{}_{}".format(
        ce_type, num_bins, bin_method, norm, multiclass_setting)

  def predict_top_label(self, fx, y):
    """Compute confidence scores and correctness of predicted labels.
    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.
    Returns:
      fx_top: np.ndarray of shape [N, 1] for confidence score of top label.
      hits: np.ndarray of shape [N, 1] denoting whether or not top label
        is a correct prediction or not.
    """
    picked_classes = np.argmax(fx, axis=1)
    labels = np.argmax(y, axis=1)
    hits = 1 * np.array(picked_classes == labels, ndmin=2).transpose()
    fx_top = np.max(fx, axis=1, keepdims=True)
    return fx_top, hits

  def _compute_error_no_bins(self, fx, y):
    """Compute error without binning."""
    num_examples = fx.shape[0]
    ce = pow(np.abs(fx - y), self.norm)
    return pow(ce.sum() / num_examples, 1. / self.norm)

  def _compute_error_all_binned(self, binned_fx, binned_y, bin_sizes):
    """Compute calibration error given binned data."""
    num_examples = np.sum(bin_sizes[:, 0])
    num_classes = binned_fx.shape[1]
    ce = pow(np.abs(binned_fx - binned_y), self.norm) * bin_sizes
    ce_sum = 0
    for k in range(num_classes):
      ce_sum += ce[:, k].sum()
    return pow(ce_sum / (num_examples*num_classes), 1. / self.norm)

  def _compute_error_label_binned(self, fx, binned_y, bin_indices):
    """Compute label binned calibration error."""
    num_examples = fx.shape[0]
    num_classes = fx.shape[1]
    ce_sum = 0.0
    for k in range(num_classes):
      for i in range(num_examples):
        ce_sum += pow(
            np.abs(fx[i, k] - binned_y[bin_indices[i, k], k]), self.norm)
    ce_sum = pow(ce_sum / num_examples, 1. / self.norm)
    return ce_sum

  def _bin_data(self, fx, y):
    """Bin fx and y.
    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.
    Returns:
      A tuple containing:
        - binned_fx: np.ndarray of shape [B, K] containing mean
            predicted score for each bin and class
        - binned_y: np.ndarray of shape [B, K]
            containing mean empirical accuracy for each bin and class
        - bin_sizes: np.ndarray of shape [B, K] containing number
            of examples in each bin and class
    """
    bin_indices = self.bin_method.compute_bin_indices(fx)
    num_classes = fx.shape[1]

    binned_fx = np.zeros((self.num_bins, num_classes))
    binned_y = np.zeros((self.num_bins, num_classes))
    bin_sizes = np.zeros((self.num_bins, num_classes))

    for k in range(num_classes):
      for bin_idx in range(self.num_bins):
        indices = np.where(bin_indices[:, k] == bin_idx)[0]
        # Disable for Numpy containers.
        # pylint: disable=g-explicit-length-test
        if len(indices) > 0:
          # pylint: enable=g-explicit-length-test
          mean_score = np.mean(fx[:, k][indices])
          mean_accuracy = np.mean(y[:, k][indices])
          bin_size = len(indices)
        else:
          mean_score = 0.0
          mean_accuracy = 0.0
          bin_size = 0
        binned_fx[bin_idx][k] = mean_score
        binned_y[bin_idx][k] = mean_accuracy
        bin_sizes[bin_idx][k] = bin_size

    return binned_fx, binned_y, bin_sizes, bin_indices

  def _compute_error_monotonic_sweep(self, fx, y):
    """Compute ECE using monotonic sweep method."""
    fx = np.squeeze(fx)
    y = np.squeeze(y)
    non_nan_inds = np.logical_not(np.isnan(fx))
    fx = fx[non_nan_inds]
    y = y[non_nan_inds]

    if self.ce_type == "em_ece_sweep":
      bins = self.em_monotonic_sweep(fx, y)
    elif self.ce_type == "ew_ece_sweep":
      bins = self.ew_monotonic_sweep(fx, y)
    n_bins = np.max(bins) + 1
    ece, _ = self._calc_ece_postbin(n_bins, bins, fx, y)
    return ece

  def _calc_ece_postbin(self, n_bins, bins, fx, y):
    """Calculate ece_bin after bins are computed and determine monotonicity."""
    ece = 0.
    monotonic = True
    last_ym = -1000
    for i in range(n_bins):
      cur = bins == i
      if any(cur):
        fxm = np.mean(fx[cur])
        ym = np.mean(y[cur])
        if ym < last_ym:  # determine if predictions are monotonic
          monotonic = False
        last_ym = ym
        n = np.sum(cur)
        ece += n * pow(np.abs(ym - fxm), self.norm)
    return (pow(ece / fx.shape[0], 1. / self.norm)), monotonic

  def em_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal mass binning scheme."""
    sort_ix = np.argsort(fx)
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)

    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins[sort_ix] = np.minimum( n_bins - 1, np.floor( (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins

  def ew_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal width binning scheme."""
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)
    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins = np.minimum(n_bins - 1, np.floor(fx * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins
  
  def compute_error(self, fx, y):
    """Compute the calibration error given softmax fx and one hot labels.
    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.
    Returns:
      calibration error
    """
    if len(fx.shape) == 1:
      print("WARNING: reshaping fx, assuming K=1")
      fx = fx.reshape(len(fx), 1)
    if len(y.shape) == 1:
      print("WARNING: reshaping one hot labels, assuming K=1")
      y = y.reshape(len(y), 1)

    if np.max(fx) > 1.:
      raise ValueError("Maximum score of {} is greater than 1.".format(
          np.max(fx)))
    if np.min(fx) < 0.:
      raise ValueError("Minimum score of {} is less than 0.".format(np.min(fx)))

    if np.max(y) > 1.:
      raise ValueError("Maximum label of {} is greater than 1.".format(
          np.max(y)))
    if np.min(y) < 0.:
      raise ValueError("Minimum label of {} is less than 0.".format(np.min(y)))

    num_classes = fx.shape[1]
    if self.multiclass_setting == "top_label" and num_classes > 1:
      fx, y = self.predict_top_label(fx, y)

    if self.num_bins > 0 and self.bin_method:
      binned_fx, binned_y, bin_sizes, bin_indices = self._bin_data(fx, y)

    if self.ce_type in ["ew_ece_bin", "em_ece_bin"]:
      calibration_error = self._compute_error_all_binned( binned_fx, binned_y, bin_sizes)
    elif self.ce_type in ["label_binned"]:
      calibration_error = self._compute_error_label_binned( fx, binned_y, bin_indices)
    elif self.ce_type.endswith(("sweep")):
      calibration_error = self._compute_error_monotonic_sweep(fx, y)
    else:
      raise NotImplementedError("Calibration error {} not supported.".format(
          self.ce_type))

    return calibration_error
    
  def forward(self, logits, labels):
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()
    y_onehot = nn.functional.one_hot(labels, num_classes=logits.shape[1]).to(torch.float32)
    y_onehot = y_onehot.detach().cpu().numpy()     
    error = self.compute_error(scores,y_onehot)
    return error


class MMCE(nn.Module):
    # this metric is proposed in paper "Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings"
    # the implementation code is adopted from netcal study in https://github.com/EFS-OpenSource/calibration-framework/blob/main/netcal/metrics/confidence/MMCE.py
    """
    Maximum Mean Calibration Error (MMCE).
    The MMCE [1]_ is a differentiable approximation to the Expected Calibration Error (ECE) using a
    reproducing _kernel Hilbert space (RKHS).
    Using a dataset :math:`\\mathcal{D}` of size :math:`N` consisting of the ground truth labels
    :math:`\\hat{y} \\in \\{1, ..., K \\}` with input :math:`x \\in \\mathcal{X}`, the MMCE is calculated by using
    a scoring classifier :math:`\\hat{p}=h(x)` that returns the highest probability for a certain class in conjunction
    with the predicted label information :math:`y \\in \\{1, ..., K \\}` and is defined by
    .. math::
       MMCE = \\sqrt{\\sum_{i, j \\in \\mathcal{D}} \\frac{1}{N^2}(\\mathbb{1}(\\hat{y}_i = y_i) - \\hat{p}_i) (\\mathbb{1}(\\hat{y}_j = y_j) - \\hat{p}_j)k(\\hat{p}_i, \\hat{p}_j)} ,
    with :math:`\\mathbb{1}(*)` as the indicator function and a Laplacian _kernel :math:`k` defined by
    .. math::
       k(\\hat{p}_i, \\hat{p}_j) = \\exp(-2.5 |\\hat{p}_i - \\hat{p}_j|) .
    Parameters
    ----------
    detection : bool, default: False
        Detection mode is currently not supported for MMCE!
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    References
    ----------
    .. [1] Kumar, Aviral, Sunita Sarawagi, and Ujjwal Jain:
       "Trainable calibration measures for neural networks from _kernel mean embeddings."
       International Conference on Machine Learning. 2018.
       `Get source online <http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf>`__.
    """

    def __init__(self):
      super(MMCE,self).__init__()

    def _batched(self,X,y,batched: bool = False):
        # batched: interpret X and y as multiple predictions

        if not batched:
            assert isinstance(X, np.ndarray), 'Parameter \'X\' must be Numpy array if not on batched mode.'
            assert isinstance(y, np.ndarray), 'Parameter \'y\' must be Numpy array if not on batched mode.'
            X, y = [X], [y]

        # if we're in batched mode, create new lists for X and y to prevent overriding
        else:
            assert isinstance(X, (list, tuple)), 'Parameter \'X\' must be type list on batched mode.'
            assert isinstance(y, (list, tuple)), 'Parameter \'y\' must be type list on batched mode.'
            X, y = [x for x in X], [y_ for y_ in y]

        # if input X is of type "np.ndarray", convert first axis to list
        # this is necessary for the following operations
        if isinstance(X, np.ndarray):
            X = [x for x in X]

        if isinstance(y, np.ndarray):
            y = [y0 for y0 in y]

        return X, y

    def _kernel(self, confidence):
        """ Laplacian _kernel """

        diff = confidence[:, None] - confidence
        return np.exp(-2.5 * np.abs(diff))

    def measure(self,X,y,batched: bool = False):
        """
        Measure calibration by given predictions with confidence and the according ground truth.
        Parameters
        ----------
        X : iterable of np.ndarray, or np.ndarray of shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            If this is an iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        y : iterable of np.ndarray with same length as X or np.ndarray of shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            If iterable over multiple instances of np.ndarray and parameter batched=True,
            interpret this parameter as multiple predictions that should be averaged.
        batched : bool, optional, default: False
            Multiple predictions can be evaluated at once (e.g. cross-validation examinations) using batched-mode.
            All predictions given by X and y are separately evaluated and their results are averaged afterwards
            for visualization.
        Returns
        -------
        float
            Returns Maximum Mean Calibration Error.
        """

        X, y = self._batched(X, y, batched)

        mmce = []
        for X_batch, y_batch in zip(X, y):

            # assert y_batch is one-hot with 2 dimensions
            if y_batch.ndim == 2:
                y_batch = np.argmax(y_batch, axis=1)

            # get max confidence and according label
            if X_batch.ndim == 1:
                confidence, labels = X_batch, np.where(X_batch > 0.5, np.ones_like(X_batch), np.zeros_like(X_batch))
            elif X_batch.ndim == 2:
                confidence, labels = np.max(X_batch, axis=1), np.argmax(X_batch, axis=1)
            else:
                raise ValueError("MMCE currently not defined for input arrays with ndim>3.")

            n_samples = float(confidence.size)

            # get matched flag and difference
            matched = (y_batch == labels).astype(confidence.dtype)
            diff = np.expand_dims(matched - confidence, axis=1)

            # now calculate product of differences for each pair
            confidence_pairs = np.matmul(diff, diff.T)

            # caculate _kernel for each pair
            kernel_pairs = self._kernel(confidence)

            miscalibration = np.sqrt(np.sum(confidence_pairs * kernel_pairs) / np.square(n_samples))
            mmce.append(miscalibration)

        mmce = np.mean(mmce)
        return mmce
    
    def forward(self, logits, labels):
      scores = F.softmax(logits, dim=1).detach().cpu().numpy()
      labels = labels.detach().cpu().numpy() 
      error = self.measure(scores,labels)
      return error


class _eval_in_IMax(nn.Module):
  # the code is adopted from https://github.com/boschresearch/imax-calibration/blob/3c0a0c8544545a868205cb8460aea3830e23894d/imax_calib/evaluations/calibration_metrics.py#L36
  # publihsed in the paper "Multi-class uncertainty calibration via mutual information maximization-based binning"
  # Some parts of the original code have been modified to improve numerical stability during computation.
  def __init__(self, nclasses, num_bins=15, marginal = False, list_approximators=["dECE",]):
    # ["dECE", "mECE", "kECE"]
    super(_eval_in_IMax,self).__init__()
    self.num_bins = num_bins
    self.list_approximators = list_approximators
    self.marginal = marginal
        
    self.IMax_FasterMode = False
    IMax_centroids = np.empty((nclasses,num_bins))
    IMax_centroids[:] = np.nan
    self.IMax_centroids = IMax_centroids


  def forward(self, logits, labels):
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy() 
    error = self.compute_top_1_and_CW_ECEs(scores,labels, list_approximators= self.list_approximators, num_bins=self.num_bins)

    return error

  def compute_top_1_and_CW_ECEs(self, multi_cls_probs, multi_cls_labels, list_approximators=["dECE", "mECE", "kECE"], num_bins=15):
      # list_approximators=["dECE", "mECE", "iECE", "kECE"]
      """
      Given the multi-class predictions and labels, this function computes the top1 and CW ECEs. Will compute it by calling the other functions in this script.
      Parameters:
      -----------
      multi_cls_probs: 2D ndarray
          predicted probabilities
      multi_cls_labels: 1D or 2D ndarray
          label indices or one-hot labels. Will be converted to one-hot
      Return:
      -------
      ece_dict: dict
          Dictionary with all the ECE estimates
      """
      assert len(multi_cls_probs.shape) == 2
      if len(multi_cls_labels.shape) == 1:  # not one-hot. so convert to one-hot
          multi_cls_labels = np.eye(multi_cls_probs.shape[1])[multi_cls_labels]

      n_classes = multi_cls_probs.shape[1]
      for ece_approx in list_approximators:
          if self.marginal == False:
            top_1_preds = multi_cls_probs.max(axis=-1)
            top_1_correct = multi_cls_probs.argmax(axis=-1) == multi_cls_labels.argmax(axis=-1)

            top_1_ECE = eval("self.measure_%s_calibration" % (ece_approx))(pred_probs=top_1_preds, correct=top_1_correct, num_bins=num_bins, class_idx=-1)["ece"]
            error = top_1_ECE

          elif self.marginal == True:

            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = [executor.submit(eval("self.measure_%s_calibration" % (ece_approx)), multi_cls_probs[:, class_idx], multi_cls_labels[:, class_idx], num_bins, 1.0/n_classes, 
                                           class_idx, self.IMax_centroids[class_idx], self.IMax_FasterMode) for class_idx in range(n_classes)]
                answers = [future.result() for future in futures]
            if ece_approx == 'kECE' and self.IMax_FasterMode:
                mean_cw_ECE = np.mean([ans[0]["ece"] for ans in answers])
                self.IMax_centroids = np.array([ans[1] for ans in answers])
            else:
                mean_cw_ECE = np.mean([ans["ece"] for ans in answers])
            error = mean_cw_ECE

      return error   
  
  @staticmethod
  def _ece(avg_confs, avg_accs, counts):
    """
    Helper function to compute the Expected Calibration Error.
    Parameters
    ----------
    avg_confs: Averaged probability of predictions per bin (confidence)
    avg_accs: Averaged true accuracy of predictions per bin
    counts: Number of predictions per bin
    Returns
    -------
    ece: float - calibration error
    """
    return np.sum((counts / counts.sum()) * np.absolute(avg_confs - avg_accs))

  @staticmethod
  def measure_dECE_calibration(pred_probs, correct, num_bins=100, threshold=-1, *args):
    """
    Compute the calibration curve using the equal size binning scheme (i.e. equal size bins)and computes the calibration error given this binning scheme (i.e. dECE).
    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()
    """
    assert len(pred_probs.shape) == 1
    bin_boundaries_prob = _eval_in_IMax.to_sigmoid(_eval_in_IMax.nolearn_bin_boundaries(num_bins, binning_scheme="eqsize"))
    assigned = _eval_in_IMax.bin_data(pred_probs, bin_boundaries_prob)
    return _eval_in_IMax.calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)
  
  @staticmethod
  def measure_mECE_calibration(pred_probs, correct, num_bins=100, threshold=-1, *args):
    """
    Compute the calibration curve using the equal mass binning scheme (i.e. equal mass bins)and computes the calibration error given this binning scheme (i.e. mECE).
    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()
    """
    assert len(pred_probs.shape) == 1
    logodds = _eval_in_IMax.to_logodds(pred_probs)
    # if logodds.max()<=1 and logodds.min()>=0:
    bin_boundaries_prob = _eval_in_IMax.to_sigmoid(_eval_in_IMax.nolearn_bin_boundaries(num_bins, binning_scheme="eqmass", x=logodds))
    assigned = _eval_in_IMax.bin_data(pred_probs, bin_boundaries_prob)
    return _eval_in_IMax.calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)
  
  @staticmethod
  def measure_kECE_calibration(pred_probs, correct, num_bins=100, threshold=-1, class_idx=-1, kcentroid=np.nan, IMax_FasterMode=False):
    """
    Compute the calibration curve using the kmeans binning scheme (i.e. use kmeans to cluster the data and then determine the bin assignments) and computes the calibration error given this binning scheme (i.e. kECE).
    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()
    """

    assert len(pred_probs.shape) == 1
    if (class_idx!=-1) and IMax_FasterMode:
       # faster mode for class-wise probability
      if np.any(np.isnan(kcentroid)):
          centroids, _ = scipy.cluster.vq.kmeans(pred_probs, num_bins)
      else:
          centroids, _ = scipy.cluster.vq.kmeans(pred_probs, kcentroid)
          Count=1
          while len(centroids)!=num_bins:
            centroids, _ = scipy.cluster.vq.kmeans(pred_probs, num_bins)
            Count += 1
            if Count>5:
              break
                          
      cluster_ids, _ = scipy.cluster.vq.vq(pred_probs, centroids)
      cluster_ids = cluster_ids.astype(np.int)         
      return _eval_in_IMax.calibration_error_and_curve(pred_probs, correct, cluster_ids, num_bins, threshold), centroids 
      
    else:
      centroids, _ = scipy.cluster.vq.kmeans(pred_probs, num_bins)
      cluster_ids, _ = scipy.cluster.vq.vq(pred_probs, centroids)
      cluster_ids = cluster_ids.astype(np.int)
      return _eval_in_IMax.calibration_error_and_curve(pred_probs, correct, cluster_ids, num_bins, threshold)  
  
  @staticmethod
  def measure_quantized_calibration(pred_probs, correct, assigned, num_bins=100, threshold=-1):
    """
    Compute the calibration curve given the bin assignments (i.e. quantized values).
    """
    assert len(pred_probs.shape) == 1
    return _eval_in_IMax.calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)

  @staticmethod
  def calibration_error_and_curve(pred_probs, correct, assigned, num_bins=100, threshold=-1):
      """
      Compute the calibration curve and calibration error. The threshold float will determine which samples to ignore because its confidence is very low.
      Parameters
      ----------
          see calibration_curve_quantized()
      Returns
      -------
      results: dict
          dictionary with calibration information
      """
      assert len(pred_probs.shape) == 1
      mask = pred_probs > threshold
      pred_probs, correct, assigned = pred_probs[mask], correct[mask], assigned[mask]
      cov = mask.mean()
      prob_pred, prob_true, counts, counts_unfilt = _eval_in_IMax.calibration_curve_quantized(pred_probs, correct, assigned=assigned, num_bins=num_bins)
      ece = _eval_in_IMax._ece(prob_pred, prob_true, counts)
      return {"ece": ece, "prob_pred": prob_pred, "prob_true": prob_true, "counts": counts, "counts_unfilt": counts_unfilt, "threshold": threshold, "cov": cov}
  
  @staticmethod
  def calibration_curve_quantized(pred_probs, correct, assigned, num_bins=100):
    """
    Get the calibration curve given the bin assignments, samples and sample-correctness.
    Parameters
    ----------
    pred_probs: numpy ndarray
        numpy array with predicted probabilities (i.e. confidences)
    correct: numpy ndarray
        0/1 indicating if the sample was correctly classified or not
    num_bins: int
        number of bins for quantization
    Returns
    -------
    prob_pred: for each bin the avg. confidence
    prob_true: for each bin the avg. accuracy
    counts: number of samples in each bin
    counts_unfilt: same as `counts` but also including zero bins
    """
    assert len(pred_probs.shape) == 1
    bin_sums_pred = np.bincount(assigned, weights=pred_probs,  minlength=num_bins)
    bin_sums_true = np.bincount(assigned, weights=correct, minlength=num_bins)
    counts = np.bincount(assigned, minlength=num_bins)
    filt = counts > 0
    prob_pred = (bin_sums_pred[filt] / counts[filt])
    prob_true = (bin_sums_true[filt] / counts[filt])
    counts_unfilt = counts
    counts = counts[filt]
    return prob_pred, prob_true, counts, counts_unfilt

  @staticmethod
  def to_sigmoid(x):
      """
      Stable sigmoid in numpy. Uses tanh for a more stable sigmoid function.
      Parameters
      ----------
      x : numpy ndarray
        Logits of the network as numpy array.
      Returns
      -------
      sigmoid: numpy ndarray
        Sigmoid output
      """
      sigmoid = 0.5 + 0.5 * np.tanh(x/2)
      assert np.all(np.isfinite(sigmoid)) == True, "Sigmoid output contains NaNs. Handle this."
      return sigmoid
  
  @staticmethod
  def nolearn_bin_boundaries(num_bins, binning_scheme, x=None):
    """
    Get the bin boundaries (in logit space) of the <num_bins> bins. This function returns only the bin boundaries which do not include any type of learning.
    For example: equal mass bins, equal size bins or overlap bins.
    Parameters
    ----------
    num_bins: int
        Number of bins
    binning_scheme: string
        The way the bins should be placed.
            'eqmass': each bin has the same portion of samples assigned to it. Requires that `x is not None`.
            'eqsize': equal spaced bins in `probability` space. Will get equal spaced bins in range [0,1] and then convert to logodds.
            'custom_range[min_lambda,max_lambda]': equal spaced bins in `logit` space given some custom range.
    x: numpy array (1D,)
        array with the 1D data to determine the eqmass bins.
    Returns
    -------
    bins: numpy array (num_bins-1,)
        Returns the bin boundaries. It will return num_bins-1 bin boundaries in logit space. Open ended range on both sides.
    """
    if binning_scheme == "eqmass":
        assert x is not None and len(x.shape) == 1
        bins = np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1)  # num_bins-1 boundaries for open ended sides
        bins = np.percentile(x, bins * 100, interpolation='lower')  # data will ensure its in Logit space
    elif binning_scheme == "eqsize":  # equal spacing in logit space is not the same in prob space because of sigmoid non-linear transformation
        bins = _eval_in_IMax.to_logodds(np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1))  # num_bins-1 boundaries for open ended sides
    elif "custom_range" in binning_scheme:  # used for example when you want bins at overlap regions. then custom range should be [ min p(y=1), max p(y=0)  ]. e.g. custom_range[-5,8]
        custom_range = eval(binning_scheme.replace("custom_range", ""))
        assert type(custom_range) == list and (custom_range[0] <= custom_range[1])
        bins = np.linspace(custom_range[0], custom_range[1], num_bins-1)  # num_bins-1 boundaries for open ended sides
    return bins
  
  @staticmethod
  def to_logodds(x):
    """
    Convert probabilities to logodds using:
    .. math::
        \\log\\frac{p}{1-p} ~ \\text{where} ~ p \\in [0,1]
    Natural log.
    Parameters
    ----------
    x : numpy ndarray
       Class probabilties as numpy array.
    Returns
    -------
    logodds : numpy ndarray
       Logodds output
    """
    assert x.max() <= 1 and x.min() >= 0
    numerator = x
    denominator = 1-x
    #numerator[numerator==0] = EPS
    # denominator[denominator==0] = EPS # 1-EPS is basically 1 so not stable!
    logodds = _eval_in_IMax.safe_log_diff(numerator, denominator, np.log)  # logodds = np.log( numerator/denominator   )
    assert np.all(np.isfinite(logodds)) == True, "Logodds output contains NaNs. Handle this."
    return logodds

  @staticmethod
  def safe_log_diff(A, B, log_func=np.log):
      """
      Numerically stable log difference function. Avoids log(0). Will compute log(A/B) safely where the log is determined by the log_func
      """
      EPS = np.finfo(float).eps 
      if np.isscalar(A):
          if A == 0 and B == 0:
              return log_func(EPS)
          elif A == 0:
              return log_func(EPS) - log_func(B)
          elif B == 0:
              return log_func(A) - log_func(EPS)
          else:
              return log_func(A) - log_func(B)
      else:
          # log(A) - log(B)
          # output = np.where(A == 0, log_func(EPS), log_func(A)) - np.where(B == 0, log_func(EPS), log_func(B))
          output = log_func(np.clip(A,a_min = EPS,a_max=None)) -  log_func(np.clip(B,a_min=EPS,a_max=None)) # rewritten version to avoid warnining

          output[np.logical_or(A == 0, B == 0)] = log_func(EPS)
          assert np.all(np.isfinite(output))
          return output

  @staticmethod
  def bin_data(x, bins):
      """
      Given bin boundaries quantize the data (x). When ndims(x)>1 it will flatten the data, quantize and then reshape back to orig shape.
      Returns the following quantized values for num_bins=10 and bins = [2.5, 5.0, 7.5, 1.0]\n
      quantize: \n
                (-inf, 2.5) -> 0\n
                [2.5, 5.0) -> 1\n
                [5.0, 7.5) -> 2\n
                [7.5, 1.0) -> 3\n
                [1.0, inf) -> 4\n
      Parameters
      ----------
      x: numpy ndarray
        Network logits as numpy array
      bins: numpy ndarray
          location of the (num_bins-1) bin boundaries
      Returns
      -------
      assigned: int numpy ndarray
          For each sample, this contains the bin id (0-indexed) to which the sample belongs.
      """
      orig_shape = x.shape
      # if not 1D data. so need to reshape data, then quantize, then reshape back
      if len(orig_shape) > 1 or orig_shape[-1] != 1:
          x = x.flatten()
      assigned = np.digitize(x, bins)  # bin each input in data. np.digitize will always return a valid index between 0 and num_bins-1 whenever bins has length (num_bins-1) to cater for the open range on both sides
      if len(orig_shape) > 1 or orig_shape[-1] != 1:
          assigned = np.reshape(assigned, orig_shape)
      return assigned.astype(np.int)
  




