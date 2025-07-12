"""
This module provides builder interface functions for learning post-hoc calibraters, including:

- Data loading from pickled logits and labels
- Calibration mapping/model construction based on configuration
- Optimizer and scheduler setup
- Loss function configuration
- Evaluation metric construction
- Model selection criteria based on calibration metrics

Author: Wenjian Huang
"""

import torch
import numpy as np
import pickle
import os
import torch.utils.data as data_utils
import torch.nn as nn

from models.monotonic_nonlinear import MonotonicModel_v1
from models.monotonic_pwlinear import PWLinearModel
from models.temperature_scaling import TempScaling
from models.tsensemble_linear import EnsembleTempScal

import utils.utils as utils
from loss.hCalib import hcalibration 
import functools

def acc(pred_logits, targets):
    """Compute classification accuracy from predicted logits and true labels."""
    preds = np.argmax(pred_logits, axis=1)
    return np.mean(preds == targets[:, 0])


def get_dataloaders(dataset_config, is_training, verbose=True):
    """
    Load logits and labels from disk and wrap them as PyTorch dataloaders.

    Args:
        dataset_config: Object containing dataset loading parameters:
            - root (str): Root directory containing the pickled data
            - name (str): Filename of the dataset
            - batch_size (int): Batch size for DataLoader
            - shuffle (bool): Whether to shuffle the dataset
            - num_workers (int): Number of worker processes
        is_training (bool): Whether to return train/test dataset
        verbose (bool): Whether to print dataset statistics

    Returns:
        train_loader/evaluation_loader (DataLoader): Loader for training or validation set
        test_loader: Loader for test set
        num_classes (int): Number of output classes
    """
    root = dataset_config.root
    name = dataset_config.name
    # load logits into memory
    pickle_path = os.path.join(root, name)
    assert (os.path.exists(pickle_path))
    with open(pickle_path, 'rb') as f:
        (y_logits_val, y_val), (y_logits_test, y_test) = pickle.load(f)

    if len(y_val.shape) == 1:
        y_val = y_val[..., None]

    if len(y_test.shape) == 1:
        y_test = y_test[..., None]

    # show some statistics
    if verbose:
        print("y_logits_val:", y_logits_val.shape)
        print("y_true_val:", y_val.shape)
        print("y_logits_test:", y_logits_test.shape)
        print("y_true_test:", y_test.shape)
        print("val accuracy: {}".format(acc(y_logits_val, y_val)))
        print("test accuracy: {}".format(acc(y_logits_test, y_test)))

    num_classes = y_logits_val.shape[1]
    def tensorify(x): return torch.tensor(x)
    if is_training:
        full_dataset = data_utils.TensorDataset(tensorify(y_logits_val), tensorify(y_val))

        train_loader = data_utils.DataLoader(full_dataset, batch_size=dataset_config.batch_size,
                                                    shuffle=dataset_config.shuffle,
                                                    num_workers=dataset_config.num_workers)
        evaluation_loader = data_utils.DataLoader(full_dataset, batch_size=dataset_config.batch_size,
                                                    shuffle=False,
                                                    num_workers=1)

        return train_loader, evaluation_loader, num_classes
    else:
        [y_logits_val, y_val, y_logits_test, y_test] = map(tensorify, [y_logits_val, y_val, y_logits_test, y_test])
        val_dataset = data_utils.TensorDataset(y_logits_val, y_val)
        test_dataset = data_utils.TensorDataset(y_logits_test, y_test)
        train_loader = data_utils.DataLoader(val_dataset, batch_size=dataset_config.batch_size,
                                           shuffle=dataset_config.shuffle,
                                           num_workers=dataset_config.num_workers)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=dataset_config.batch_size,
                                            shuffle=False,
                                            num_workers=1)
        return train_loader, test_loader, num_classes



def get_model(model_config, num_classes):
    """
    Construct a calibration model based on the configuration.

    Args:
        model_config: Configuration object with:
            - name (str): Name of the model type
            - params (dict): Model-specific parameters
        num_classes (int): Number of output classes

    Returns:
        A PyTorch nn.Module instance representing the calibration model.
    """
    model_name = model_config.name
    if model_name == 'nonlinear':
        model_config.params.num_classes = num_classes
        model = MonotonicModel_v1(**model_config.params)
    elif model_name == 'pwlinear':
        model_config.params.num_classes = num_classes
        model = PWLinearModel(**model_config.params)
    elif model_name == 'temperature_scaling':
        model = TempScaling()
    elif model_name == 'linear':
        model = EnsembleTempScal(model_config.params.temp_num)
    else:
        raise ValueError("model_name {} is not defined".format(model_name))

    return model


def get_scheduler(learning_rate, epochs, schedulersetting, warmup_epochs, optimizer, batch_num_per_epoch):
    """
    Return a learning rate scheduler for training.

    Currently uses ReduceLROnPlateau scheduler.

    Args:
        schedulersetting (dict): Dictionary with 'lr_reduce_patience'
        optimizer: Optimizer whose learning rate will be scheduled

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau
    """    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=schedulersetting['lr_reduce_patience'], verbose=True)
    return scheduler


def get_optimizer(opt_config, model):
    """
    Create an optimizer for training the model.

    Args:
        opt_config: Configuration with:
            - name (str): Optimizer name ('SGD', 'Adam', 'AdamW')
            - params (dict): Parameters for the optimizer
        model: Model whose parameters will be optimized

    Returns:
        torch.optim.Optimizer instance
    """
    opt_name = opt_config.name
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **opt_config.params)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **opt_config.params)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **opt_config.params)
    else:
        raise ValueError("Optimizer {} is not defined".format(opt_name))
    return optimizer


def get_loss_fn(loss_config, num_classes=100):
    """
    Build a loss function based on configuration.

    Args:
        loss_config: Dictionary with:
            - name (str): Loss name ('cross_entropy', 'eceloss', etc.)
            - params (dict): Additional parameters for loss (if needed)
        num_classes: Number of classes (used for some calibration-specific losses)

    Returns:
        Loss function (nn.Module or callable)
    """
    if loss_config.name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_config.name == 'eceloss':
        loss_fn = utils._ECELoss(n_bins=num_classes)
    elif loss_config.name == "hcalib":
        loss_fn = functools.partial(hcalibration, num_classes=num_classes, **loss_config.params)
    elif loss_config.name == "brier":
        loss_fn = utils._BrierLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_config.name}")        
    return loss_fn



def get_metrics_fn(metric_names, nbins, nclasses, **kwargs):
    """
    Build a dictionary of evaluation metrics based on provided names.

    Args:
        metric_names (list): List of metric identifiers
        nbins (int): Number of bins used in calibration metrics
        nclasses (int): Number of classes
        **kwargs: Additional parameters for metric constructors

    Returns:
        dict: Mapping of metric name to metric instance (callable or nn.Module)
    """    
    metrics = {}
    if 'ECE_ew' in metric_names:
        metrics.update({'ECE_ew': utils._ECELoss(nbins, **kwargs)})
        # Metric: Expected Calibration Error (Equal-width)
        # Originally abbreviated as 'ece' in the original training log.
        
    if 'ACE' in metric_names:
        metrics.update({'ACE': utils._AECELoss(nbins)})
        # Metric: Average Calibration Error
        # Originally abbreviated as 'aECE' in the original training log.

    if 'ECE_r2' in metric_names:
        metrics.update({'ECE_r2': utils._DebiasedECELoss(n_bins=nbins, debiased=False, marginal=False, adaptive=False, power=2.0, nclasses=nclasses )})
        # Corresponds to "15b TCE2(top-label calibration error)" from the paper "Better Uncertainty Calibration via Proper Scores for Classification and Beyond".
        # Code adapted from "Verified Uncertainty Calibration".
        # The implementation involves a square-root as in the original formulation.
        # Originally abbreviated as 'eCE_2' in the original training log.

    if 'dECE' in metric_names:
        metrics.update({'dECE': utils._DebiasedECELoss(n_bins=nbins, debiased=True, marginal=False, adaptive=True, power=1.0, nclasses=nclasses )})
        # Metric: Debiased Adaptive ECE
        # A variant of ECE with debiasing and adaptive binning.
        # Originally abbreviated as 'deadaeCE' in the original training log.

    if 'ECE_em' in metric_names:
        metrics.update({'ECE_em': utils._AdaptiveECELoss()})
        # Metric: Expected Calibration Error (Equal-mass)
        # Originally referred to as 'adaECE' in the original training log.

    if 'KSerror' in metric_names:
        metrics.update({'KSerror':  utils.sKSCE()})
        # Metric: KS Calibration Error (Kolmogorov–Smirnov CE)
        # First introduced in the paper "Calibration of Neural Networks Using Splines",
        # and the original implementation was taken from:
        # https://github.com/kartikgupta-at-anu/spline-calibration/blob/master/cal_metrics/KS.py
        # Later reused and referenced by "Better Uncertainty Calibration via Proper Scores for Classification and Beyond".
        # That paper modified the metric to use squared differences instead of absolute ones,
        # and applied a square root to the final value.
        # However, the original formulation did not include squaring.
        # Therefore, we reverted the modification and replaced the squared difference with absolute error,
        # to match the original method.
        # Originally referred to as 'sKSCE' in the original training log.

    if 'MMCE' in metric_names:
        metrics.update({'MMCE': utils.MMCE()})
        # Metric: Maximum Mean Calibration Error
        # Proposed in "Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings".
        # This PyTorch implementation is adopted from netcal package's reproduction:
        # https://github.com/EFS-OpenSource/calibration-framework/blob/main/netcal/metrics/confidence/MMCE.py
        # Originally referred to as 'MMCE' in the original training log.

    if 'KDE_ECE' in metric_names:
        metrics.update({'KDE_ECE': utils._sKDECE(order=1.0 )})
        # Metric: KDE-based Calibration Error (KDE TCE1) for top-label probability
        # Summarized (not originally proposed) as (KDE TCE1) in "better uncertainty calibration via proper scores for classification and beyond"
        # Implementation further taken from "Mix-n-match: Ensemble and compositional methods for uncertainty calibration in deep learning"
        # It uses kernel density estimation to approximate the expected calibration error (ECE),
        # and performs integration over the absolute difference (i.e., L1 norm), not squared error.
        # Originally referred to as 'kdeECEt_1' in the original training log.

    if 'ECE_s_r2' in metric_names:
        metrics.update({'ECE_s_r2': utils.CalibrationMetric(ce_type='em_ece_sweep', num_bins=nbins, norm=2, multiclass_setting="top_label")})
        # Metric: Equal-Mass Sweep ECE (L2-norm)
        # From "Mitigating Bias in Calibration Error Estimation".
        # Code source: https://github.com/google-research/google-research/tree/master/caltrain
        # The authors recommend this configuration (equal-mass, sweep ece) as suggested choice, thus adopted.
        # Originally referred to as 'sweepECE_2' in the original training log.

    if 'ECE_s_r1' in metric_names:
        metrics.update({'ECE_s_r1': utils.CalibrationMetric(ce_type='em_ece_sweep', num_bins=nbins, norm=1, multiclass_setting="top_label")})
        # Metric: Equal-Mass Sweep ECE (L1-norm)
        # From the same paper and implementation as ECE_s_r2, with norm set to 1 instead of 2.
        # Originally referred to as 'sweepECE_1' in the original training log.

    if 'CWECE_s' in metric_names:
        metrics.update({'CWECE_s': utils._CwECELoss(nbins-1, avg=False)})  # same results as Kull etal.
        # Metric: Class-wise ECE (sum form)
        # Produces results equivalent to the form reported by Kull et al.
        # Originally referred to as 'CwECEsum' in the original training log.

    if 'CWECE_a' in metric_names:
        metrics.update({'CWECE_a': utils._CwECELoss(nbins)})
        # Metric: Class-wise ECE (averaged form)
        # Averages class-wise calibration errors across all classes.
        # Originally referred to as 'CwECE' in the original training log.

    if 'CWECE_r2' in metric_names:
        metrics.update({'CWECE_r2': utils._DebiasedECELoss(n_bins=nbins, debiased=False, marginal=True, adaptive=False, power=2.0, nclasses=nclasses)})
        # Metric: Class-wise Calibration Error (L2 norm with sqrt operation)
        # Corresponds to 15b CWECE2 (15 bin class-wise calibration error) from "Better Uncertainty Calibration via Proper Scores for Classification and Beyond".
        # Also references code from "Verified Uncertainty Calibration".
        # Originally referred to as 'marCE_2' in the original training log.

    if 'tCWECE' in metric_names:
        metrics.update({'tCWECE': utils._eval_in_IMax(nclasses=nclasses, num_bins=nbins, marginal=True, list_approximators=["dECE",])})
        # Metric: Thresholded Class-wise ECE (equal-width binning)
        # Thresholding denoted as cls-prior explained the following paper
        # From: "Multi-class Uncertainty Calibration via Mutual Information Maximization-based Binning".
        # Implementation: https://github.com/boschresearch/imax-calibration/blob/3c0a0c8544545a868205cb8460aea3830e23894d/imax_calib/evaluations/calibration_metrics.py#L36
        # Originally referred to as 'CwECE_thr' in the original training log.

    if 'tCWECE_k' in metric_names:
        metrics.update({'tCWECE_k': utils._eval_in_IMax(nclasses=nclasses, num_bins=nbins, marginal=True, list_approximators=["kECE",])})        
        # Metric: Thresholded Class-wise ECE (k-means binning)
        # From the same paper and repo as tCWECE, but using k-means clustering for bin edges.
        # Originally referred to as 'kCwECE_thr' in the original training log.

    if 'SKCE' in metric_names:
        metrics.update({'SKCE': utils._SKCE()})
        # Metric: Squared Kernel Calibration Error (SKCE)
        # Originally also referred to as 'SKCE' in the original training log.
        # Based on "Calibration Tests in Multi-class Classification: A Unifying Framework".
        # Our appendix provides further explanation on this canonical calibration metric.
        # Python usage: https://github.com/devmotion/pycalibration
        # Other expalantion: https://devmotion.github.io/CalibrationErrors.jl/dev/kce/#footnote-WLZ19

        # Note: The kernel function used is k((p,y),(p',y')) → ℝ, which differs in form from our K(p,p') ∈ ℝ^{L×L}.
        # However, as explained in the later paper "Calibration Tests Beyond Classification",
        # the two forms are mathematically equivalent for classification:
        #  [K(p,p')]_{y,y'} = k((p,y),(p',y')).
        # Thus, this use of the devmotion package is fully compatible with our kernel-based definition.

        # The above default usage of SKCE in this package uses the unbiased estimator, so outputs may be negative.
        # For example: skce = SKCE(tensor(ExponentialKernel(), WhiteKernel())) by default refers to
        # skce = ca.SKCE(ca.tensor(ca.ExponentialKernel(), ca.WhiteKernel()), unbaised=True)
        # For further details, please refers to 
        # [1] "Calibration tests in multi-class classification: A unifying framework"
        # [2] "Calibration tests beyond classification"
        # [3] https://github.com/devmotion/CalibrationErrors.jl/blob/main/src/skce.jl

    if 'DKDE_CE' in metric_names:
        metrics.update({'DKDE_CE': utils._ECEKDE(order=2.0,bandwidths=torch.tensor(1.0) )})
        # Metric: KDE-based Canonical Calibration Error (order=2)
        # Originally also referred to as 'kdeECEc_2_1.00e+00' in the original training log.
        # This metric follows the KDE-ECE formulation summarized in our paper appendix,
        # consistent with the implementation from "A Consistent and Differentiable Lp Canonical Calibration Error Estimator".
        # The bandwidth value (h=1.0) is one of the candidates explicitly listed in the above paper.
        # Although the original code allows adaptive bandwidth selection, a fixed h is preferred for fair comparisons—using dynamic h may be inappropriate.
        # Note: GPU memory usage is highly sensitive to whether logits and labels are detached.
        
        # This metric estimates calibration error over full probability vectors (canonical setting),
        # using kernel density estimation and integrating the squared (L2) error.
        # While the original paper’s formula includes a square root, their code omits it. We follow the code behavior here (i.e., no square root).
        # The metric was cited (not proposed) in "Better Uncertainty Calibration via Proper Scores for Classification and Beyond", abbreviated as KDE CE1.

    if 'NLL' in metric_names:
        metrics.update({'NLL': nn.CrossEntropyLoss()})
        # Metric: Negative Log-Likelihood (standard classification loss)

    return metrics



def get_selectormetrics_op(selector_names):
    """
    Build a dictionary of model selection strategies for given metric names.

    Args:
        selector_names (list): Metric names used for model selection

    Returns:
        dict: Mapping from metric name to selection function (e.g., `min`)
    """    
    ops = {}
    ops.update({'ECE_ew': min}) if 'ECE_ew' in selector_names else None
    ops.update({'ACE': min}) if 'ACE' in selector_names else None
    ops.update({'ECE_r2': min}) if 'ECE_r2' in selector_names else None  # 速度已优化
    ops.update({'dECE': min}) if 'dECE' in selector_names else None   # 速度已优化
    ops.update({'ECE_em': min}) if 'ECE_em' in selector_names else None
    ops.update({'KSerror': min}) if 'KSerror' in selector_names else None
    ops.update({'MMCE': min}) if 'MMCE' in selector_names else None
    ops.update({'KDE_ECE': min}) if 'KDE_ECE' in selector_names else None # 速度已优化
    ops.update({'ECE_s_r2': min}) if 'ECE_s_r2' in selector_names else None
    ops.update({'ECE_s_r1': min}) if 'ECE_s_r1' in selector_names else None
    ops.update({'CWECE_s': min}) if 'CWECE_s' in selector_names else None 
    ops.update({'CWECE_a': min}) if 'CWECE_a' in selector_names else None 
    ops.update({'CWECE_r2': min}) if 'CWECE_r2' in selector_names else None  # 速度已优化
    ops.update({'tCWECE': min}) if 'tCWECE' in selector_names else None 
    ops.update({'tCWECE_k': min}) if 'tCWECE_k' in selector_names else None 
    ops.update({'SKCE': min}) if 'SKCE' in selector_names else None 
    ops.update({'DKDE_CE': min}) if 'DKDE_CE' in selector_names else None 
    ops.update({'NLL': min}) if 'NLL' in selector_names else None 
    return ops

