"""
Model Evaluation Utility

This script serves as a subprocess module invoked by `evaluate_fast.py` for fast calibration evaluation
of multiple calibrated models. It is designed to support parallel evaluation via multiprocessing
and inter-process communication.

Author: Wenjian Huang
"""


import sys
import pickle
from utils.builder import get_metrics_fn
import torch
import warnings


def evaluatemetrics(evalargs):    
    """
    Computes evaluation metrics for a given model output.

    Args:
        evalargs (tuple): A tuple containing:
            - logits (torch.Tensor): The raw model output (pre-softmax).
            - labels (torch.Tensor): Ground truth labels with shape [N, 1].
            - metric_names (List[str]): List of metric names to compute.
            - loss_fn (Callable): Loss function used to compute overall loss.
            - config (object): Configuration object containing attributes like `nbins`, `num_classes`, and `loss`.
            - iscalloss (bool): Flag indicating whether to compute the loss function value.

    Returns:
        dict: A dictionary of computed metric values, including 'fulldataloss' if `iscalloss` is True.
    """    
    logits, labels, metric_names, loss_fn, config, iscalloss = evalargs
    
    # Construct metric functions based on provided metric names
    eval_metrics = get_metrics_fn(metric_names, config.nbins, config.num_classes)  

    # Compute specified metrics
    get_metrics = {metric: float(eval_metrics[metric](logits, labels[:, 0])) for metric in eval_metrics}
    
    # Optionally compute the full dataset loss
    if iscalloss:
        if config.loss.name == 'hcalib':
            # Apply softmax before computing H-Calibration loss
            get_metrics.update({'fulldataloss': float(       loss_fn( torch.softmax(logits, dim=1),  labels[:, 0].long() ).detach()               )})
        else:
            get_metrics.update({'fulldataloss': float(       loss_fn( logits, labels[:, 0].long()).detach()                )})
    
    return get_metrics


def run_evaluation():
    """
    Reads evaluation arguments from stdin, computes metrics, and writes results to stdout.

    This function is designed to be run in a subprocess and uses pickle for data serialization
    to ensure compatibility with multiprocessing environments.
    """    
    input_data = sys.stdin.buffer.read()
    evalargs = pickle.loads(input_data)
    result = evaluatemetrics(evalargs)

    output_data = pickle.dumps(result)
    sys.stdout.buffer.write(output_data)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    # Format Python warnings to appear on a single line for cleaner stderr output
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return f'{filename}:{lineno}: {category.__name__}: {message}\n'

    warnings.formatwarning = warning_on_one_line
    try:
        run_evaluation()
    except Warning as w:
        warnings.warn(str(w))
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

