"""
Model evaluation for already trained post-hoc calibrator

This script evaluates a trained neural network model on a test dataset.
It supports multiple calibration evaluation metrics, 
saves logits, and logs pre- and post-calibration performance.

Author: Wenjian Huang
Date: July 4, 2025
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils.utils as utils
import json
from easydict import EasyDict

import utils.builder as builder

from concurrent.futures import ThreadPoolExecutor


# Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='experiments/base_model',
                    help="Directory containing config.json")
parser.add_argument('--save_logits', default=True, type=bool, help='save logits?') # if saving the logits, the logits will be saved in the exp_dir

# you can manually set smaller inference batch_size if the default batch_size in configuration json exceeds your gpu memory limit. 
# It would not change the evaluation metrics since the evaluation is conducted dataset wise after inference.
parser.add_argument('--batch_size', type=int, default=5000, help="specify the batch size for inference") 

parser.add_argument('--calibration_type', type=str, default='FullEval',
        help="please choose calibration type: TopCalEval, NonTopCalEval, PSREval, or FullEval")
# TopCalEval is for top-label calibration metric evaluation; NonTopEval is for classwise calibration or canonical calibration metric evaluation.
# PSREval is for psr-based calibration evaluation; different choices will use different model selection metrics, as well as different type of evaluators.
# FullEval combine all three model selection metrics and evaluation metrics.

# Thread pool size for parallel metric evaluation
EvalPoolSize = 5

@torch.no_grad()
def nested_eval(metrics_eval, score_eval, label_eval):
    """Wraps a metric evaluation for use in a thread pool."""
    return metrics_eval(score_eval, label_eval)


@torch.no_grad()
def parallel_eval(eval_metrics, scores, labels):
    """
    Parallel evaluation of calibration metrics. Certain metrics are not safely parallelizable as they contain parallel computation already,
    so they are computed serially.
    """    
    unparallel = ['dECE','CWECE_r2','ECE_r2','tCWECE_k','tCWECE','SKCE']
    parallel = [key for key in list(eval_metrics.keys()) if key not in unparallel]
    
    with ThreadPoolExecutor(max_workers=EvalPoolSize) as executor:
        futures = [executor.submit(nested_eval, eval_metrics[metric], scores, labels) for metric in parallel]
        post_metrics = [future.result() for future in futures]
    post_metrics = { parallel[i]: float(post_metrics[i]) for i in range(len(parallel)) }
    
    post_metrics.update({metric: float(eval_metrics[metric](scores, labels)) for metric in eval_metrics.keys() if metric in unparallel})
    return post_metrics


@torch.no_grad()
def evaluate(model, loss_fn, dataloader, eval_metrics, config, save_logits=False, exp_dir=None, save_suffix=None, return_logit=False, pre_metrics=None, iscalloss=True, IsMetricPool=False):
    global nested_eval
    """
    Evaluates the given model on the dataset using specified metrics and loss function.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loss_fn (function): Loss function.
        dataloader (DataLoader): DataLoader providing input and labels.
        eval_metrics (dict): Dictionary of metric functions.
        config (EasyDict): Configuration parameters.
        save_logits (bool): Whether to save logits to disk.
        exp_dir (str): Directory to save logits and results.
        save_suffix (str): Optional suffix to distinguish saved logits.
        return_logit (bool): Whether to return logits along with metrics.
        pre_metrics (dict): Pre-calculated metrics to avoid redundancy.
        iscalloss (bool): Whether to compute and log the full-dataset loss.
        IsMetricPool (bool): Whether to use parallel evaluation of metrics.

    Returns:
        (dict, dict): Pre- and post-calibration metrics.
    """
    

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    all_uncals_logit = []
    all_calibs_probs, all_calibs_logit = [], []
    all_labels = []

    for data_batch, labels_batch in dataloader:
        # move to GPU if available
        if config.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        if config.model.name == 'linear':
            calibs_probs = model(data_batch) # here the lossinput is the calibrated probability
            calibs_logit = torch.log(calibs_probs+1e-25)
        else:
            calibs_logit = model(data_batch) # here the lossinput is the calibrated logit
            calibs_probs = F.softmax(calibs_logit, dim=1)
        
        # append inference results
        all_uncals_logit.append(data_batch)
        all_calibs_logit.append(calibs_logit)
        all_calibs_probs.append(calibs_probs)
        all_labels.append(labels_batch)
    
    # merge all batch-wise prediction results
    all_uncals_logit = torch.cat(all_uncals_logit)
    all_uncals_probs = F.softmax(all_uncals_logit, dim=1)

    all_calibs_logit = torch.cat(all_calibs_logit)
    all_calibs_probs = torch.cat(all_calibs_probs)
    
    all_labels = torch.cat(all_labels).type(torch.long)
    
    # save prediction if needed
    if save_logits and exp_dir is not None:
        suffix = ''
        if save_suffix is not None:
            suffix = '_' + save_suffix

        np.save(os.path.join(exp_dir, 'pre_logits'+ suffix + '.npy'), all_uncals_logit.detach().cpu().numpy())
        np.save(os.path.join(exp_dir, 'post_logits'+ suffix + '.npy'), all_calibs_logit.detach().cpu().numpy())
        np.save(os.path.join(exp_dir, 'labels'+ suffix + '.npy'), all_labels.detach().cpu().numpy())
    
    # if uncalibrated data evaluation exists, pass to save computational cost
    if (pre_metrics is None):
        pre_metrics = {metric: float(eval_metrics[metric](all_uncals_logit, all_labels[:, 0])) for metric in eval_metrics}
        fulldataloss = loss_fn(all_uncals_probs, all_labels[:, 0].long()).detach() if config.loss.name == 'hcalib' else loss_fn(all_uncals_logit, all_labels[:, 0].long()).detach()
        pre_metrics.update({'fulldataloss': float(fulldataloss)})

    # if apply parallel computation for metric evaluation
    if IsMetricPool==False:
        post_metrics = {metric: float(eval_metrics[metric](all_calibs_logit, all_labels[:, 0])) for metric in eval_metrics}
    else:
        post_metrics = parallel_eval(eval_metrics, all_calibs_logit, all_labels[:,0])
    
    # if dataset wise loss is required
    if iscalloss:
        fulldataloss = loss_fn(all_calibs_probs, all_labels[:, 0].long()).detach() if config.loss.name == 'hcalib' else loss_fn(all_calibs_logit, all_labels[:, 0].long()).detach()
        post_metrics.update({'fulldataloss': float(fulldataloss)})
    
    # logging the evaluation results
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) if v>=1e-3 else "{}: {:.3e}".format(k, v) for k, v in pre_metrics.items())
    logging.info("- Eval metrics pre-calibration: " + metrics_string)
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) if v>=1e-3 else "{}: {:.3e}".format(k, v) for k, v in post_metrics.items())
    logging.info("- Eval metrics post-calibration: " + metrics_string)
    
    if return_logit is True:
        return pre_metrics, post_metrics, {'logits': all_uncals_logit, 'logits_calib': all_calibs_logit, 'target': all_labels}

    return pre_metrics, post_metrics
    
    


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """

    global args, config
    args = parser.parse_args()
    json_path = os.path.join(args.exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path, 'r') as f:
        config = EasyDict(json.load(f))

    if hasattr(config.dataset, "batch_size") and args.batch_size is not None:
        config.dataset.update({"batch_size": args.batch_size})
    
    # use GPU if available
    config.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed = 1357 if 'seed' not in config else config.seed
    torch.manual_seed(1357)
    if config.cuda:
        torch.cuda.manual_seed(1357)
    utils.fix_rng_seed(seed)
    
    # initialize the evaluation log file
    assert os.path.exists(args.exp_dir)
    evalog = os.path.join(args.exp_dir, 'evaluate.log')
    if os.path.exists(evalog):
        with open(evalog,'r') as f:
            content = f.read()
    else:
        content = ''


    # Get the logger
    utils.set_logger(os.path.join(args.exp_dir, 'evaluate.log'))
    
    logging.info("Start evaluation...")

    # fetch dataloaders
    val_loader, test_loader, num_classes = builder.get_dataloaders(config.dataset, is_training=False, verbose=config.verbose)
    config.num_classes = num_classes


    IsMetricPool = False # when set to True, it doesn't speed up to much in practice and may cause higher memory usage or errors.
    AppendLog = False # when set it false, the saved json is updated by difference other than rewritten totally.
    SaveGPUmem = False # when set it true, gpu memory is less but inference speed is much slower.
    
    # fetch different selected models for different types of model evaluations
    if args.calibration_type == 'TopCalEval':
        selectors = ['dECE',]
        eval_metrics = ['ECE_ew','ACE','ECE_r2','dECE','ECE_em','KSerror','MMCE','KDE_ECE','ECE_s_r2','ECE_s_r1']
    elif args.calibration_type == 'NonTopCalEval':
        selectors = ['CWECE_a',]
        eval_metrics = ['CWECE_s','CWECE_a','CWECE_r2','tCWECE','tCWECE_k','SKCE','DKDE_CE']
    elif args.calibration_type == 'PSREval':
        selectors = ['NLL',]
        eval_metrics = ['NLL',]
    elif args.calibration_type == 'FullEval':
        selectors = ['dECE','CWECE_a','NLL']
        eval_metrics = ['ECE_ew','ACE','ECE_r2','dECE','ECE_em','KSerror','MMCE','KDE_ECE','ECE_s_r2','ECE_s_r1',
                    'CWECE_s','CWECE_a','CWECE_r2','tCWECE','tCWECE_k','SKCE','DKDE_CE','NLL']

    eval_metrics = builder.get_metrics_fn(eval_metrics, config.nbins, config.num_classes)  
    
    # DKDE_CE enables a gpu memory saving option. UseCPU means computation with cpu, reducing gpu memory but increasing running time.
    if SaveGPUmem:
        eval_metrics['DKDE_CE'].UseCPU = True
    
    pre_metrics_valset = None 
    pre_metrics_testset = None 
    
    loss_fn = builder.get_loss_fn(config.loss, num_classes=num_classes)

    for selector in selectors:
        # start evaluation        
        logging.info("-------- selection metric_name: {} --------".format(selector))

        model = builder.get_model(config.model, num_classes)
        model = model.cuda() if config.cuda else model

        # Reload weights from the saved file
        utils.load_checkpoint(os.path.join(args.exp_dir, 'selected_ckpts', 'selector_{}.pth.tar'.format(selector)), model)
        
        # make logit saving folder
        if args.save_logits:
            save_suffix = "selector_{}".format(selector)
            savelogitsroot = os.path.join(args.exp_dir,'calibration_outputs')
            if not os.path.exists(savelogitsroot):
                os.makedirs(savelogitsroot)     
        
        # evaluate
        pre_metrics_testset, post_metrics_testset = evaluate(model, loss_fn, test_loader, eval_metrics, config, args.save_logits, savelogitsroot, save_suffix=save_suffix, pre_metrics=pre_metrics_testset, IsMetricPool=IsMetricPool)
        
        # save result json file
        save_path = os.path.join(args.exp_dir, "testing_results_precalibration.json")
        if not os.path.exists(save_path):
            utils.save_dict_to_json(pre_metrics_testset, save_path, append=AppendLog) # append 在原来的记录里面添加新的记录

        save_path = os.path.join(args.exp_dir, "testing_results_postcalibration_by_selector_{}.json".format(selector))
        utils.save_dict_to_json(post_metrics_testset, save_path, append=AppendLog)
            





