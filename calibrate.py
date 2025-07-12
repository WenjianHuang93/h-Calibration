"""
Training: Model Calibration via H-Calibration

This script performs post-hoc calibration of a neural network model using h-calibration objectives,
as specified in a configuration file.

Author: Wenjian Huang

Main Functionalities:
- Load experiment configurations from JSON
- Build model, optimizer, scheduler, and dataloaders
- Train the calibrator on validation data using a chosen calibration loss
- Evaluate calibrated models using calibration metrics during training
- Save intermediate and best-performing models based on selected criteria
- Optionally apply early stopping

Example usage:
python calibrate.py --exp_dir path/to/exp_dir/with/config.json --cuda_devices 0 --calibration_type 'FullEval'
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import utils.utils as utils

from evaluate import evaluate
from easydict import EasyDict
import json
import utils.builder as builder
from collections import defaultdict

# Argument parser for command-line configuration
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp_dir/base',help="Directory containing config.json")
parser.add_argument('--cuda_devices', type=str, default='', help="index of gpus to use")
parser.add_argument('--batch_size', type=int, default=None, help="")

parser.add_argument('--calibration_type', type=str, default='FullEval',
        help="please choose calibration type: TopCalEval, NonTopCalEval, PSREval, or FullEval")
# TopCalEval is for top-label calibration; NonTopEval is for classwise calibration or canonical calibration
# PSREval is for psr-based calibration evaluation; different choices will use different model selection metrics
# FullEval combine all three model selection metrics


# Core calibration training loop
def calibrate(model, optimizer, loss_fn, dataloader, config, IsLossPool=False):
    """
    Trains the calibration model using the specified loss function.

    Args:
        model: nn.Module, calibration model
        optimizer: optimizer for model parameters
        loss_fn: loss function used for training
        dataloader: DataLoader providing input data
        config: configuration object
        IsLossPool: whether to parallelize loss computation (default: False)
    """

    # set model to training mode
    model.train()

    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if config.cuda:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

            # compute model output
            if config.model.name == 'linear':
                probas = model(train_batch) 
                logits = torch.log(probas+1e-25)
            else:
                logits = model(train_batch) 
                probas = F.softmax(logits, dim=1)
            
            loss = loss_fn(probas, labels_batch[:, 0].long(), IsLossPool=IsLossPool, lossweight=config.loss.params.lossweight) if config.loss.name == "hcalib" else loss_fn(logits, labels_batch[:, 0].long())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            optimizer.step()

            loss_val = loss.squeeze().data.cpu().numpy()
            # update the average loss
            loss_avg.update(loss_val)

            t.set_postfix(loss='{:05.5f}'.format(loss_avg()))
            t.update()



def calibrate_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, monitor_metrics, monitor_metrcis_op, config, exp_dir, selectors, 
                           scheduler=None, IsMetricPool=False, IsLossPool=False):
    """
    Performs training and evaluation of the calibrator across epochs.

    Args:
        model: calibration model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: optimizer instance
        loss_fn: loss function for calibration
        monitor_metrics: list of metric functions monitored during training
        monitor_op: dict mapping metric name to min/max
        config: experiment configuration
        exp_dir: path to save checkpoints and logs
        selectors: list of metrics used for model selection
        scheduler: learning rate scheduler (optional)
        IsMetricPool: whether to parallelize metric evaluation
        IsLossPool: whether to parallelize loss computation
    """

    global early_stopping_metric

    fn_to_val = {min: np.inf, max: -np.inf}
    best_metric_vals = {metric: fn_to_val[monitor_metrcis_op[metric]] for metric in monitor_metrics}

    if 'early_stopping' in config.calibrate:
        early_stopping = utils.EarlyStopping(patience=config.calibrate.early_stopping.patience,delta=0)

    all_metrics = defaultdict(list)
    
    pre_metrics = None # save this metric to avoid repeat evaluation for time saving
    
    for epoch in range(config.calibrate.num_epochs):
        logging.info("Epoch {}/{}".format(epoch+1, config.calibrate.num_epochs))

        # Training for one epoch (one full pass over the training set)
        calibrate(model, optimizer, loss_fn, train_dataloader, config, IsLossPool=IsLossPool)
        
        # Evaluate for one epoch on validation set (actually, in our setting validation set is training set.)
        pre_metrics, post_metrics = evaluate(model, loss_fn, val_dataloader, monitor_metrics, config, pre_metrics = pre_metrics, IsMetricPool=IsMetricPool)
        
        if scheduler is not None:
            scheduler.step(post_metrics[early_stopping_metric])
        
        # save the checkpoints
        if type(model) is nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        checkpointroot = exp_dir + '/all_ckpts'
        if not os.path.exists(checkpointroot):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpointroot))
            os.makedirs(checkpointroot)   
        filepath = os.path.join(checkpointroot, 'epoch_'+ str(epoch+1) + '.pth.tar')
        state = {'epoch': epoch + 1, 'state_dict': state_dict, 'optim_dict': optimizer.state_dict()}
        torch.save(state, filepath)

        # Track and update best metrics
        for metric in monitor_metrics:
            all_metrics[metric].append(post_metrics[metric])
            if monitor_metrcis_op[metric] == min:
                is_best = post_metrics[metric] <= best_metric_vals[metric]
            else:
                is_best = post_metrics[metric] >= best_metric_vals[metric]
            
            if is_best:
                logging.info(" - Found new best {}".format(metric))
                best_metric_vals[metric] = post_metrics[metric]

                if metric in selectors:
                    checkpointroot = exp_dir + '/selected_ckpts'
                    if not os.path.exists(checkpointroot):
                        print("Checkpoint Directory does not exist! Making directory {}".format(checkpointroot))
                        os.makedirs(checkpointroot)   
                    utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': state_dict, 'optim_dict': optimizer.state_dict()}, savepath=checkpointroot, name=metric)

        # See if early stopping
        if 'early_stopping' in config.calibrate:
            early_stopping(post_metrics[early_stopping_metric], model)
            if early_stopping.early_stop:
                logging.info("EARLY STOPPING")
                break
    utils.save_dict_to_json(all_metrics, os.path.join(exp_dir, "training_monitored_metrics.json"))



if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    if args.cuda_devices != '':
        args.cuda_devices = [int(s) for s in str.split(args.cuda_devices, ',')]
    else:
        args.cuda_devices = []
    json_path = os.path.join(args.exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path, 'r') as f:
        config = EasyDict(json.load(f))

    # if other batch_size is given, replace the orginal default batch_size configuration
    if hasattr(config.dataset, "batch_size") and args.batch_size is not None:
        config.dataset.update({"batch_size": args.batch_size})
    
    # Initialize the nonlinear monotonic mapping
    if config.model.name == 'nonlinear':
        config.model.params.update({"add_condition_to_integrand":False,"conditioned":False})

    config.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed = 1357 if 'seed' not in config else config.seed
    torch.manual_seed(seed)
    if config.cuda:
        torch.cuda.manual_seed(seed)
    utils.fix_rng_seed(seed)

    # Set the logger
    utils.set_logger(os.path.join(args.exp_dir, 'calibrate.log'))
    logging.info(args.exp_dir)

    # Create the input data pipeline
    logging.info("Training, loading the dataset ...")

    # fetch dataloaders
    val_loader, test_loader, num_classes = builder.get_dataloaders(config.dataset, is_training=True, verbose=config.verbose)
    config.num_classes = num_classes
    
    # parallel processing pool for fast calculation of loss or metric
    IsLossPool, IsMetricPool = False, False
    

    # Register the model selector based on different types of calibraton evaluation
    global early_stopping_metric
    early_stopping_metric = 'ECE_ew'

    if args.calibration_type == 'TopCalEval':
        model_selection_metric = ['dECE',]
    elif args.calibration_type == 'NonTopCalEval':
        model_selection_metric = ['CWECE_a',]
    elif args.calibration_type == 'PSREval':
        model_selection_metric = ['NLL',]
    elif args.calibration_type == 'FullEval':
        model_selection_metric = ['dECE','CWECE_a','NLL']
    monitor_metrics = [early_stopping_metric] + model_selection_metric
    
    monitor_metrcis_op = builder.get_selectormetrics_op(monitor_metrics)
    monitor_metrics = builder.get_metrics_fn(monitor_metrics, config.nbins, config.num_classes)     
    selectors = model_selection_metric


    # # Define the model and optimizer
    model = builder.get_model(config.model, num_classes)
    # try:
    #     # compiled model for torch 2.0 if available
    #     # however, in such case, we have to rewrite the calibration mapping class to avoid device mismatch
    #     # we choose to disable this for simplicity
    #     model = torch.compile(model)
    # except:
    #     model = model

    
    # initialize some adaptive hyperparameters for nonlinear mapping based calibrator
    if config.model.name == 'nonlinear':
        logitdata = torch.stack([val_loader.dataset[i][0] for i in range(len(val_loader.dataset))])
        model.meanlogit[0] = (logitdata-torch.logsumexp(logitdata,dim=1,keepdim=True)).flatten().mean().item()
        model.logitstd[0] = (logitdata-torch.logsumexp(logitdata,dim=1,keepdim=True)).flatten().std().item()

    # if using multiple gpus for training
    if len(args.cuda_devices) > 1:
        model = nn.DataParallel(model.cuda(), device_ids=args.cuda_devices)
    elif len(args.cuda_devices)==1:
        model = model.cuda() if config.cuda else model

    # get the optimizer
    optimizer = builder.get_optimizer(config.optimizer, model)
    
    # get the scheduler
    scheduler = builder.get_scheduler(
        config.optimizer.params.lr, 
        config.calibrate.num_epochs, 
        config.calibrate.scheduler,
        warmup_epochs=0,
        optimizer=optimizer, 
        batch_num_per_epoch=1)
    
    # fetch loss function and metrics
    loss_fn = builder.get_loss_fn(config.loss, num_classes=num_classes)  # here the loss is configurated by the config.loss.params
    

    # Train the model
    logging.info("Starting calibration ... ")

    calibrate_and_evaluate(model, val_loader, test_loader, optimizer, loss_fn, monitor_metrics, monitor_metrcis_op, 
                                            config, args.exp_dir, selectors, scheduler=scheduler, IsMetricPool=IsMetricPool, IsLossPool=IsLossPool)
    
    logging.info("End of training ... ")
