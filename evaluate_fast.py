"""
Fast model evaluation version for evaluate.py

This script evaluates models using calibration metrics in a highly parallelized and GPU-efficient way.
It is a faster alternative to `evaluate.py` and produces equivalent results.
You can choose it to replace evaluate.py if you are going to evaluate many calibrated models simultaneously.
Otherwise, I recommend to use evaluate.py directly if only few calibrated models are going to be evaluated.

Author: Wenjian Huang
Date: July 4, 2025
"""


import argparse
import logging
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import utils.utils as utils
import json
from easydict import EasyDict
import time as tm
import pickle
import utils.builder as builder
import sys
import subprocess

import multiprocessing
from multiprocessing import Manager
from datetime import datetime
import numpy as np


# ---------------------------- Argument Parsing ---------------------------- #
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


@torch.no_grad()
def modelpredict(model, dataloader, config):
    global nested_eval, eval_metrics
    """
    Perform model inference and return uncalibrated logits, calibrated logits, and labels.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): PyTorch DataLoader for evaluation.
        config (EasyDict): Configuration object with model settings.

    Returns:
        Tuple of Tensors: (uncalibrated_logits, calibrated_logits, labels)
    """

    # set model to evaluation mode
    model.eval()
    all_uncals_logit, all_calibs_logit, all_labels = [], [], []

    for data_batch, labels_batch in dataloader:
        # move to GPU if available
        if config.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        if config.model.name == 'linear':
            calibs_probs = model(data_batch) # the calibrated probability
            calibs_logit = torch.log(calibs_probs+1e-25)
        else:
            calibs_logit = model(data_batch)
            calibs_probs = F.softmax(calibs_logit, dim=1)
        
        all_uncals_logit.append(data_batch)
        all_calibs_logit.append(calibs_logit)
        all_labels.append(labels_batch)

    all_uncals_logit = torch.cat(all_uncals_logit)
    all_calibs_logit = torch.cat(all_calibs_logit)
    all_labels = torch.cat(all_labels).type(torch.long)

    return all_uncals_logit, all_calibs_logit, all_labels


# ---------------------------- Parallel Execution ---------------------------- #
def execute_subprocess(evalargs, core_id, result_queue, index):
    """
    Subprocess execution for evaluating metrics on CPU.

    Args:
        evalargs (tuple): Arguments to pass to the worker.
        core_id (int): CPU core to bind the subprocess to.
        result_queue (Queue): Queue to store subprocess results.
        index (int): Task index.
    """
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        input_data = pickle.dumps(evalargs)
        data_size = len(input_data)
        # Set buffer size to 1.5x of input size, with a minimum of 1MB
        bufsize = max(10**6, int(1.5 * data_size)) 
        process = subprocess.Popen([sys.executable, '-m', 'utils.eval_fast_worker'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=env,
                                   bufsize=bufsize,  # dynamically set buffer size
                                   preexec_fn=lambda: os.sched_setaffinity(0, {core_id}))
        
        # Non-chunked input transfer
        process.stdin.write(input_data)   
        stdout_data, stderr_data = process.communicate()        

        if process.returncode != 0:
            stderr_message = stderr_data.decode().strip()
            if "warning" in stderr_message.lower():
                print(f"Subprocess returned warnings: {stderr_message}")
            else:
                raise Exception(f"Subprocess failed with error: {stderr_message}")

        result = pickle.loads(stdout_data)
        result_queue.put((index, result))

    except Exception as e:
        result_queue.put((index, str(e)))        
  

def get_available_cpus():
    """
    Retrieve the list of available CPU core IDs for process affinity.
    """
    pid = os.getpid()
    affinity = os.sched_getaffinity(pid)
    return list(affinity)


def filter_metrics_by_device(metric_names,exec_backend):
    # Split metrics based on execution backend (CPU or GPU) to enable backend-specific acceleration
    cpu_metrics = ['ECE_em','ACE','dECE','KDE_ECE','CWECE_r2',
                 'ECE_r2','KSerror','ECE_s_r1','ECE_s_r2',
                 'MMCE','tCWECE_k','tCWECE','SKCE']
    gpu_metrics = ['NLL','ECE_ew','CWECE_a','CWECE_s','DKDE_CE']
    if exec_backend=='GPU':
        selected = [metric for metric in metric_names if metric in gpu_metrics]
    elif exec_backend=='CPU':
        selected = [metric for metric in metric_names if metric in cpu_metrics]
    return selected
    

def get_evalmetrics(calibration_type):
    # Choose different evaluation metrics and model selectors based on calibration evaluation type
    if calibration_type == 'TopCalEval':
        eval_metrics = ['ECE_ew','ACE','ECE_r2','dECE','ECE_em','KSerror','MMCE','KDE_ECE','ECE_s_r2','ECE_s_r1']
    elif calibration_type == 'NonTopCalEval':
        eval_metrics = ['CWECE_s','CWECE_a','CWECE_r2','tCWECE','tCWECE_k','SKCE','DKDE_CE']
    elif calibration_type == 'PSREval':
        eval_metrics = ['NLL',]
    elif calibration_type == 'FullEval':
        eval_metrics = ['ECE_ew','ACE','ECE_r2','dECE','ECE_em','KSerror','MMCE','KDE_ECE','ECE_s_r2','ECE_s_r1',
                    'CWECE_s','CWECE_a','CWECE_r2','tCWECE','tCWECE_k','SKCE','DKDE_CE','NLL']
    return eval_metrics
                

def get_pending_tasks(exp_dir,calibration_type):
    # Identify calibration models whose calibration evaluation results are incomplete
    if calibration_type == 'TopCalEval':
        selectors = ['dECE',]
    elif calibration_type == 'NonTopCalEval':
        selectors = ['CWECE_a',]
    elif calibration_type == 'PSREval':
        selectors = ['NLL',]
    elif calibration_type == 'FullEval':
        selectors = ['dECE','CWECE_a','NLL']
    
    eval_metrics = get_evalmetrics(calibration_type)

    completed_models = []
    saving_prefix = 'testing_results_postcalibration_by_selector_'  
    for file in os.listdir(exp_dir):
        if file.startswith(saving_prefix):
            with open(exp_dir + '/' + file, 'r') as f:
                content = json.load(f)
            # If all required metrics are already present, consider this model completed
            if len(list(set(eval_metrics) - set(content.keys()))) == 0:
                completed_models.append(file.replace(saving_prefix,'').replace('.json',''))
    
    # Return the list of model selectors that still need evaluation
    incompleted = list(set(selectors) - set(completed_models))
    return incompleted


def calcu_gpu_metrics(metrics_gpu, logit, labels, config):
    # Calculate evaluation metrics using GPU-executed functions
    metrics = {metric: float(metrics_gpu[metric](logit, labels[:, 0])) for metric in metrics_gpu} 
    # Compute full-data loss depending on loss type
    fulldataloss = loss_fn(torch.softmax(logit, dim=1), labels[:, 0].long()).detach() if config.loss.name == 'hcalib' else loss_fn(logit, labels[:, 0].long()).detach()
    metrics.update({'fulldataloss': float( fulldataloss )})
    return metrics


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """

    # Get available CPU cores for multiprocessing
    available_cpus = get_available_cpus()
    cpu_count = len(available_cpus)

    args = parser.parse_args()
    
    # Load experiment configuration
    json_path = os.path.join(args.exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    with open(json_path, 'r') as f:
        config = EasyDict(json.load(f))
    
    # Override batch size if provided in command line
    if hasattr(config.dataset, "batch_size") and args.batch_size is not None:
        config.dataset.update({"batch_size": args.batch_size})
    
    # Enable CUDA if available
    config.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed = 1357 if 'seed' not in config else config.seed
    torch.manual_seed(seed)
    if config.cuda:
        torch.cuda.manual_seed(seed)
    utils.fix_rng_seed(seed)
    
    # Set up logger
    evalog = os.path.join(args.exp_dir, 'evaluate.log')
    if os.path.exists(evalog):
        with open(evalog,'r') as f:
            content = f.read()
    else:
        content = '' # No previous log content

    # Get the logger
    utils.set_logger(os.path.join(args.exp_dir, 'evaluate.log'))
    logging.info("Working folder: " + args.exp_dir)

    # fetch dataloaders
    val_loader, test_loader, num_classes = builder.get_dataloaders(config.dataset, is_training=False, verbose=config.verbose)
    config.num_classes = num_classes

    # Load model and loss function
    model = builder.get_model(config.model, num_classes)
    model = model.cuda() if config.cuda else model
    loss_fn = builder.get_loss_fn(config.loss, num_classes=num_classes)
    
    # Select metric names based on evaluation mode
    metric_names = get_evalmetrics(args.calibration_type)
    metric_names_gpu = filter_metrics_by_device(metric_names,'GPU')
    metric_names_cpu = filter_metrics_by_device(metric_names,'CPU')   
    eval_metrics_gpu = builder.get_metrics_fn(metric_names_gpu, config.nbins, config.num_classes)  

    while True:
        # Check if all model evaluations are completed
        undoeval = get_pending_tasks(args.exp_dir,args.calibration_type)
        if len(undoeval)==0:
            # All tasks completed
            print('done!')
            sys.exit(0)
        undoeval.sort()

        PoolOutput, PoolInput = [], []
        MetricsGPU = [] if config.cuda else None

        for selectmodel in undoeval:
            if len(PoolOutput)>= cpu_count*2:
                # To avoid OOM, limit batch to 2×CPU count
                break
            
            # Load selected calibration model checkpoint
            modelid = os.path.join(args.exp_dir, 'selected_ckpts','selector_{}.pth.tar'.format(selectmodel))
            utils.load_checkpoint(modelid, model)
            
            # Perform inference to get pre/post logits and labels
            uncals_logit, calibs_logit, labels = modelpredict(model, test_loader, config)    
            
            if args.save_logits:
                suffix = "_selector_{}".format(selectmodel)
                np.save(os.path.join(args.exp_dir,'calibration_outputs', 'pre_logits'+ suffix + '.npy'), 
                        uncals_logit.detach().cpu().numpy())
                np.save(os.path.join(args.exp_dir,'calibration_outputs', 'post_logits'+ suffix + '.npy'), 
                        calibs_logit.detach().cpu().numpy())
                np.save(os.path.join(args.exp_dir,'calibration_outputs', 'labels'+ suffix + '.npy'), 
                        labels.detach().cpu().numpy())

            # Prepare post-calibration result saving path
            PoolOutput.append(os.path.join(args.exp_dir, "testing_results_postcalibration_by_selector_{}.json".format(selectmodel)))

            # Compute GPU metrics (if applicable), prepare inputs for CPU metrics
            if config.cuda:
                MetricsGPU.append( calcu_gpu_metrics(eval_metrics_gpu, calibs_logit, labels, config)  )      
                PoolInput.append((calibs_logit.detach().cpu(), labels.detach().cpu(), metric_names_cpu, loss_fn, config, False))
            else:
                PoolInput.append((calibs_logit.detach().cpu(), labels.detach().cpu(), metric_names, loss_fn, config, True))

            # Also evaluate pre-calibration if not yet done
            precal_file = os.path.join(args.exp_dir, "testing_results_precalibration.json")
            if (not os.path.exists(precal_file)) and (precal_file not in PoolOutput):
                PoolOutput.append(precal_file)
                if config.cuda:
                    MetricsGPU.append( calcu_gpu_metrics(eval_metrics_gpu, uncals_logit, labels, config) )      
                    PoolInput.append((uncals_logit.detach().cpu(), labels.detach().cpu(), metric_names_cpu, loss_fn, config, False))
                else:
                    PoolInput.append((uncals_logit.detach().cpu(), labels.detach().cpu(), metric_names, loss_fn, config, True))

            print( str(datetime.now()) + ': ' + modelid + ' model inference is finished !')
                                    
        
        # Parallel evaluation logic
        if len(PoolInput)>0: # If CPU metrics need evaluation or no GPU is available
            assert len(PoolInput) == len(PoolOutput)
            
            # Create a manager and result queue for multiprocessing
            manager = Manager()
            result_queue = manager.Queue()

            # Limit number of concurrent subprocesses per CPU core
            max_workers = cpu_count * 1 # Set to 2 or 3 for more concurrency if memory allows
            print('cpu_count: ' + str(cpu_count))
            print('available_cpus: ' + str(available_cpus))

            processes = []
            tasks = [(evalargs, i) for i, evalargs in enumerate(PoolInput)]
            
            # Initialize subprocesses up to max_workers
            for i in range(min(max_workers, len(tasks))):
                evalargs, index = tasks.pop(0)
                core_id = available_cpus[i % cpu_count]
                p = multiprocessing.Process(target=execute_subprocess, args=(evalargs, core_id, result_queue, index))
                p.start()
                processes.append((p, core_id))

            # Manage processes. Monitor and restart completed slots with remaining tasks
            while tasks or any(p.is_alive() for p, _ in processes):
                for p, core_id in list(processes):
                    if not p.is_alive():
                        processes.remove((p, core_id))
                        if tasks:
                            evalargs, index = tasks.pop(0)
                            new_process = multiprocessing.Process(target=execute_subprocess, args=(evalargs, core_id, result_queue, index))
                            new_process.start()
                            processes.append((new_process, core_id))
                tm.sleep(0.1)  # Avoid tight loop and reduce CPU usage
            
            # Wait for all to finish
            for p, _ in processes:
                p.join()

            # Collect results from the queue
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Sort results by index to ensure output order matches input
            results.sort(key=lambda x: x[0])
            MetricsCPU = [result[1] for result in results]
            
            print( str(datetime.now()) + ': cpu metrics eval finished !')


        # Final results processing and logging
        for i in range(len(PoolOutput)):
            savingdir = PoolOutput[i]

            # Identify model for logging
            if 'postcalibration' in savingdir:
                selectmodel = savingdir.split('/')[-1].replace('testing_results_postcalibration_by_selector_','').replace('.json','')
                logging.info("-------- Evaluation for selectmodel : {} --------".format(selectmodel))
            else:
                logging.info("-------- Evaluation for pre calibration --------")
            
            # Combine metrics from GPU and CPU
            if MetricsGPU is not None:
                metrics_gpu = MetricsGPU[i]
            else:
                metrics_gpu = {}

            if 'MetricsCPU' in locals():
                if isinstance(MetricsCPU[i],str):
                    logging.info('Error Raised by the Prarallel Computing Processes, with Error: ' + MetricsCPU[i])            
                    continue
                metrics_cpu = MetricsCPU[i]
            else:
                metrics_cpu = {}
            
            # Format metrics nicely for logging
            metrics = {**metrics_gpu,**metrics_cpu}

            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) if v>=1e-3 else "{}: {:.3e}".format(k, v) for k, v in metrics.items())
            logging.info("- Eval metrics: " + metrics_string)
            
            # Save metrics to JSON (append mode for safe updates)
            utils.save_dict_to_json(metrics, savingdir, append=True)

    


