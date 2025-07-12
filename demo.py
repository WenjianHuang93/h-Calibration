"""
Demo Script: Calibration and Evaluation Workflow

Author: Wenjian
Email: wjhuang@pku.edu.cn
Date: June, 2025
License: MIT
"""


import subprocess
import os


### ==== Set GPU devices for training/calibration ====
# To use a specific GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# CUDA_VISIBLE_DEVICES = "2"

# To use multiple GPUs:
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
CUDA_VISIBLE_DEVICES = "0,1,2,3"

# Generate a comma-separated list of device indices based on the environment setting
if CUDA_VISIBLE_DEVICES=='':
    cuda_devices = CUDA_VISIBLE_DEVICES
else:
    cuda_devices = ','.join([str(id) for id in range(len(CUDA_VISIBLE_DEVICES.split(',')))])


### ==== Define experiment directory for the task ====
exp_dir = './expdir_final/CIFAR10/WideResNet32/TopCalEval'

### ==== Train the calibration model ====
code = './calibrate.py'
returncode = subprocess.call(['python',code,'--exp_dir',exp_dir, '--cuda_devices', cuda_devices, '--calibration_type', 'TopCalEval'])

### ==== Evaluate the calibration model ====

code = './evaluate.py'
returncode = subprocess.call(['python',code,'--exp_dir', exp_dir, '--calibration_type', 'TopCalEval'])

# Optionally, you can use evaluate_fast.py for parallel evaluation using subprocesses.
# This method provides speed-up when evaluating a large number of models in parallel.
# However, when only a few calibration models are involved, the speed gain is limited.
# For typical use cases, I recommend using evaluate.py.
code = './evaluate_fast.py'
returncode = subprocess.call(['python',code,'--exp_dir', exp_dir, '--calibration_type', 'TopCalEval'])


### ==== Other Example: Non-Top Label Calibration ====

exp_dir = './expdir_final/CIFAR10/WideResNet32/NonTopCalEval'

returncode = subprocess.call(['python','./calibrate.py','--exp_dir',exp_dir, '--cuda_devices', cuda_devices, '--calibration_type', 'NonTopCalEval'])

returncode = subprocess.call(['python','./evaluate.py','--exp_dir', exp_dir, '--calibration_type', 'NonTopCalEval'])

### ==== Other Example: Full Experiment Calibration ====

# The above `expdir_final` examples use automatically selected calibration mapping.
# That is, the mapping with the smallest selector error (e.g., dECE for top-label,
# CWECE_a for non-top-label) is automatically chosen based on the performance on
# the validation set (typically the training set). 

# For a full experiment, we need to run all candidate mapping parameters and
# then perform automatic mapping selection. The following is an example of calibrating
# a specific mapping parameter in such a full evaluation setup.

exp_dir = './expdir_full/CIFAR10/WideResNet32/LinearMap_tempnum_16'

returncode = subprocess.call(['python','./calibrate.py','--exp_dir',exp_dir, '--cuda_devices', cuda_devices, '--calibration_type', 'FullEval'])

returncode = subprocess.call(['python','./evaluate.py','--exp_dir', exp_dir, '--calibration_type', 'FullEval'])
