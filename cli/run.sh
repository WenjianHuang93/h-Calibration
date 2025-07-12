#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
root='expdir_final'

tasks=('CIFAR10/DenseNet40' 'CIFAR10/ResNet110' 'CIFAR10/WideResNet32' \
          'CIFAR100/DenseNet40' 'CIFAR100/ResNet110' 'CIFAR100/WideResNet32' \
          'SVHN/ResNet152_SD' 'BIRDS/ResNet50NTS' 'CARS/ResNet50pre' \
          'CARS/ResNet101' 'CARS/ResNet101pre' 'ImageNet/DenseNet161' \
          'ImageNet/PNASNet5large' 'ImageNet/ResNet152' 'ImageNet/Swintiny')

evaltypes=('TopCalEval' 'NonTopCalEval' 'PSREval')

for task in ${tasks[@]}; do
    for eval in ${evaltypes[@]}; do
        
        exp_dir=$root/$task/$eval
        devnum=`echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l`
        cuda_devices='' && for ((i=0;i<$devnum;i++)); do cuda_devices=$cuda_devices$i','; done
        
        python calibrate.py --exp_dir "$exp_dir" --cuda_devices ${cuda_devices%*,} --calibration_type "$eval"
        python evaluate.py --exp_dir "$exp_dir" --batch_size 200 --calibration_type "$eval"

    done
done



