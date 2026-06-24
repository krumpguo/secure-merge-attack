import os

import numpy as np
import torch.nn as nn
import torch

import argparse
import time
import sys
sys.path.append('/data/kuangpu.guo/Secure_merge/attack/')

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model

def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# 获取当前时间戳
times_begin = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"************************************开始时间: {times_begin}*************************************")
args = parse_arguments()
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
args.model = 'ViT-B-32'
# args.model = 'ViT-L-14'
args.device_num = 5
model = args.model
args.data_location = '/data/kuangpu.guo/Secure_merge/attack/data'
args.save = '/data/kuangpu.guo/Secure_merge/attack/checkpoints/' + model
args.logs_path = '/data/kuangpu.guo/Secure_merge/attack/logs/' + model
pretrained_checkpoint = '/data/kuangpu.guo/Secure_merge/attack/checkpoints/'+model+'/zeroshot.pt'
args.device  = f'cuda:{args.device_num}'
args.batch_size = 128
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '/data/kuangpu.guo/Secure_merge/attack/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

task_vector_sum = sum(task_vectors)

# scaling_coef_s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
scaling_coef_s = [0.2]
# 存储结果的列表
results = []

for scaling_coef_ in scaling_coef_s:
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)

    log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

    accs = []
    
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        dataset_acc = metrics.get('top1') * 100
        log.info(str(dataset) + ':' + str(dataset_acc) + '%')
        accs.append(dataset_acc)
    
    avg_acc = np.mean(accs)
    log.info(f'scaling_coef_ : {scaling_coef_}; Avg ACC: {avg_acc}%')


    # 保存结果
    results.append({
        'scaling_coef': scaling_coef_,
        'avg_acc': avg_acc,
        'dataset_accs': accs
    })
    
    times_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print("************************************结束时间:*************************************", times_end)

# # 找到最佳结果
# if results:
#     # 方法1：按平均准确率排序，取最高
#     best_result = max(results, key=lambda x: x['avg_acc'])
    
#     # 打印所有结果
#     log.info("\n" + "="*60)
#     log.info("所有 scaling_coef 的结果汇总:")
#     log.info("="*60)
    
#     for i, result in enumerate(results):
#         log.info(f"scaling_coef={result['scaling_coef']:.3f}: 平均准确率 = {result['avg_acc']:.4f}%")
    
#     log.info("\n" + "="*60)
#     log.info("最佳结果:")
#     log.info("="*60)
#     log.info(f"最佳 scaling_coef: {best_result['scaling_coef']}")
#     log.info(f"对应的平均准确率: {best_result['avg_acc']:.4f}%")

