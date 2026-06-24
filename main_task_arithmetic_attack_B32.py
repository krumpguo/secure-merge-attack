import copy
import os

import numpy as np

import torch.nn as nn
import torch

import time
import sys
sys.path.insert(0, '/data/kuangpu.guo/Secure_merge/attack/src_new_attack')
sys.path.append('/data/kuangpu.guo/Secure_merge/attack/')

from utils import load_model_checkpoint
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
from utils_attack_B32 import universal_attention_attack_params, attack_vit_mlp_only

def apply_vector(vector, pretrained_checkpoint):
    with torch.no_grad():
        pretrained_model = load_model_checkpoint(pretrained_checkpoint)
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


args = parse_arguments()

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
attack_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']


args.model = 'ViT-B-32'
args.device_num = 0
scaling_coef_ = 0.2
model = args.model
attack_protectd_method = 'params_mlp_and_invertible'
model_layer = 12

args.data_location = '/data/kuangpu.guo/Secure_merge/attack/data'
args.save = '/data/kuangpu.guo/Secure_merge/attack/checkpoints/' + model
args.logs_path = '/data/kuangpu.guo/Secure_merge/attack/logs/' + model

pretrained_checkpoint = (
    '/data/kuangpu.guo/Secure_merge/attack/checkpoints/'
    + model + '/zeroshot.pt'
)
ptm_model = load_model_checkpoint(pretrained_checkpoint, map_location='cpu')
ptm_params = ptm_model.state_dict()

args.device = f'cuda:{args.device_num}'
args.batch_size = 128

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, f'log_{str_time_}_single_task_protect.txt')

# Outer loop: attack one dataset at a time
for attack_dataset in attack_datasets:

    log.info('=' * 80)
    log.info(f'Attack dataset: {attack_dataset}')
    log.info('=' * 80)

    task_vectors = []

    for dataset_name in exam_datasets:
        finetuned_ckpt = (
            f'/data/kuangpu.guo/Secure_merge/attack/checkpoints/'
            f'{model}/{dataset_name}/finetuned.pt'
        )
        if dataset_name == attack_dataset:
            log.info(f'Attack protected {attack_protectd_method} to {dataset_name}')
            log.info('Using pretrained model as attack anchor')

            print('No cached attacked model found; starting attack...')

            protectd_ckpt = finetuned_ckpt.replace('finetuned.pt', f'finetuned_{attack_protectd_method}_protection.pt')
            protectd_model = load_model_checkpoint(protectd_ckpt, map_location='cpu')

            attacked_model = copy.deepcopy(protectd_model)

            attacked_model_state = universal_attention_attack_params(
                defender_params=attacked_model.state_dict(),
                anchor_params=ptm_params,
                layer_indices=list(range(12)),
                device=args.device,
            )

            attacked_model.load_state_dict(attacked_model_state, strict=False)

            attacked_model = attack_vit_mlp_only(attacked_model, ptm_model)

            log.info(f'Evaluating attacked model on {attack_dataset}...')
            eval_single_dataset(attacked_model, attack_dataset, args)

            attacked_model = attacked_model.to('cpu')

            task_vector = TaskVector(
                pretrained_checkpoint,
                attacked_model.state_dict()
            )

        else:
            # Use vanilla fine-tuned checkpoints for non-attack tasks
            task_vector = TaskVector(
                pretrained_checkpoint,
                finetuned_ckpt
            )

        task_vectors.append(task_vector)

    # Task Arithmetic merge
    task_vector_sum = sum(task_vectors)
    image_encoder = task_vector_sum.apply_to(
        pretrained_checkpoint,
        scaling_coef=scaling_coef_
    )

    # Evaluate merged model on all datasets
    accs = []
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        dataset_acc = metrics.get('top1') * 100
        log.info(str(dataset) + ':' + str(dataset_acc) + '%')
        accs.append(dataset_acc)

    formatted_accs = ", ".join([f"{x:.2f}" for x in accs])
    log.info(f"[Eval ALL Attack ] {attack_dataset}: {formatted_accs}")

    avg_acc = np.mean(accs)
    log.info(f'scaling_coef_ : {scaling_coef_}; Avg ACC: {avg_acc}%')

torch.cuda.empty_cache()
sys.exit(0)
