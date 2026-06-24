
from typing import Dict, NamedTuple, Tuple
import torch

from collections import defaultdict
from typing import NamedTuple, Dict, Tuple, Optional, Union, List
from jax import random as jax_random

import jax.numpy as jnp
import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

from torch import optim
import copy
import torch.nn.functional as F
import warnings
import scipy


def solve_permutation(W_prot, W_base, layer_name="unknown"):
    """
    通用求解器：利用匈牙利算法找到最佳行置换
    输入: [N, D] 的矩阵 (会对 N 进行匹配)
    返回: indices, 使得 W_prot[indices] ≈ W_base
    """
    # 转换为 Numpy 并处理数据
    wp = W_prot.detach().cpu().numpy()
    wb = W_base.detach().cpu().numpy()
    
    # 确保维度一致
    if wp.shape != wb.shape:
        print(f" [Error] Shape mismatch in {layer_name}: {wp.shape} vs {wb.shape}")
        return None

    # 归一化 (Cosine Similarity)
    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    
    # 计算 Cost Matrix (Negative Correlation)
    # 我们希望相关性最大，即 Cost 最小
    Cost = - np.matmul(norm(wp), norm(wb).T)
    
    # 求解
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(Cost)
    
    # 构建恢复索引: restore_indices[col] = row
    dim = wp.shape[0]
    restore_indices = np.zeros(dim, dtype=int)
    restore_indices[col_ind] = row_ind
    
    # 计算匹配误差用于监控
    diff = np.mean(np.abs(wp[restore_indices] - wb))
    # print(f"   > Solved {layer_name}: Dim={dim}, Diff={diff:.4f}")
    
    return torch.from_numpy(restore_indices).long().to(W_prot.device)

def attack_vit_mlp_only(protected_model, anchor_model):
    """
    【仅 MLP 攻击版本】
    逻辑：
    1. 忽略 Attention 层的旋转和置换保护。
    2. 针对每个 Transformer Block，仅对 MLP 的中间隐藏层进行重排恢复。

    anchor_model: Pretrained model used as the alignment reference.
    """
    print(">>> Starting MLP-only attack (pretrained anchor)...")

    # 获取 visual 模块
    resblocks = protected_model.model.visual.transformer.resblocks
    anchor_resblocks = anchor_model.model.visual.transformer.resblocks

    with torch.no_grad():
        for i, (blk_p, blk_anchor) in enumerate(zip(resblocks, anchor_resblocks)):
            
            # --- 核心：MLP 内部置换攻击 ---
            # Align protected MLP hidden units to the pretrained model ordering.
            # W_fc 形状: [hidden_dim, embed_dim] -> 针对行(dim 0)进行置换
            w_fc_p = blk_p.mlp.c_fc.weight
            w_fc_anchor = blk_anchor.mlp.c_fc.weight
            
            # 计算置换索引
            mlp_perm = solve_permutation(w_fc_p, w_fc_anchor, f"Block_{i}_MLP")
            
            if mlp_perm is not None:
                # 1. 修正第一层 MLP (c_fc) 的输出（行置换 + Bias）
                blk_p.mlp.c_fc.weight.data = blk_p.mlp.c_fc.weight.data[mlp_perm]
                if blk_p.mlp.c_fc.bias is not None:
                    blk_p.mlp.c_fc.bias.data = blk_p.mlp.c_fc.bias.data[mlp_perm]
                
                # 2. 修正第二层 MLP (c_proj) 的输入（列置换）
                # W_proj 形状: [embed_dim, hidden_dim] -> 针对列(dim 1)进行置换
                blk_p.mlp.c_proj.weight.data = blk_p.mlp.c_proj.weight.data[:, mlp_perm]

            if i % 4 == 0:
                print(f"   - Block {i}: MLP alignment applied.")

    print(">>> MLP-only attack completed.")
    return protected_model


#下面是对attention进行攻击

def solve_linear_transformation(X_src, X_tgt, lambda_reg=1e-8):
    """
    求解矩阵 T，使得 X_src @ T ≈ X_tgt。
    使用双精度 (float64) 以保证数值稳定性。
    """
    X_src = X_src.double()
    X_tgt = X_tgt.double()
    
    d_in = X_src.shape[1]
    # 岭回归: (X^T X + lambda I)^-1 X^T Y
    XTX = X_src.T @ X_src
    reg = lambda_reg * torch.eye(d_in, device=X_src.device, dtype=torch.float64)
    
    # 求解 T
    T = torch.linalg.lstsq(XTX + reg, X_src.T @ X_tgt).solution
    return T.float()

def attack_attention_headwise_precise(
    params: Dict[str, torch.Tensor],
    anchor_params: Dict[str, torch.Tensor],
    layer_idx: int,
    num_heads: int = 12,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    【精确版】逐头 Attention 攻击，精确处理 Weight 和 Bias 的逆变换。
    完全适配 protect_attention_with_general_invertible 的保护逻辑。

    Use the pretrained model as anchor and recover protected attention weights
    by solving the linear relationship between protected and pretrained parameters.
    """
    print(f"Executing Precise Head-wise Attack on Layer {layer_idx}...")
    
    # Key 定义
    prefix = f"model.visual.transformer.resblocks.{layer_idx}.attn"
    key_in_w = f"{prefix}.in_proj_weight"
    key_in_b = f"{prefix}.in_proj_bias"
    key_out_w = f"{prefix}.out_proj.weight"
    key_out_b = f"{prefix}.out_proj.bias"

    if key_in_w not in params:
        print(f"Warning: Layer {layer_idx} keys not found.")
        return {}

    # 1. 加载参数 (移至 GPU 计算更快)
    W_in_prot = params[key_in_w].to(device)
    W_in_anchor = anchor_params[key_in_w].to(device)
    W_out_prot = params[key_out_w].to(device)
    W_out_anchor = anchor_params[key_out_w].to(device)
    
    b_in_prot = params.get(key_in_b)
    if b_in_prot is not None:
        b_in_prot = b_in_prot.to(device)
        
    # 2. 准备容器
    total_dim = W_in_prot.shape[1]
    head_dim = total_dim // num_heads
    
    # 切分 Q, K, V (In Proj)
    # PyTorch存储格式: [Q, K, V] 堆叠在 dim 0
    def get_qkv(w):
        return w.split(total_dim, dim=0)
    
    Q_prot, K_prot, V_prot = get_qkv(W_in_prot)
    Q_anchor, K_anchor, V_anchor = get_qkv(W_in_anchor)
    
    if b_in_prot is not None:
        bQ_prot, bK_prot, bV_prot = get_qkv(b_in_prot)
    else:
        bQ_prot = bK_prot = bV_prot = None

    # 结果列表
    rec_Q, rec_K, rec_V = [], [], []
    rec_bQ, rec_bK, rec_bV = [], [], []
    rec_W_out_cols = [] # W_out 按列恢复

    # 3. 逐头独立攻击
    for h in range(num_heads):
        start = h * head_dim
        end   = (h + 1) * head_dim
        
        # --- (A) Q, K, V 处理 (左乘变换) ---
        # 保护逻辑: W_new = P^T @ W_old
        # 攻击目标: 用 anchor 微调模型求解 T，使得 W_prot.T @ T ≈ W_anchor.T
        # 还原逻辑: W_rec = T^T @ W_prot
        # Bias还原: b_rec = b_new @ T
        
        # Helper: 处理单个投影矩阵
        def recover_proj_weights(w_p, w_ref, b_p):
            # w_p: (head_dim, embed_dim) -> 转置为 (embed_dim, head_dim) 作为样本
            # 求解 T: (head_dim, head_dim)
            T = solve_linear_transformation(w_p.T, w_ref.T)
            
            # 还原 Weight: T.T @ w_p
            w_rec = torch.matmul(T.T, w_p)
            
            # 还原 Bias: b_p @ T (注意 b_p 是行向量)
            b_rec = None
            if b_p is not None:
                # b_p shape: (head_dim,)
                b_rec = torch.matmul(b_p, T)
            return w_rec, b_rec

        # Q 恢复
        q_rec, bq_rec = recover_proj_weights(
            Q_prot[start:end], Q_anchor[start:end],
            bQ_prot[start:end] if bQ_prot is not None else None
        )
        rec_Q.append(q_rec)
        rec_bQ.append(bq_rec)
        
        # K 恢复
        k_rec, bk_rec = recover_proj_weights(
            K_prot[start:end], K_anchor[start:end],
            bK_prot[start:end] if bK_prot is not None else None
        )
        rec_K.append(k_rec)
        rec_bK.append(bk_rec)
        
        # V 恢复
        v_rec, bv_rec = recover_proj_weights(
            V_prot[start:end], V_anchor[start:end],
            bV_prot[start:end] if bV_prot is not None else None
        )
        rec_V.append(v_rec)
        rec_bV.append(bv_rec)
        
        # --- (B) Output Proj 处理 (右乘变换) ---
        # 保护逻辑: W_out_new = W_out_old @ (M^-1)^T
        # 攻击目标: 找到 T，使得 W_out_new @ T ≈ W_out_anchor
        # 还原逻辑: W_out_rec = W_out_new @ T
        # 注意: W_out 是 (embed_dim, head_dim) 的切片
        
        o_p = W_out_prot[:, start:end]
        o_ref = W_out_anchor[:, start:end]
        
        # 直接求解: o_p @ T ≈ o_ref
        T_out = solve_linear_transformation(o_p, o_ref)
        
        o_rec = torch.matmul(o_p, T_out)
        rec_W_out_cols.append(o_rec)

    # 4. 拼装结果
    # Weight
    W_Q_rec = torch.cat(rec_Q, dim=0)
    W_K_rec = torch.cat(rec_K, dim=0)
    W_V_rec = torch.cat(rec_V, dim=0)
    W_in_rec = torch.cat([W_Q_rec, W_K_rec, W_V_rec], dim=0)
    
    W_out_rec = torch.cat(rec_W_out_cols, dim=1) # 列拼接
    
    # Bias
    if bQ_prot is not None:
        b_Q_rec = torch.cat(rec_bQ, dim=0)
        b_K_rec = torch.cat(rec_bK, dim=0)
        b_V_rec = torch.cat(rec_bV, dim=0)
        b_in_rec = torch.cat([b_Q_rec, b_K_rec, b_V_rec], dim=0)
    else:
        b_in_rec = None

    # 5. 返回字典 (保持在 GPU 或转回 CPU 由外部决定)
    res = {
        key_in_w: W_in_rec,
        key_out_w: W_out_rec
    }
    if b_in_rec is not None:
        res[key_in_b] = b_in_rec
        
    # out_proj.bias 不需要处理，直接复制原 protected 的即可
    if key_out_b in params:
        res[key_out_b] = params[key_out_b].clone()
        
    return res

def universal_attention_attack_params(
    defender_params: Dict[str, torch.Tensor],
    anchor_params: Dict[str, torch.Tensor],
    layer_indices: Optional[List[int]] = None,
    device='cuda'
) -> Dict[str, torch.Tensor]:
    """
    通用攻击入口函数：恢复所有层的 Attention 参数。

    anchor_params: Pretrained model state_dict used as the alignment reference.
    """
    if layer_indices is None:
        # 自动检测层数，这里假设是 12
        layer_indices = list(range(12))
        
    recovered_model = defender_params.copy()
    
    print("Starting Universal Linear Transformation Attack (pretrained anchor)...")
    
    for idx in layer_indices:
        print(f"Attacking Layer {idx}...")
        try:
            
            layer_params = attack_attention_headwise_precise(
                defender_params,
                anchor_params,
                idx,
                device=device,
            )            
                        
           # 更新参数
            if layer_params:
                recovered_model.update(layer_params)
            else:
                print(f"Skipping Layer {idx}: Keys not found or format mismatch.")
                
        except Exception as e:
            print(f"Error attacking Layer {idx}: {e}")
            # 即使出错也继续下一层
            continue
            
    print("Universal Attack Completed.")
    return recovered_model
