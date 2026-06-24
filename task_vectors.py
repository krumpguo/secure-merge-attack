
import sys
from tabnanny import verbose
sys.path.insert(0, "/data/kuangpu.guo/Secure_merge/attack")  # 这里放“包含src目录的根目录”
sys.path.insert(0, "/data/kuangpu.guo/Secure_merge/attack/src_new_attack")
import torch
import copy

from utils import load_model_checkpoint

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                if isinstance(pretrained_checkpoint, str): 
                    pretrained_state_dict = load_model_checkpoint(pretrained_checkpoint, map_location='cpu').state_dict()
                else:
                    pretrained_state_dict = copy.deepcopy(pretrained_checkpoint)

                if isinstance(finetuned_checkpoint, str): 
                    finetuned_state_dict = load_model_checkpoint(finetuned_checkpoint, map_location='cpu').state_dict()
                    print('TaskVector:' + finetuned_checkpoint)
                else:
                    finetuned_state_dict = copy.deepcopy(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = load_model_checkpoint(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    
    def apply_to_without_mlp(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model.
        
        Args:
            pretrained_checkpoint: 预训练模型路径
            scaling_coef: 非MLP参数的缩放系数
            scaling_coef_mlp: MLP参数的缩放系数，如果为None则使用scaling_coef
        """
        scaling_coef_mlp = scaling_coef / 7 * 8
        with torch.no_grad():
            pretrained_model = load_model_checkpoint(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            
            # 如果未指定MLP缩放系数，则使用统一的缩放系数
            if scaling_coef_mlp is None:
                scaling_coef_mlp = scaling_coef
            
            # 统计信息
            total_params = 0
            mlp_params = 0
            other_params = 0
            
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                
                total_params += 1
                
                # 判断是否是MLP参数
                is_mlp_param = any(pattern in key for pattern in [
                    '.mlp.c_fc.',  # 第一个线性层
                    '.mlp.c_proj.',  # 第二个线性层
                    '.mlp.0.',  # c_fc 作为第一个元素
                    '.mlp.2.',  # c_proj 作为第三个元素
                ])
                
                if is_mlp_param:
                    # 使用MLP缩放系数
                    new_state_dict[key] = pretrained_state_dict[key] + scaling_coef_mlp * self.vector[key]
                    mlp_params += 1
                else:
                    # 使用普通缩放系数
                    new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                    other_params += 1
            
            # 打印统计信息
            print(f"参数处理统计:")
            print(f"  总参数数: {total_params}")
            print(f"  MLP参数数: {mlp_params} (使用系数: {scaling_coef_mlp})")
            print(f"  其他参数数: {other_params} (使用系数: {scaling_coef})")
        
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def zero_mlp_layers(self):
        """将 mlp 相关参数的 task vector 值置为0"""
        with torch.no_grad():
            for key in list(self.vector.keys()):
                # 匹配 mlp.c_fc 和 mlp.c_proj 的参数
                if '.mlp.' in key or key.endswith('.mlp.weight') or key.endswith('.mlp.bias'):
                    self.vector[key] = torch.zeros_like(self.vector[key])
        
