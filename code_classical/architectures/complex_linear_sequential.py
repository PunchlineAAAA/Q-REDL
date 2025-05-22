import torch
import torch.nn as nn
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear, ComplexBatchNorm1d, ComplexReLU


def complex_linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    """
    创建一个复数线性序列网络
    
    参数:
    - input_dims: 输入维度
    - hidden_dims: 隐藏层维度列表
    - output_dim: 输出维度
    - k_lipschitz: Lipschitz常数（如果使用）
    
    返回:
    - 复数线性序列网络模型
    """
    # 检查输入是数组还是单个整数
    if isinstance(input_dims, list) or isinstance(input_dims, tuple):
        input_dim = int(np.prod(input_dims))
    else:
        input_dim = input_dims
    
    # 创建网络层列表
    layers = []
    
    # 第一层：输入层到第一个隐藏层
    layers.append(ComplexLinear(input_dim, hidden_dims[0]))
    layers.append(ComplexBatchNorm1d(hidden_dims[0]))
    layers.append(ComplexReLU())
    
    # 中间隐藏层
    for i in range(len(hidden_dims)-1):
        layers.append(ComplexLinear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(ComplexBatchNorm1d(hidden_dims[i+1]))
        layers.append(ComplexReLU())
    
    # 输出层
    layers.append(ComplexLinear(hidden_dims[-1], output_dim))
    
    # 创建并返回序列模型
    return nn.Sequential(*layers)