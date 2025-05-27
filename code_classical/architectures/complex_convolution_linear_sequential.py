import torch
import torch.nn as nn
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexMaxPool2d, ComplexReLU


class ComplexFlatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


def complex_convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim=3, k_lipschitz=None):
    """
    创建一个复数卷积+线性序列网络
    
    参数:
    - input_dims: 输入维度[channels, height, width]
    - linear_hidden_dims: 线性层的隐藏维度列表
    - conv_hidden_dims: 卷积层的隐藏维度列表
    - output_dim: 输出维度
    - kernel_dim: 卷积核大小
    - k_lipschitz: Lipschitz常数（如果使用）
    
    返回:
    - 复数卷积+线性序列网络模型
    """
    # 确认输入维度格式正确
    assert len(input_dims) == 3, "输入维度必须是[channels, height, width]格式"
    in_channels, height, width = input_dims
    
    # 创建网络层列表
    layers = []
    
    # 卷积层部分
    current_channels = in_channels
    for hidden_dim in conv_hidden_dims:
        layers.append(ComplexConv2d(current_channels, hidden_dim, kernel_size=kernel_dim, padding=kernel_dim//2))
        layers.append(ComplexBatchNorm2d(hidden_dim))
        layers.append(ComplexReLU())
        # 添加最大池化层，减小特征图尺寸
        layers.append(ComplexMaxPool2d(kernel_size=2, stride=2))
        current_channels = hidden_dim
    
    # 计算卷积层输出后的特征图尺寸
    # 每次池化后尺寸减半
    final_height = height // (2 ** len(conv_hidden_dims))
    final_width = width // (2 ** len(conv_hidden_dims))
    
    # 展平层
    layers.append(ComplexFlatten())
    
    # 卷积层输出后的特征总数
    conv_output_size = current_channels * final_height * final_width
    
    # 线性层部分
    current_dim = conv_output_size
    for hidden_dim in linear_hidden_dims:
        layers.append(ComplexLinear(current_dim, hidden_dim))
        layers.append(ComplexBatchNorm1d(hidden_dim))
        layers.append(ComplexReLU())
        current_dim = hidden_dim
    
    # 输出层
    layers.append(ComplexLinear(current_dim, output_dim))
    
    # 创建并返回序列模型
    return nn.Sequential(*layers)