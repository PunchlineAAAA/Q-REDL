import torch
import torch.nn as nn
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear, ComplexConv2d, ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexMaxPool2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


class ComplexFlatten(nn.Module):
    """
    将多维复数张量展平为一维复数张量
    """
    def forward(self, x):
        # 复数张量展平，保持批次维度不变
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class ComplexVGG16BN(nn.Module):
    """
    复数版本的VGG16网络结构，带批归一化
    """
    def __init__(self, output_dim, k_lipschitz=None):
        super(ComplexVGG16BN, self).__init__()
        
        # 配置VGG16网络结构
        self.features = self._make_features()
        
        # 分类器部分
        self.avgpool = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.flatten = ComplexFlatten()
        
        # 假设输入是224x224，根据VGG16结构，特征提取后尺寸为7x7
        self.classifier = nn.Sequential(
            ComplexLinear(512 * 7 * 7, 4096),
            ComplexBatchNorm1d(4096),
            complex_relu,
            ComplexLinear(4096, 4096),
            ComplexBatchNorm1d(4096),
            complex_relu,
            ComplexLinear(4096, output_dim)
        )
    
    def _make_features(self):
        """
        创建VGG16的特征提取部分
        """
        # VGG16配置：数字表示输出通道数，'M'表示最大池化层
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        layers = []
        in_channels = 3  # 假设输入是3通道RGB图像
        
        for v in cfg:
            if v == 'M':
                layers.append(ComplexMaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = ComplexConv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv2d, ComplexBatchNorm2d(v), complex_relu])
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入复数张量
        
        返回:
        - 模型输出
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def complex_vgg16_bn(output_dim, k_lipschitz=None):
    """
    创建复数版的VGG16网络，带批归一化
    
    参数:
    - output_dim: 输出维度
    - k_lipschitz: Lipschitz常数（如果使用）
    
    返回:
    - 复数VGG16网络模型
    """
    return ComplexVGG16BN(output_dim, k_lipschitz)