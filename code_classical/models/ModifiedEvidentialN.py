import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from architectures.linear_sequential import linear_sequential
from architectures.convolution_linear_sequential import convolution_linear_sequential
from architectures.vgg_sequential import vgg16_bn


class ModifiedEvidentialNet(nn.Module):
    """
    修改版的证据神经网络（Modified Evidential Neural Network）
    这是一种基于证据理论的深度学习模型，可以估计预测的不确定性
    """
    def __init__(self,
                 input_dims,  # 输入维度，整数列表
                 output_dim,  # 输出维度，整数
                 hidden_dims=[64, 64, 64],  # 隐藏层维度，整数列表
                 kernel_dim=None,  # 卷积核维度（如果使用卷积架构），整数
                 architecture='linear',  # 编码器架构名称，字符串
                 k_lipschitz=None,  # Lipschitz常数，浮点数或None（如果不使用Lipschitz约束）
                 batch_size=64,  # 批量大小，整数
                 lr=1e-3,  # 学习率，浮点数
                 loss='IEDL',  # 损失函数名称，字符串
                 clf_type='softplus',  # 分类器类型，字符串
                 fisher_c=1.0,  # Fisher信息系数，浮点数
                 lamb1=1.0,  # lambda1参数，用于损失计算，浮点数
                 lamb2=1.0,  # lambda2参数，用于损失计算，浮点数
                 seed=123):  # 随机种子，整数
        super().__init__()

        # 设置随机种子以保证结果可复现
        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # 架构参数
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = input_dims, output_dim, hidden_dims, kernel_dim
        self.k_lipschitz = k_lipschitz
        # 训练参数
        self.batch_size, self.lr = batch_size, lr
        self.loss = loss

        # 目标浓度和KL散度系数
        self.target_con = 1.0  # 目标浓度参数，用于KL散度计算
        self.kl_c = -1.0  # KL散度系数，如果为-1则使用动态权重
        self.fisher_c = fisher_c  # Fisher信息系数
        self.lamb1 = lamb1  # lambda1参数，用于损失计算
        self.lamb2 = lamb2  # lambda2参数，用于损失计算

        # 初始化各损失组件
        self.loss_mse = torch.tensor(0.0)  # 均方误差损失
        self.loss_var = torch.tensor(0.0)  # 方差损失
        self.loss_kl = torch.tensor(0.0)  # KL散度损失
        self.loss_fisher = torch.tensor(0.0)  # Fisher信息损失

        # 根据指定的架构类型构建网络
        if architecture == 'linear':
            # 线性网络架构
            self.sequential = linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.output_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            # 卷积网络架构
            assert len(input_dims) == 3  # 确保输入维度适合卷积网络（通道数，高度，宽度）
            self.sequential = convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],  # 卷积层隐藏维度
                                                            output_dim=self.output_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
        elif architecture == 'vgg':
            # VGG网络架构
            assert len(input_dims) == 3  # 确保输入维度适合VGG网络
            self.sequential = vgg16_bn(output_dim=self.output_dim, k_lipschitz=self.k_lipschitz)
        else:
            raise NotImplementedError

        # 定义Softmax层用于输出概率分布
        self.softmax = nn.Softmax(dim=-1)
        self.clf_type = clf_type

        # 优化器设置
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 如果是卷积架构，添加学习率调度器
        if architecture == 'conv':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def forward(self, input, labels_=None, return_output='alpha', compute_loss=False, epoch=10.):
        """
        前向传播
        
        参数:
        - input: 输入数据
        - labels_: 标签数据（如果需要计算损失）
        - return_output: 返回输出类型，可选 'hard'（硬标签）, 'soft'（软标签）, 'alpha'（Dirichlet分布参数）
        - compute_loss: 是否计算损失
        - epoch: 当前训练轮次，用于动态权重计算
        
        返回:
            根据return_output参数返回不同类型的输出
        """
        # 确保在计算损失时提供了标签
        assert not (labels_ is None and compute_loss)

        # 前向传播过程
        logits = self.sequential(input)  # 获取网络的原始输出
        evidence = F.softplus(logits)  # 应用softplus激活函数获取证据
        alpha = evidence + self.lamb2  # 计算Dirichlet分布的α参数

        # 如果需要计算损失
        if compute_loss:
            # 将标签转换为one-hot编码
            labels_1hot = torch.zeros_like(logits).scatter_(-1, labels_.unsqueeze(-1), 1)
            # 计算均方误差损失
            self.loss_mse = self.compute_mse(labels_1hot, evidence)
            # 计算KL散度的alpha参数
            kl_alpha = evidence * (1 - labels_1hot) + self.lamb2
            # 计算KL散度损失
            self.loss_kl = self.compute_kl_loss(kl_alpha, self.lamb2)

            # 根据kl_c的值决定如何组合损失
            if self.kl_c == -1:
                # 使用动态权重调整KL散度的影响，随着训练进行逐渐增加
                regr = np.minimum(1.0, epoch / 10.)
                self.grad_loss = self.loss_mse + regr * self.loss_kl
            else:
                # 使用固定系数
                self.grad_loss = self.loss_mse + self.kl_c * self.loss_kl

        # 根据return_output参数返回不同类型的输出
        if return_output == 'hard':
            # 返回硬标签（类别预测）
            return self.predict(logits)
        elif return_output == 'soft':
            # 返回软标签（概率分布）
            return self.softmax(logits)
        elif return_output == 'alpha':
            # 返回Dirichlet分布参数alpha
            return alpha
        else:
            raise AssertionError

    def compute_mse(self, labels_1hot, evidence):
        """
        计算均方误差损失
        
        参数:
        - labels_1hot: One-hot编码的标签
        - evidence: 模型产生的证据
        
        返回:
        - loss_mse: 均方误差损失
        """
        num_classes = evidence.shape[-1]  # 类别数量

        # 计算预测与真实标签之间的差距
        # 公式基于Dirichlet分布的期望与标签的差异
        gap = labels_1hot - (evidence + self.lamb2) / \
              (evidence + self.lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + self.lamb2 * num_classes)

        # 计算平方误差
        loss_mse = gap.pow(2).sum(-1)

        # 返回平均损失
        return loss_mse.mean()

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        """
        计算KL散度损失
        
        参数:
        - alphas: Dirichlet分布的α参数
        - target_concentration: 目标浓度参数
        - epsilon: 小常数，防止数值不稳定
        
        返回:
        - loss: KL散度损失
        """
        # 创建目标alpha值，所有维度都是target_concentration
        target_alphas = torch.ones_like(alphas) * target_concentration

        # 计算alpha和目标alpha的总和
        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        # 计算KL散度的第一部分 (涉及gamma函数)
        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        # 处理可能的非有限值
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        # 确保所有值都是有限的
        assert torch.all(torch.isfinite(alp0_term)).item()

        # 计算KL散度的第二部分 (涉及gamma函数和digamma函数)
        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        # 处理可能的非有限值
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        # 确保所有值都是有限的
        assert torch.all(torch.isfinite(alphas_term)).item()

        # 合并两部分并计算平均损失
        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss

    def step(self):
        """
        执行一步优化
        """
        self.optimizer.zero_grad()  # 清除梯度
        self.grad_loss.backward()   # 反向传播计算梯度
        self.optimizer.step()       # 更新参数

    def predict(self, p):
        """
        根据logits预测类别
        
        参数:
        - p: logits输出
        
        返回:
        - output_pred: 预测的类别索引
        """
        # 返回概率最高的类别索引
        output_pred = torch.max(p, dim=-1)[1]
        return output_pred