import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# 需要重新定义的架构导入
# 注意：这些架构需要重新实现为复数版本
from architectures.complex_linear_sequential import complex_linear_sequential
from architectures.complex_convolution_linear_sequential import complex_convolution_linear_sequential
from architectures.complex_vgg_sequential import complex_vgg16_bn


class ModifiedEvidentialNet(nn.Module):
    """
    复数版证据神经网络（Complex Modified Evidential Neural Network）
    这是一种基于证据理论的深度学习模型，使用复数张量计算，可以估计预测的不确定性
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

        # 根据指定的架构类型构建复数网络
        if architecture == 'linear':
            # 线性复数网络架构
            self.sequential = complex_linear_sequential(input_dims=self.input_dims,
                                                hidden_dims=self.hidden_dims,
                                                output_dim=self.output_dim,
                                                k_lipschitz=self.k_lipschitz)
        elif architecture == 'conv':
            # 卷积复数网络架构
            assert len(input_dims) == 3  # 确保输入维度适合卷积网络（通道数，高度，宽度）
            self.sequential = complex_convolution_linear_sequential(input_dims=self.input_dims,
                                                            linear_hidden_dims=self.hidden_dims,
                                                            conv_hidden_dims=[64, 64, 64],  # 卷积层隐藏维度
                                                            output_dim=self.output_dim,
                                                            kernel_dim=self.kernel_dim,
                                                            k_lipschitz=self.k_lipschitz)
        elif architecture == 'vgg':
            # VGG复数网络架构
            assert len(input_dims) == 3  # 确保输入维度适合VGG网络
            self.sequential = complex_vgg16_bn(output_dim=self.output_dim, k_lipschitz=self.k_lipschitz)
        else:
            raise NotImplementedError

        # 需要实现一个复数版本的Softmax
        # 由于标准softmax不适用于复数，这里我们将使用模值的方式处理
        self.clf_type = clf_type

        # 优化器设置
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 如果是卷积架构，添加学习率调度器
        if architecture == 'conv':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)

    def complex_softmax(self, x):
        """
        复数版本的softmax函数
        计算复数张量的模值，然后应用标准softmax
        
        参数:
        - x: 复数张量
        
        返回:
        - 应用于复数张量模值的softmax概率分布
        """
        # 计算复数张量的模值（绝对值）
        x_abs = torch.abs(x)
        # 应用标准softmax到模值上
        return F.softmax(x_abs, dim=-1)

    def complex_softplus(self, x):
        """
        复数版本的softplus函数
        分别应用于实部和虚部
        
        参数:
        - x: 复数张量
        
        返回:
        - 处理后的复数张量
        """
        # 分离实部和虚部
        real = x.real
        imag = x.imag
        
        # 分别应用softplus
        real_out = F.softplus(real)
        imag_out = F.softplus(imag)
        
        # 重新组合为复数
        return torch.complex(real_out, imag_out)

    def forward(self, input, labels_=None, return_output='alpha', compute_loss=False, epoch=10.):
        """
        前向传播
        
        参数:
        - input: 输入数据（复数张量）
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
        
        # 对于复数输出，我们需要一个特殊的softplus处理
        evidence = self.complex_softplus(logits)  # 应用复数版softplus激活函数获取证据
        
        # 计算Dirichlet分布的α参数
        # 对于复数值，我们使用模值加上lamb2作为alpha
        alpha_real = evidence.real + self.lamb2
        alpha_imag = evidence.imag
        alpha = torch.complex(alpha_real, alpha_imag)

        # 如果需要计算损失
        if compute_loss:
            # 将标签转换为one-hot编码（这里需要处理复数情况）
            labels_1hot = torch.zeros_like(logits.real).scatter_(-1, labels_.unsqueeze(-1), 1)
            labels_1hot = torch.complex(labels_1hot, torch.zeros_like(labels_1hot))
            
            # 计算均方误差损失（使用复数版本）
            self.loss_mse = self.compute_complex_mse(labels_1hot, evidence)
            
            # 计算KL散度的alpha参数
            # 由于KL散度定义在实数域上，我们只使用实部进行计算
            kl_alpha_real = evidence.real * (1 - labels_1hot.real) + self.lamb2
            
            # 计算KL散度损失（只使用实部）
            self.loss_kl = self.compute_kl_loss(kl_alpha_real, self.lamb2)

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
            return self.complex_softmax(logits)
        elif return_output == 'alpha':
            # 返回Dirichlet分布参数alpha
            return alpha
        else:
            raise AssertionError

    def compute_complex_mse(self, labels_1hot, evidence):
        """
        计算复数张量的均方误差损失
        
        参数:
        - labels_1hot: One-hot编码的标签（复数张量）
        - evidence: 模型产生的证据（复数张量）
        
        返回:
        - loss_mse: 均方误差损失
        """
        num_classes = evidence.shape[-1]  # 类别数量
        
        # 计算总证据（使用模值的平方和）
        evidence_sum = torch.sum(torch.abs(evidence)**2, dim=-1, keepdim=True)
        
        # 计算预测与真实标签之间的差距
        # 公式需要调整以适应复数情况
        # 我们使用复数除法，然后计算实部和虚部的均方差
        denominator = (torch.abs(evidence) + self.lamb1 * (evidence_sum - torch.abs(evidence)**2) + 
                       self.lamb2 * num_classes)
        
        # 为了避免除零错误，添加一个小的epsilon
        epsilon = 1e-8
        normalized_evidence = (torch.abs(evidence) + self.lamb2) / (denominator + epsilon)
        
        # 计算与标签的差距
        gap = labels_1hot.real - normalized_evidence
        
        # 计算平方误差
        loss_mse = gap.pow(2).sum(-1)
        
        # 返回平均损失
        return loss_mse.mean()

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        """
        计算KL散度损失（保持不变，因为KL散度在实数域上定义）
        
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
        根据复数logits预测类别
        
        参数:
        - p: 复数logits输出
        
        返回:
        - output_pred: 预测的类别索引
        """
        # 使用模值来确定最大值
        p_abs = torch.abs(p)
        output_pred = torch.max(p_abs, dim=-1)[1]
        return output_pred