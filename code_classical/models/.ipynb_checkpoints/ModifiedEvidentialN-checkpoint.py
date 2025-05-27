import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 需要重新定义的架构导入
# 注意：这些架构需要重新实现为复数版本
from architectures.complex_linear_sequential import complex_linear_sequential
from architectures.complex_convolution_linear_sequential import (
    complex_convolution_linear_sequential,
)
from architectures.complex_vgg_sequential import complex_vgg16_bn

from utils.metrics import quantum_x_entropy


class ModifiedEvidentialNet(nn.Module):
    """
    复数版证据神经网络（Complex Modified Evidential Neural Network）
    这是一种基于证据理论的深度学习模型，使用复数张量计算，可以估计预测的不确定性
    """

    def __init__(
        self,
        input_dims,  # 输入维度，整数列表
        output_dim,  # 输出维度，整数
        hidden_dims=[64, 64, 64],  # 隐藏层维度，整数列表
        kernel_dim=None,  # 卷积核维度（如果使用卷积架构），整数
        architecture="linear",  # 编码器架构名称，字符串
        k_lipschitz=None,  # Lipschitz常数，浮点数或None（如果不使用Lipschitz约束）
        batch_size=64,  # 批量大小，整数
        lr=1e-3,  # 学习率，浮点数
        loss="IEDL",  # 损失函数名称，字符串
        clf_type="softplus",  # 分类器类型，字符串
        fisher_c=1.0,  # Fisher信息系数，浮点数
        lamb1=1.0,  # lambda1参数，用于损失计算，浮点数
        lamb2=1.0,  # lambda2参数，用于损失计算，浮点数
        lambda_qx=0.01,  # lambda_qx，X-熵正则项的权重
        seed=123,
    ):  # 随机种子，整数
        super().__init__()

        # 设置随机种子以保证结果可复现
        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.FloatTensor)

        # 架构参数
        self.input_dims, self.output_dim, self.hidden_dims, self.kernel_dim = (
            input_dims,
            output_dim,
            hidden_dims,
            kernel_dim,
        )
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

        self.lambda_qx = lambda_qx  # lambda_qx参数，用于控制X-熵正则项的权重

        # 初始化各损失组件
        self.loss_mse = torch.tensor(0.0)  # 均方误差损失
        self.loss_var = torch.tensor(0.0)  # 方差损失
        self.loss_kl = torch.tensor(0.0)  # KL散度损失
        self.loss_fisher = torch.tensor(0.0)  # Fisher信息损失

        # 根据指定的架构类型构建复数网络
        if architecture == "linear":
            # 线性复数网络架构
            self.sequential = complex_linear_sequential(
                input_dims=self.input_dims,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                k_lipschitz=self.k_lipschitz,
            )
        elif architecture == "conv":
            # 卷积复数网络架构
            assert (
                len(input_dims) == 3
            )  # 确保输入维度适合卷积网络（通道数，高度，宽度）
            self.sequential = complex_convolution_linear_sequential(
                input_dims=self.input_dims,
                linear_hidden_dims=self.hidden_dims,
                conv_hidden_dims=[64, 64, 64],  # 卷积层隐藏维度
                output_dim=self.output_dim,
                kernel_dim=self.kernel_dim,
                k_lipschitz=self.k_lipschitz,
            )
        elif architecture == "vgg":
            # VGG复数网络架构
            assert len(input_dims) == 3  # 确保输入维度适合VGG网络
            self.sequential = complex_vgg16_bn(
                output_dim=self.output_dim, k_lipschitz=self.k_lipschitz
            )
        else:
            raise NotImplementedError

        # 需要实现一个复数版本的Softmax
        # 由于标准softmax不适用于复数，这里我们将使用模值的方式处理
        self.clf_type = clf_type

        # 优化器设置
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 如果是卷积架构，添加学习率调度器
        if architecture == "conv":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )

    def complex_softmax(self, x, use_squared=True):
        """
        复数softmax: 先对复数张量取模，再进行标准softmax（数值稳定）
        """
        x_work = torch.abs(x) ** 2 if use_squared else torch.abs(x)
        x_work = x_work - x_work.max(dim=-1, keepdim=True)[0]  # 数值稳定技巧
        return F.softmax(x_work, dim=-1)

    def complex_softplus(self, x):
        # magnitude = torch.abs(x)
        # # phase = torch.angle(x)    # polar
        # phase = x.imag

        # # 避免 phase NaN（如复数为 0）
        # phase = torch.where(magnitude < 1e-8, torch.zeros_like(phase), phase)
        # magnitude = torch.clamp(magnitude, max=20.0)  # 避免溢出

        # new_magnitude = F.softplus(magnitude)
        # return new_magnitude * torch.exp(1j * phase)
        # return torch.complex(F.softplus(x.real), F.softplus(x.imag))
        return torch.complex(F.softplus(torch.abs(x)), F.softplus(x.imag))

    def forward(
        self,
        input,
        labels_=None,
        return_output="alpha",
        compute_loss=False,
        epoch=10.0,
        batch_index=0,
    ):
        assert not (labels_ is None and compute_loss)

        if not torch.is_complex(input):
            raise ValueError(
                "Input must be complex-valued. Please encode using encode_complex()."
            )

        input = input.to(torch.complex64)
        input = input.to(next(self.parameters()).device)

        complex_logits = self.sequential(input)
        evidence = self.complex_softplus(complex_logits)
        norm = torch.abs(evidence).sum(-1, keepdim=True) + 1e-6  # 按类别维归一化
        evidence = evidence / norm  # 保持比例但控制总量
        # 获取模长和相位并做clamp
        magnitude = torch.abs(evidence)
        phase = torch.angle(evidence)
        clamped_magnitude = magnitude.clamp(max=30.0)
        evidence = clamped_magnitude * torch.exp(1j * phase)

        alpha = evidence + self.lamb2  # 保留复数结构，不丢弃虚部

        if compute_loss:
            labels_1hot = torch.zeros_like(alpha.real).scatter_(
                -1, labels_.unsqueeze(-1), 1
            )
            labels_1hot = torch.complex(labels_1hot, torch.zeros_like(labels_1hot))

            # projected probability 仅使用实部
            # alpha_real = alpha.real + 1e-6
            alpha_real = alpha.real.clamp(min=1e-3)  # 保底避免除以 0 和 log(0)
            alpha_sum = torch.sum(alpha_real, dim=-1, keepdim=True)
            projected_prob = alpha_real / alpha_sum
            self.loss_mse = F.mse_loss(projected_prob, labels_1hot.real)

            # KL 散度也只基于实部
            # kl_alpha = evidence.real * (1 - labels_1hot.real) + self.lamb2
            # self.loss_kl = self.compute_kl_loss(kl_alpha, self.lamb2)
            # REDL 推荐形式的 KL：真实 Dirichlet vs 均匀先验
            self.loss_kl = self.compute_kl_loss(
                alpha.real, target_concentration=self.lamb2
            )
            self.loss_kl = torch.clamp(self.loss_kl, min=0.0)

            # 计算复数量子X-entropy正则项（对复数alpha操作）
            qx_loss = quantum_x_entropy(alpha).mean()
            self.loss_qx = qx_loss

            if self.kl_c == -1:
                regr = np.minimum(1.0, epoch / 10.0)
                self.grad_loss = self.loss_mse + regr * self.loss_kl
            else:
                self.grad_loss = self.loss_mse + self.kl_c * self.loss_kl

            self.grad_loss += self.lambda_qx * qx_loss

            # ✅ 只在第一个 batch 打印一次
            # if batch_index % 300 == 0:
            #     print(
            #         f"[Debug] epoch={epoch} | loss_mse={self.loss_mse.item():.6f} | "
            #         f"loss_kl={self.loss_kl.item():.6f} | loss_qx={self.loss_qx.item():.6f} | "
            #         f"total_loss={self.grad_loss.item():.6f}"
            #     )

        if return_output == "hard":
            prob = self.complex_softmax(alpha, use_squared=True)
            return prob.argmax(dim=-1)

        elif return_output == "soft":
            prob = self.complex_softmax(alpha, use_squared=True)
            return prob

        elif return_output == "alpha":
            return alpha

        else:
            raise AssertionError

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
        alp0_term = torch.where(
            torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term)
        )
        # 确保所有值都是有限的
        assert torch.all(torch.isfinite(alp0_term)).item()

        # 计算KL散度的第二部分 (涉及gamma函数和digamma函数)
        alphas_term = torch.sum(
            torch.lgamma(target_alphas + epsilon)
            - torch.lgamma(alphas + epsilon)
            + (alphas - target_alphas)
            * (torch.digamma(alphas + epsilon) - torch.digamma(alp0 + epsilon)),
            dim=-1,
            keepdim=True,
        )
        # 处理可能的非有限值
        alphas_term = torch.where(
            torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term)
        )
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
        self.grad_loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新参数

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
