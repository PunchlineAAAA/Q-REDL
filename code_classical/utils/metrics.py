import torch
import numpy as np
from torch.distributions import Categorical
from torch.distributions import Dirichlet
from sklearn import metrics
import wandb
import pandas as pd
from PIL import Image as im
from utils.encoding import encode_complex

name2abbrv = {
    "max_prob": "max_prob",
    "max_alpha": "max_alpha",
    "max_modified_prob": "max_modified_prob",
    "alpha0": "alpha0",
    "precision": "alpha0",
    "differential_entropy": "diff_ent",
    "mutual_information": "mi",
    "quantum_x_entropy": "qx_ent",
    "quantum_discord": "qd",
    "quantum_nonspecificity": "qn",
}


def compute_X_Y_alpha(model, loader, device, noise_epsilon=0.0, complex_support=True):
    """
    支持量子证据理论的复数alpha增强版本
    """
    X_all, Y_all, model_pred_all = [], [], []

    for batch_index, (X, Y) in enumerate(loader):
        X = X.to(torch.float32)
        X = (X + noise_epsilon * torch.randn_like(X)).to(device)
        if not torch.is_complex(X):
            X = encode_complex(X, method="rect")
        Y = Y.to(device)

        model_pred = model(X, None, return_output="alpha", compute_loss=False)

        # 如果指定，支持复数alpha
        if complex_support and torch.is_complex(model_pred):
            # 保持复数结构
            model_pred = model_pred
        else:
            # 如果量子框架需要，转换为复数
            if not torch.is_complex(model_pred):
                model_pred = model_pred.to(torch.complex64)

        X_all.append(X.to("cpu"))
        Y_all.append(Y.to("cpu"))
        model_pred_all.append(model_pred.to("cpu"))

    X_all = torch.cat(X_all, dim=0)
    Y_all = torch.cat(Y_all, dim=0)
    model_pred_all = torch.cat(model_pred_all, dim=0)

    return Y_all, X_all, model_pred_all


def quantum_density_operator(alpha):
    """
    计算广义量子质量函数的量子密度算子
    参数:
        alpha: 表示量子证据参数的复数张量
    返回:
        归一化密度算子
    """
    # 如果不是复数则转换为复数
    if not torch.is_complex(alpha):
        alpha = alpha.to(torch.complex64)

    # 计算平方幅度用于归一化
    alpha_squared = torch.abs(alpha) ** 2

    # 归一化得到类概率分布
    alpha_norm = alpha_squared / torch.sum(alpha_squared, dim=-1, keepdim=True)

    return alpha_norm, alpha


def quantum_x_entropy(alpha, return_components=False):
    """
    计算论文中定义的量子X熵
    X(Q_M) = -∑_ν Ê_ν log(Ê_ν/(2^d̂_ν - 1))

    参数:
        alpha: 复数量子证据参数
        return_components: 是否分别返回不一致性和非特异性
    """
    # 获取归一化密度算子
    alpha_norm, alpha_complex = quantum_density_operator(alpha)

    eps = 1e-8
    alpha_norm = alpha_norm + eps

    # 类别数量（结构维度）
    num_classes = alpha.shape[-1]

    # 不一致/冲突项: -∑ Ê_ν log Ê_ν (冯·诺依曼熵部分)
    discord = -torch.sum(alpha_norm * torch.log(alpha_norm + eps), dim=-1)

    # 非特异性/同时性项: ∑ Ê_ν log(2^d̂_ν - 1)
    # 为简化起见，我们使用num_classes作为结构维度
    structural_factor = torch.log(
        torch.tensor(2.0**num_classes - 1, dtype=torch.float32)
    )
    nonspecificity = torch.sum(alpha_norm, dim=-1) * structural_factor

    # 总量子X熵
    x_entropy = discord + nonspecificity

    if return_components:
        return x_entropy, discord, nonspecificity
    return x_entropy


def quantum_belief_entropy(alpha):
    """
    计算量子信念熵（冯·诺依曼熵部分）
    S_X(Q_M) = -tr(ρ̂_Q_M log ρ̂_Q_M)
    """
    alpha_norm, _ = quantum_density_operator(alpha)
    eps = 1e-8
    alpha_norm = alpha_norm + eps

    # 冯·诺依曼熵
    entropy = -torch.sum(alpha_norm * torch.log(alpha_norm), dim=-1)
    return entropy


def quantum_nonspecificity(alpha):
    """
    计算量子非特异性度量
    X_N(Q_M) = ∑ Ê_ν log(2^d̂_ν - 1)
    """
    alpha_norm, _ = quantum_density_operator(alpha)
    num_classes = alpha.shape[-1]

    structural_factor = torch.log(
        torch.tensor(2.0**num_classes - 1, dtype=torch.float32)
    )
    nonspecificity = torch.sum(alpha_norm, dim=-1) * structural_factor

    return nonspecificity


def accuracy(Y, alpha):
    """支持复数alpha的增强准确率计算"""
    if torch.is_complex(alpha):
        # 使用幅度进行预测
        alpha_mag = torch.abs(alpha)
        corrects = (Y.squeeze() == alpha_mag.max(-1)[1]).type(torch.DoubleTensor)
    else:
        corrects = (Y.squeeze() == alpha.max(-1)[1]).type(torch.DoubleTensor)

    accuracy = corrects.sum() / corrects.size(0)
    return accuracy.cpu().detach().numpy()


def enhanced_confidence(
    Y, alpha, uncertainty_type="quantum_x_entropy", save_path=None, return_scores=False
):
    """
    支持量子证据理论的增强置信度度量
    """
    if torch.is_complex(alpha):
        alpha_mag = torch.abs(alpha)
        corrects = (Y.squeeze() == alpha_mag.max(-1)[1]).cpu().detach().numpy()
    else:
        alpha_mag = alpha
        corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == "quantum_x_entropy":
        x_entropy = quantum_x_entropy(alpha)
        scores = -x_entropy.cpu().detach().numpy()  # 置信度用负值

    elif uncertainty_type == "quantum_discord":
        discord = quantum_belief_entropy(alpha)
        scores = -discord.cpu().detach().numpy()

    elif uncertainty_type == "quantum_nonspecificity":
        nonspec = quantum_nonspecificity(alpha)
        scores = -nonspec.cpu().detach().numpy()

    elif uncertainty_type == "max_alpha":
        if torch.is_complex(alpha):
            scores = alpha_mag.max(-1)[0].cpu().detach().numpy()
        else:
            scores = alpha.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_prob":
        if torch.is_complex(alpha):
            p = alpha_mag / torch.sum(alpha_mag, dim=-1, keepdim=True)
        else:
            denom = torch.sum(alpha, dim=-1, keepdim=True).clamp(min=1e-6)
            p = alpha / denom
        scores = p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "alpha0":
        if torch.is_complex(alpha):
            scores = alpha_mag.sum(-1).cpu().detach().numpy()
        else:
            scores = alpha.sum(-1).cpu().detach().numpy()

    elif uncertainty_type == "differential_entropy":
        if torch.is_complex(alpha):
            alpha_work = alpha_mag
        else:
            alpha_work = alpha

        eps = 1e-6
        alpha_work = alpha_work + eps
        alpha0 = alpha_work.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha_work), dim=-1) - torch.lgamma(alpha0)
        digamma_term = torch.sum(
            (alpha_work - 1.0)
            * (
                torch.digamma(alpha_work)
                - torch.digamma(
                    (alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha_work)
                )
            ),
            dim=-1,
        )
        differential_entropy = log_term - digamma_term
        scores = -differential_entropy.cpu().detach().numpy()

    elif uncertainty_type == "mutual_information":
        if torch.is_complex(alpha):
            alpha_work = alpha_mag
        else:
            alpha_work = alpha

        eps = 1e-6
        alpha_work = alpha_work + eps
        alpha0 = alpha_work.sum(-1)
        probs = alpha_work / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha_work)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1)
        digamma_term = torch.digamma(alpha_work + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha_work) + 1.0
        )
        dirichlet_mean = alpha_work / alpha0.reshape((alpha0.size()[0], 1)).expand_as(
            alpha_work
        )
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = -distributional_uncertainty.cpu().detach().numpy()

    else:
        raise ValueError(f"无效的不确定性类型: {uncertainty_type}!")

    if save_path is not None:
        if uncertainty_type in [
            "differential_entropy",
            "mutual_information",
            "quantum_x_entropy",
            "quantum_discord",
        ]:
            unc = -scores
        else:
            unc = scores

        scores_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc) + 1e-8)
        np.save(save_path, scores_norm)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)

    if return_scores:
        return aupr, auroc, scores
    else:
        return metrics.auc(fpr, tpr)


def quantum_anomaly_detection(
    alpha,
    ood_alpha,
    uncertainty_type="quantum_x_entropy",
    save_path=None,
    return_scores=False,
):
    """
    基于量子证据理论的增强异常检测
    """
    if uncertainty_type == "quantum_x_entropy":
        id_scores = -quantum_x_entropy(alpha).cpu().detach().numpy()
        ood_scores = -quantum_x_entropy(ood_alpha).cpu().detach().numpy()

    elif uncertainty_type == "quantum_discord":
        id_scores = -quantum_belief_entropy(alpha).cpu().detach().numpy()
        ood_scores = -quantum_belief_entropy(ood_alpha).cpu().detach().numpy()

    elif uncertainty_type == "quantum_nonspecificity":
        id_scores = -quantum_nonspecificity(alpha).cpu().detach().numpy()
        ood_scores = -quantum_nonspecificity(ood_alpha).cpu().detach().numpy()

    elif uncertainty_type == "alpha0":
        if torch.is_complex(alpha):
            id_scores = torch.abs(alpha).sum(-1).cpu().detach().numpy()
            ood_scores = torch.abs(ood_alpha).sum(-1).cpu().detach().numpy()
        else:
            id_scores = alpha.sum(-1).cpu().detach().numpy()
            ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()

    elif uncertainty_type == "max_alpha":
        if torch.is_complex(alpha):
            id_scores = torch.abs(alpha).max(-1)[0].cpu().detach().numpy()
            ood_scores = torch.abs(ood_alpha).max(-1)[0].cpu().detach().numpy()
        else:
            id_scores = alpha.max(-1)[0].cpu().detach().numpy()
            ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_prob":
        if torch.is_complex(alpha):
            alpha_mag = torch.abs(alpha)
            ood_alpha_mag = torch.abs(ood_alpha)
            p = alpha_mag / torch.sum(alpha_mag, dim=-1, keepdim=True)
            ood_p = ood_alpha_mag / torch.sum(ood_alpha_mag, dim=-1, keepdim=True)
        else:
            denom = torch.sum(alpha, dim=-1, keepdim=True).clamp(min=1e-6)
            p = alpha / denom
            ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)

        id_scores = p.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    else:
        # 对于其他不确定性类型，回退到原始方法
        return anomaly_detection(
            alpha, ood_alpha, uncertainty_type, save_path, return_scores
        )

    corrects = np.concatenate(
        [np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0
    )
    scores = np.concatenate([id_scores, ood_scores], axis=0)

    if save_path is not None:
        if uncertainty_type in ["quantum_x_entropy", "quantum_discord"]:
            scores_norm = (-scores - np.min(-scores)) / (
                np.max(-scores) - np.min(-scores) + 1e-8
            )
        else:
            scores_norm = (scores - np.min(scores)) / (
                np.max(scores) - np.min(scores) + 1e-8
            )
        np.save(save_path, scores_norm)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)

    if return_scores:
        return aupr, auroc, id_scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)


# 保留原始函数以保证向后兼容性
def confidence(
    Y, alpha, uncertainty_type="max_prob", save_path=None, return_scores=False
):
    """向后兼容的原始置信度函数"""
    if torch.is_complex(alpha):
        alpha = torch.abs(alpha)  # 或 alpha.real 取决于你的任务定义
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == "max_alpha":
        scores = alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == "max_prob":
        denom = torch.sum(alpha, dim=-1, keepdim=True).clamp(min=1e-6)
        p = alpha / denom
        scores = p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == "alpha0":
        scores = alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == "differential_entropy":
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        digamma_term = torch.sum(
            (alpha - 1.0)
            * (
                torch.digamma(alpha)
                - torch.digamma(
                    (alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha)
                )
            ),
            dim=-1,
        )
        differential_entropy = log_term - digamma_term
        scores = -differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == "mutual_information":
        eps = 1e-6
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=-1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0
        )
        dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        exp_data_uncertainty = -1 * torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = -distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"无效的不确定性类型: {uncertainty_type}!")

    if save_path is not None:
        if uncertainty_type in ["differential_entropy", "mutual_information"]:
            unc = -scores
        else:
            unc = scores

        scores_norm = (unc - min(unc)) / (max(unc) - min(unc))
        np.save(save_path, scores_norm)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores
    else:
        return metrics.auc(fpr, tpr)


# 保留其他原始函数
def our_confidence(
    Y,
    alpha,
    uncertainty_type="max_prob",
    save_path=None,
    return_scores=False,
    lamb1=1.0,
    lamb2=1.0,
):
    """鲁棒增强版 our_confidence 函数，支持复数 Dirichlet α，确保不会因 NaN/Inf 报错"""
    if torch.is_complex(alpha):
        alpha = alpha.real  # 仅保留实部用于分类

    eps = 1e-6
    corrects = (Y.squeeze() == alpha.max(-1)[1]).cpu().detach().numpy()

    if uncertainty_type == "max_alpha":
        scores = alpha.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_prob":
        denom = torch.sum(alpha, dim=-1, keepdim=True).clamp(min=eps)
        p = alpha / denom
        scores = p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_modified_prob":
        num_classes = alpha.shape[-1]
        evidence = alpha - lamb2
        S = (
            evidence
            + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence)
            + lamb2 * num_classes
        ).clamp(min=eps)
        p = alpha / S
        scores = p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "alpha0":
        scores = alpha.sum(-1).cpu().detach().numpy()

    elif uncertainty_type == "differential_entropy":
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        digamma_term = torch.sum(
            (alpha - lamb2)
            * (
                torch.digamma(alpha)
                - torch.digamma(alpha0.unsqueeze(-1).expand_as(alpha))
            ),
            dim=-1,
        )
        differential_entropy = log_term - digamma_term
        scores = -differential_entropy.cpu().detach().numpy()

    elif uncertainty_type == "mutual_information":
        alpha = alpha + eps
        alpha0 = alpha.sum(-1)
        probs = alpha / alpha0.unsqueeze(-1).expand_as(alpha)
        total_uncertainty = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.unsqueeze(-1).expand_as(alpha) + 1.0
        )
        dirichlet_mean = probs
        exp_data_uncertainty = -torch.sum(dirichlet_mean * digamma_term, dim=-1)
        distributional_uncertainty = total_uncertainty - exp_data_uncertainty
        scores = -distributional_uncertainty.cpu().detach().numpy()

    else:
        raise ValueError(f"无效的不确定性类型: {uncertainty_type}!")

    # ✅ 数值稳定处理
    scores = scores.astype(np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    corrects = np.nan_to_num(corrects, nan=0.0)

    # ✅ 安全过滤非法值（保险）
    valid_mask = np.isfinite(scores) & np.isfinite(corrects)
    scores = scores[valid_mask]
    corrects = corrects[valid_mask]

    if save_path is not None:
        unc = (
            -scores
            if uncertainty_type in ["differential_entropy", "mutual_information"]
            else scores
        )
        scores_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc) + eps)
        np.save(save_path, scores_norm)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)

    if return_scores:
        return aupr, auroc, scores
    else:
        return auroc


def brier_score(Y, alpha):
    """支持复数alpha的增强Brier分数"""
    batch_size = alpha.size(0)

    if torch.is_complex(alpha):
        alpha_mag = torch.abs(alpha)
        p = torch.nn.functional.normalize(alpha_mag, p=1, dim=-1)
    else:
        p = torch.nn.functional.normalize(alpha, p=1, dim=-1)

    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score


def anomaly_detection(
    alpha, ood_alpha, uncertainty_type="max_prob", save_path=None, return_scores=False
):
    """原始异常检测函数"""
    if torch.is_complex(alpha):
        alpha = torch.abs(alpha)
        ood_alpha = torch.abs(ood_alpha)

    if uncertainty_type == "alpha0":
        scores = alpha.sum(-1).cpu().detach().numpy()
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()
    elif uncertainty_type == "max_alpha":
        scores = alpha.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == "max_prob":
        denom = torch.sum(alpha, dim=-1, keepdim=True).clamp(min=1e-6)
        p = alpha / denom
        scores = p.max(-1)[0].cpu().detach().numpy()

        ood_p = ood_alpha / torch.sum(ood_alpha, dim=-1, keepdim=True)
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()
    elif uncertainty_type == "differential_entropy":
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        id_digamma_term = torch.sum(
            (alpha - 1.0)
            * (
                torch.digamma(alpha)
                - torch.digamma(
                    (alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha)
                )
            ),
            dim=-1,
        )
        id_differential_entropy = id_log_term - id_digamma_term

        ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(
            ood_alpha0
        )
        ood_digamma_term = torch.sum(
            (ood_alpha - 1.0)
            * (
                torch.digamma(ood_alpha)
                - torch.digamma(
                    (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha)
                )
            ),
            dim=-1,
        )
        ood_differential_entropy = ood_log_term - ood_digamma_term

        scores = -id_differential_entropy.cpu().detach().numpy()
        ood_scores = -ood_differential_entropy.cpu().detach().numpy()
    elif uncertainty_type == "mutual_information":
        eps = 1e-6
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)
        probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
        ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(
            ood_alpha
        )

        id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
        id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
            alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0
        )
        id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(
            alpha
        )
        id_exp_data_uncertainty = -1 * torch.sum(
            id_dirichlet_mean * id_digamma_term, dim=1
        )
        id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

        ood_total_uncertainty = -1 * torch.sum(
            ood_probs * torch.log(ood_probs + 0.00001), dim=1
        )
        ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
            ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0
        )
        ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape(
            (ood_alpha0.size()[0], 1)
        ).expand_as(ood_alpha)
        ood_exp_data_uncertainty = -1 * torch.sum(
            ood_dirichlet_mean * ood_digamma_term, dim=1
        )
        ood_distributional_uncertainty = (
            ood_total_uncertainty - ood_exp_data_uncertainty
        )

        scores = -id_distributional_uncertainty.cpu().detach().numpy()
        ood_scores = -ood_distributional_uncertainty.cpu().detach().numpy()
    else:
        raise ValueError(f"无效的不确定性类型: {uncertainty_type}!")

    corrects = np.concatenate(
        [np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0
    )
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        if uncertainty_type in ["differential_entropy", "mutual_information"]:
            scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))
        else:
            scores_norm = (scores - min(scores)) / (max(scores) - min(scores))

        np.save(save_path, scores_norm)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)


def our_anomaly_detection(
    alpha,
    ood_alpha,
    uncertainty_type="max_prob",
    save_path=None,
    return_scores=False,
    lamb1=1.0,
    lamb2=1.0,
):
    """增强版 our_anomaly_detection，支持复数并避免 NaN/inf 问题"""
    eps = 1e-6

    if torch.is_complex(alpha):
        alpha = alpha.real
        ood_alpha = ood_alpha.real

    def safe_probs(tensor):
        denom = torch.sum(tensor, dim=-1, keepdim=True).clamp(min=eps)
        return tensor / denom

    if uncertainty_type == "alpha0":
        id_scores = alpha.sum(-1).cpu().detach().numpy()
        ood_scores = ood_alpha.sum(-1).cpu().detach().numpy()

    elif uncertainty_type == "max_alpha":
        id_scores = alpha.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_alpha.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_prob":
        p = safe_probs(alpha)
        ood_p = safe_probs(ood_alpha)
        id_scores = p.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "max_modified_prob":
        num_classes = alpha.shape[-1]

        def compute_mod_prob(t):
            evidence = t - lamb2
            S = (
                evidence
                + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence)
                + lamb2 * num_classes
            ).clamp(min=eps)
            return t / S

        p = compute_mod_prob(alpha)
        ood_p = compute_mod_prob(ood_alpha)
        id_scores = p.max(-1)[0].cpu().detach().numpy()
        ood_scores = ood_p.max(-1)[0].cpu().detach().numpy()

    elif uncertainty_type == "differential_entropy":
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
        id_digamma_term = torch.sum(
            (alpha - lamb2)
            * (
                torch.digamma(alpha)
                - torch.digamma(alpha0.unsqueeze(-1).expand_as(alpha))
            ),
            dim=-1,
        )
        id_entropy = id_log_term - id_digamma_term

        ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(
            ood_alpha0
        )
        ood_digamma_term = torch.sum(
            (ood_alpha - lamb2)
            * (
                torch.digamma(ood_alpha)
                - torch.digamma(ood_alpha0.unsqueeze(-1).expand_as(ood_alpha))
            ),
            dim=-1,
        )
        ood_entropy = ood_log_term - ood_digamma_term

        id_scores = -id_entropy.cpu().detach().numpy()
        ood_scores = -ood_entropy.cpu().detach().numpy()

    elif uncertainty_type == "mutual_information":
        alpha = alpha + eps
        ood_alpha = ood_alpha + eps
        alpha0 = alpha.sum(-1)
        ood_alpha0 = ood_alpha.sum(-1)

        probs = alpha / alpha0.unsqueeze(-1).expand_as(alpha)
        ood_probs = ood_alpha / ood_alpha0.unsqueeze(-1).expand_as(ood_alpha)

        def mutual_info(probs, alpha, alpha0):
            total_uncertainty = -torch.sum(probs * torch.log(probs + eps), dim=-1)
            digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
                alpha0.unsqueeze(-1).expand_as(alpha) + 1.0
            )
            dirichlet_mean = probs
            exp_data_uncertainty = -torch.sum(dirichlet_mean * digamma_term, dim=-1)
            return total_uncertainty - exp_data_uncertainty

        id_mi = mutual_info(probs, alpha, alpha0)
        ood_mi = mutual_info(ood_probs, ood_alpha, ood_alpha0)

        id_scores = -id_mi.cpu().detach().numpy()
        ood_scores = -ood_mi.cpu().detach().numpy()

    else:
        raise ValueError(f"无效的不确定性类型: {uncertainty_type}")

    # ✅ 数值修复
    id_scores = np.nan_to_num(id_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    ood_scores = np.nan_to_num(ood_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    corrects = np.concatenate(
        [np.ones_like(id_scores), np.zeros_like(ood_scores)], axis=0
    )
    scores = np.concatenate([id_scores, ood_scores], axis=0)

    # ✅ 安全过滤
    valid_mask = np.isfinite(scores) & np.isfinite(corrects)
    scores = scores[valid_mask]
    corrects = corrects[valid_mask]

    if save_path is not None:
        scores_norm = (scores - np.min(scores)) / (
            np.max(scores) - np.min(scores) + eps
        )
        np.save(save_path, scores_norm)

    fpr, tpr, _ = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)

    if return_scores:
        return aupr, auroc, id_scores, ood_scores
    else:
        return auroc


def entropy(alpha, uncertainty_type, n_bins=10, plot=True):
    """支持复数alpha的增强熵计算"""
    entropy = []

    if torch.is_complex(alpha):
        alpha_work = torch.abs(alpha)
    else:
        alpha_work = alpha

    if uncertainty_type == "categorical":
        p = torch.nn.functional.normalize(alpha_work, p=1, dim=-1)
        entropy.append(Categorical(p).entropy().squeeze().cpu().detach().numpy())
    elif uncertainty_type == "dirichlet":
        entropy.append(Dirichlet(alpha_work).entropy().squeeze().cpu().detach().numpy())

    return entropy


# 保留原始的附加度量函数
def diff_entropy(alpha, ood_alpha, save_path=None, return_scores=False):
    """原始 diff_entropy 函数"""
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)

    id_log_term = torch.sum(torch.lgamma(alpha), dim=-1) - torch.lgamma(alpha0)
    id_digamma_term = torch.sum(
        (alpha - 1.0)
        * (
            torch.digamma(alpha)
            - torch.digamma((alpha0.reshape((alpha0.size()[0], 1))).expand_as(alpha))
        ),
        dim=-1,
    )
    id_differential_entropy = id_log_term - id_digamma_term

    ood_log_term = torch.sum(torch.lgamma(ood_alpha), dim=-1) - torch.lgamma(ood_alpha0)
    ood_digamma_term = torch.sum(
        (ood_alpha - 1.0)
        * (
            torch.digamma(ood_alpha)
            - torch.digamma(
                (ood_alpha0.reshape((ood_alpha0.size()[0], 1))).expand_as(ood_alpha)
            )
        ),
        dim=-1,
    )
    ood_differential_entropy = ood_log_term - ood_digamma_term

    scores = -id_differential_entropy.cpu().detach().numpy()
    ood_scores = -ood_differential_entropy.cpu().detach().numpy()

    corrects = np.concatenate(
        [np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0
    )
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))

        results = np.concatenate(
            [corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1
        )
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)


def dist_uncertainty(alpha, ood_alpha, save_path=None, return_scores=False):
    """原始 dist_uncertainty 函数"""
    eps = 1e-6
    alpha = alpha + eps
    ood_alpha = ood_alpha + eps
    alpha0 = alpha.sum(-1)
    ood_alpha0 = ood_alpha.sum(-1)
    probs = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    ood_probs = ood_alpha / ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(
        ood_alpha
    )

    id_total_uncertainty = -1 * torch.sum(probs * torch.log(probs + 0.00001), dim=1)
    id_digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(
        alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha) + 1.0
    )
    id_dirichlet_mean = alpha / alpha0.reshape((alpha0.size()[0], 1)).expand_as(alpha)
    id_exp_data_uncertainty = -1 * torch.sum(id_dirichlet_mean * id_digamma_term, dim=1)
    id_distributional_uncertainty = id_total_uncertainty - id_exp_data_uncertainty

    ood_total_uncertainty = -1 * torch.sum(
        ood_probs * torch.log(ood_probs + 0.00001), dim=1
    )
    ood_digamma_term = torch.digamma(ood_alpha + 1.0) - torch.digamma(
        ood_alpha0.reshape((ood_alpha0.size()[0], 1)).expand_as(ood_alpha) + 1.0
    )
    ood_dirichlet_mean = ood_alpha / ood_alpha0.reshape(
        (ood_alpha0.size()[0], 1)
    ).expand_as(ood_alpha)
    ood_exp_data_uncertainty = -1 * torch.sum(
        ood_dirichlet_mean * ood_digamma_term, dim=1
    )
    ood_distributional_uncertainty = ood_total_uncertainty - ood_exp_data_uncertainty

    scores = -id_distributional_uncertainty.cpu().detach().numpy()
    ood_scores = -ood_distributional_uncertainty.cpu().detach().numpy()

    corrects = np.concatenate(
        [np.ones(alpha.size(0)), np.zeros(ood_alpha.size(0))], axis=0
    )
    scores = np.concatenate([scores, ood_scores], axis=0)

    if save_path is not None:
        scores_norm = (-scores - min(-scores)) / (max(-scores) - min(-scores))

        results = np.concatenate(
            [corrects.reshape(-1, 1), scores_norm.reshape(-1, 1)], axis=-1
        )
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path)

    fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
    auroc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(corrects, scores)
    if return_scores:
        return aupr, auroc, scores, ood_scores
    else:
        return metrics.auc(fpr, tpr)
