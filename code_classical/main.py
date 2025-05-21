import os
import logging
import torch
from seml.utils import flatten
import wandb
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from pathlib import Path
import itertools
import random
from pprint import pprint

from train import train
from dataset import get_dataset

from models.model_loader import load_model
from models.ModifiedEvidentialN import ModifiedEvidentialNet

from utils.io_utils import DataWriter
from utils.metrics import (
    accuracy,
    confidence,
    anomaly_detection,
    our_confidence,
    our_anomaly_detection,
)
from utils.metrics import compute_X_Y_alpha, name2abbrv

# 创建模型的字典，键为模型类型，值为模型类
create_model = {"menet": ModifiedEvidentialNet}
# 设置日志级别为INFO
logging.getLogger().setLevel(logging.INFO)


def main(config_dict):
    """主函数，负责整个训练和评估流程"""
    # 从配置字典中提取各种配置参数
    config_id = config_dict["config_id"]  # 配置ID
    suffix = config_dict["suffix"]  # 后缀名

    seeds = config_dict["seeds"]  # 随机种子列表

    dataset_name = config_dict["dataset_name"]  # 数据集名称
    ood_dataset_names = config_dict["ood_dataset_names"]  # 分布外(OOD)数据集名称
    split = config_dict["split"]  # 数据集划分方式

    # 模型参数
    model_type = config_dict["model_type"]  # 模型类型
    name_model_list = config_dict["name_model"]  # 模型名称列表

    # 架构参数
    directory_model = config_dict["directory_model"]  # 模型目录
    architecture = config_dict["architecture"]  # 架构类型
    input_dims = config_dict["input_dims"]  # 输入维度
    output_dim = config_dict["output_dim"]  # 输出维度
    hidden_dims = config_dict["hidden_dims"]  # 隐藏层维度
    kernel_dim = config_dict["kernel_dim"]  # 核维度
    k_lipschitz = config_dict["k_lipschitz"]  # Lipschitz常数

    # 训练参数
    max_epochs = config_dict["max_epochs"]  # 最大训练轮数
    patience = config_dict["patience"]  # 早停的耐心值
    frequency = config_dict["frequency"]  # 评估频率
    batch_size = config_dict["batch_size"]  # 批次大小
    lr_list = config_dict["lr"]  # 学习率列表
    loss = config_dict["loss"]  # 损失函数类型
    lamb1_list = config_dict["lamb1_list"]  # 正则化参数λ1列表
    lamb2_list = config_dict["lamb2_list"]  # 正则化参数λ2列表

    # 分类器类型和Fisher约束参数
    clf_type = config_dict["clf_type"]
    fisher_c_list = config_dict["fisher_c"]
    noise_epsilon = config_dict["noise_epsilon"]  # 噪声强度

    # 目录和存储设置
    model_dir = config_dict["model_dir"]  # 模型保存目录
    results_dir = config_dict["results_dir"]  # 结果保存目录
    stat_dir = config_dict["stat_dir"]  # 统计数据保存目录
    store_results = config_dict["store_results"]  # 是否保存结果
    store_stat = config_dict["store_stat"]  # 是否保存统计数据

    use_wandb = config_dict["use_wandb"]  # 是否使用Weights & Biases进行实验跟踪

    # 设置设备(GPU或CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 遍历所有的参数组合
    for setting in itertools.product(
        seeds, lr_list, fisher_c_list, name_model_list, lamb1_list, lamb2_list
    ):
        (seed, lr, fisher_c, name_model, lamb1, lamb2) = setting

        # 设置随机种子以确保可重复性
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        ## 加载数据集
        train_loader, val_loader, test_loader, N, output_dim = get_dataset(
            dataset_name, batch_size=batch_size, split=split, seed=seed
        )

        # 记录配置信息
        logging.info(f"Received the following configuration: seed {seed}")
        logging.info(
            f"DATASET | "
            f"dataset_name {dataset_name} - "
            f"ood_dataset_names {ood_dataset_names} - "
            f"split {split}"
        )

        ## 训练或加载预训练模型
        if name_model is not None:
            # 如果指定了模型名称，则加载预训练模型
            logging.info(f"MODEL: {name_model}")
            config_dict = OrderedDict(
                name_model=name_model,
                model_type=model_type,
                seed=seed,
                dataset_name=dataset_name,
                split=split,
                loss=loss,
                epsilon=noise_epsilon,
            )

            if use_wandb:
                # 初始化Weights & Biases运行
                run = wandb.init(
                    project="IEDL",
                    reinit=True,
                    group=f"{dataset_name}_{ood_dataset_names}",
                    name=f"{model_type}_{loss}_ep{noise_epsilon}_{seed}",
                )

            # 加载模型
            model = load_model(
                directory_model=directory_model,
                name_model=name_model,
                model_type=model_type,
            )
            stat_dir = stat_dir + f"{name_model}"

        else:
            # 从头开始训练新模型
            logging.info(
                f"ARCHITECTURE | "
                f" model_type {model_type} - "
                f" architecture {architecture} - "
                f" input_dims {input_dims} - "
                f" output_dim {output_dim} - "
                f" hidden_dims {hidden_dims} - "
                f" kernel_dim {kernel_dim} - "
                f" k_lipschitz {k_lipschitz}"
            )
            logging.info(
                f"TRAINING | "
                f" max_epochs {max_epochs} - "
                f" patience {patience} - "
                f" frequency {frequency} - "
                f" batch_size {batch_size} - "
                f" lr {lr} - "
                f" loss {loss}"
            )
            logging.info(
                f"MODEL PARAMETERS | "
                f" clf_type {clf_type} - "
                f" fisher_c {fisher_c} - "
                f" lamb1 {lamb1} -"
                f" lamb2 {lamb2}"
            )

            # 创建配置字典
            config_dict = OrderedDict(
                model_type=model_type,
                seed=seed,
                dataset_name=dataset_name,
                split=split,
                architecture=architecture,
                input_dims=input_dims,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                kernel_dim=kernel_dim,
                k_lipschitz=k_lipschitz,
                max_epochs=max_epochs,
                patience=patience,
                frequency=frequency,
                batch_size=batch_size,
                clf_type=clf_type,
                lr=lr,
                loss=loss,
                fisher_c=fisher_c,
                lamb1=lamb1,
                lamb2=lamb2,
            )

            if use_wandb:
                # 初始化Weights & Biases运行
                run = wandb.init(
                    project="IEDL",
                    reinit=True,
                    group=f"{__file__}_{dataset_name}_{architecture}_{suffix}",
                    name=f"{model_type}_{seed}_{loss}_lr{lr}_f{fisher_c}_{clf_type}",
                )
                wandb.config.update(config_dict)

            # 筛选模型创建所需的参数
            filtered_config_dict = {
                "seed": seed,
                "architecture": architecture,
                "input_dims": input_dims,
                "output_dim": output_dim,
                "hidden_dims": hidden_dims,
                "kernel_dim": kernel_dim,
                "k_lipschitz": k_lipschitz,
                "batch_size": batch_size,
                "lr": lr,
                "loss": loss,
                "clf_type": clf_type,
                "fisher_c": fisher_c,
                "lamb1": lamb1,
                "lamb2": lamb2,
            }

            # 创建模型
            model = create_model[model_type](**filtered_config_dict)

            if torch.cuda.is_available():
                # 如果有多个GPU，使用DataParallel
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    print(
                        f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
                    )
                    model = torch.nn.DataParallel(model)
                    model = model.module

            # 生成完整的配置名称
            full_config_name = ""
            for k, v in config_dict.items():
                if isinstance(v, dict):
                    v = flatten(v)
                    v = [str(val) for key, val in v.items()]
                    v = "-".join(v)
                if k != "name_model":
                    full_config_name += str(v) + "-"
            full_config_name = full_config_name[:-1]

            # 设置模型保存路径
            model_path = model_dir + f"{seed}"
            stat_dir = stat_dir + f"model-{full_config_name}"

            Path(model_dir).mkdir(parents=True, exist_ok=True)

            # 将模型移至设备并开始训练
            model.to(device)
            train(
                model,
                train_loader,
                val_loader,
                max_epochs=max_epochs,
                frequency=frequency,
                patience=patience,
                model_path=model_path,
                full_config_dict=config_dict,
                use_wandb=use_wandb,
                device=device,
                output_dim=output_dim,
            )

            # 加载最佳模型
            model.load_state_dict(torch.load(model_path + "_best")["model_state_dict"])

        ## 测试模型
        model.to(device)
        model.eval()  # 设置为评估模式

        with torch.no_grad():
            # 计算测试集上的预测结果
            id_Y_all, id_X_all, id_alpha_pred_all = compute_X_Y_alpha(
                model, test_loader, device
            )

            # 保存指标
            metrics = {}
            scores = {}
            ood_scores = {}
            # 计算分布内数据的准确率
            metrics["id_accuracy"] = accuracy(
                Y=id_Y_all, alpha=id_alpha_pred_all
            ).tolist()

            # 计算各种不确定性度量指标
            for name in [
                "max_prob",
                "max_modified_prob",
                "max_alpha",
                "alpha0",
                "differential_entropy",
                "mutual_information",
            ]:
                # 根据模型类型跳过某些指标
                if model_type == "duq" and name != "max_alpha":
                    continue
                if name == "max_modified_prob" and model_type != "menet":
                    continue

                abb_name = name2abbrv[name]  # 获取缩写名称
                save_path = None
                if store_stat:
                    save_path = f"{stat_dir}/{config_id}_id_{abb_name}.csv"
                    Path(stat_dir).mkdir(parents=True, exist_ok=True)

                # 根据模型类型选择适当的置信度计算方法
                if model_type == "evnet" or model_type == "duq":
                    aupr, auroc, score = confidence(
                        Y=id_Y_all,
                        alpha=id_alpha_pred_all,
                        uncertainty_type=name,
                        save_path=save_path,
                        return_scores=True,
                    )
                elif model_type == "menet" or model_type == "ablation":
                    aupr, auroc, score = our_confidence(
                        Y=id_Y_all,
                        alpha=id_alpha_pred_all,
                        uncertainty_type=name,
                        save_path=save_path,
                        return_scores=True,
                    )
                else:
                    raise NotImplementedError

                # 保存AUPR和AUROC指标
                metrics[f"id_{abb_name}_apr"], metrics[f"id_{abb_name}_auroc"] = (
                    aupr,
                    auroc,
                )
                scores[f"{abb_name}"] = score

            # 处理分布外(OOD)数据集
            ood_dataset_loaders = {}
            for ood_dataset_name in ood_dataset_names:
                config_dict["ood_dataset_name"] = ood_dataset_name
                # 加载OOD数据集
                _, _, ood_test_loader, _, _ = get_dataset(
                    ood_dataset_name, batch_size=batch_size, split=split, seed=seed
                )
                ood_dataset_loaders[ood_dataset_name] = ood_test_loader

                # 计算OOD数据的预测结果
                ood_Y_all, ood_X_all, ood_alpha_pred_all = compute_X_Y_alpha(
                    model, ood_test_loader, device, noise_epsilon=noise_epsilon
                )

                # 如果是在原始数据集上添加噪声，计算准确率
                if ood_dataset_name == dataset_name and noise_epsilon != 0:
                    metrics["ood_accuracy"] = accuracy(
                        Y=ood_Y_all, alpha=ood_alpha_pred_all
                    ).tolist()

                # 计算OOD检测指标
                for name in [
                    "max_prob",
                    "max_modified_prob",
                    "max_alpha",
                    "alpha0",
                    "differential_entropy",
                    "mutual_information",
                ]:
                    if model_type == "duq" and name != "max_alpha":
                        continue
                    if name == "max_modified_prob" and model_type != "menet":
                        continue

                    abb_name = name2abbrv[name]
                    save_path = None
                    if store_stat:
                        save_path = f"{stat_dir}/{config_id}_ood_{abb_name}.csv"

                    # 根据模型类型选择适当的异常检测方法
                    if model_type == "evnet" or model_type == "duq":
                        aupr, auroc, _, ood_score = anomaly_detection(
                            alpha=id_alpha_pred_all,
                            ood_alpha=ood_alpha_pred_all,
                            uncertainty_type=name,
                            save_path=save_path,
                            return_scores=True,
                        )
                    elif model_type == "menet" or model_type == "ablation":
                        aupr, auroc, _, ood_score = our_anomaly_detection(
                            alpha=id_alpha_pred_all,
                            ood_alpha=ood_alpha_pred_all,
                            uncertainty_type=name,
                            save_path=save_path,
                            return_scores=True,
                        )
                    else:
                        raise NotImplementedError

                    metrics[f"ood_{abb_name}_apr"], metrics[f"ood_{abb_name}_auroc"] = (
                        aupr,
                        auroc,
                    )
                    ood_scores[f"{abb_name}"] = ood_score

                # 打印指标
                print("Metrics: ")
                pprint(metrics)

                # 记录到Weights & Biases
                if use_wandb:
                    data_df = pd.DataFrame(data=[metrics])
                    wandb_table = wandb.Table(dataframe=data_df)
                    wandb.log({"{}".format(ood_dataset_name): wandb_table})

                # 保存结果到CSV文件
                if store_results:
                    row_dict = config_dict.copy()
                    for k, v in config_dict.items():
                        if isinstance(v, list):
                            row_dict[k] = str(v)

                    row_dict.update(metrics)  # 浅拷贝

                    Path(results_dir).mkdir(parents=True, exist_ok=True)
                    data_writer = DataWriter(dump_period=1)
                    csv_file = f"{results_dir}/{config_id}.csv"
                    data_writer.add(row_dict, csv_file)

        # 结束Weights & Biases运行
        if use_wandb:
            run.finish()

    return


if __name__ == "__main__":
    # 是否使用命令行参数
    use_argparse = True

    if use_argparse:
        # 解析命令行参数
        import argparse

        my_parser = argparse.ArgumentParser()
        my_parser.add_argument("--configid", action="store", type=str, required=True)
        my_parser.add_argument("--suffix", type=str, default="debug", required=False)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_suffix = args.suffix
    else:
        # 使用默认值
        args_configid = "test"
        args_suffix = "debug"

    # 处理配置ID路径
    if "/" in args_configid:
        args_configid_split = args_configid.split("/")
        my_config_id = args_configid_split[-1]
        config_tree = "/".join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ""

    # 设置项目路径和配置文件路径
    PROJPATH = os.getcwd()
    cfg_dir = f"{PROJPATH}/configs"
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = f"{PROJPATH}/configs/{config_tree}/{my_config_id}.json"
    logging.info(f"Reading Configuration from {cfg_path}")

    # 读取配置文件
    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    # 添加额外的配置参数
    proced_config_dict["config_id"] = my_config_id
    proced_config_dict["suffix"] = args_suffix

    # 设置保存路径
    proced_config_dict["model_dir"] = f"{PROJPATH}/saved_models/{my_config_id}/"
    proced_config_dict["results_dir"] = f"{PROJPATH}/saved_models/{my_config_id}/"
    proced_config_dict["stat_dir"] = f"{PROJPATH}/results/{config_tree}_stat/"

    # 运行主函数
    main(proced_config_dict)