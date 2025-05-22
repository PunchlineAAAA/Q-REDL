import torch
import numpy as np
import wandb
from utils.encoding import encode_complex


def compute_loss_accuracy(model, loader, epoch, device=torch.device("cpu"), is_fisher=False):
    """
    计算模型在给定数据集上的损失和准确率
    
    参数:
        model: 要评估的模型
        loader: 数据加载器
        epoch: 当前训练的轮次
        device: 计算设备(CPU/GPU)
        is_fisher: 是否使用Fisher信息损失
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度以提高效率和减少内存使用
        total_loss_ = 0.0

        # 如果使用Fisher信息损失，初始化各个损失组件
        if is_fisher:
            loss_mse_ = 0.0   # 均方误差损失
            loss_var_ = 0.0   # 方差损失
            loss_kl_ = 0.0    # KL散度损失
            loss_fisher_ = 0.0  # Fisher信息损失

        # 遍历数据批次
        for batch_index, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)  # 将数据移到指定设备

            X = encode_complex(X, method="rect")

            # 根据模型损失类型决定前向传播方式
            if model.loss == 'DUQ':
                Y_pred = model(X, Y, return_output='hard', compute_loss=False, epoch=epoch)
            else:
                Y_pred = model(X, Y, return_output='hard', compute_loss=True, epoch=epoch)

            # 收集所有批次的预测和真实标签，用于计算整体准确率
            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)

            # 累加损失
            total_loss_ += model.grad_loss.item()

            # 对于Fisher损失，累加各个损失组件
            if is_fisher:
                loss_mse_ += model.loss_mse_.item()
                loss_var_ += model.loss_var_.item()
                loss_kl_ += model.loss_kl_.item()
                loss_fisher_ += model.loss_fisher_.item()

        # 计算平均损失
        total_loss_ = total_loss_ / Y_pred_all.size(0)
        if is_fisher:
            loss_mse_ = loss_mse_ / Y_pred_all.size(0)
            loss_var_ = loss_var_ / Y_pred_all.size(0)
            loss_kl_ = loss_kl_ / Y_pred_all.size(0)
            loss_fisher_ = loss_fisher_ / Y_pred_all.size(0)

        # 打印预测类别分布
        # print("Y_pred counts:", torch.bincount(Y_pred_all))
        
        # 计算准确率
        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()

    model.train()  # 将模型恢复为训练模式
    # 根据损失类型返回不同的结果
    if is_fisher:
        return accuracy, total_loss_, loss_mse_, loss_var_, loss_kl_, loss_fisher_
    else:
        return accuracy, total_loss_


def train(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=5, model_path='saved_model',
          full_config_dict={}, use_wandb=False, device=torch.device("cpu"), is_fisher=False, output_dim=10):
    """
    训练神经网络模型的主函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        max_epochs: 最大训练轮次
        frequency: 评估和记录的频率（每frequency个epoch进行一次）
        patience: 早停的耐心值（当前未使用）
        model_path: 保存模型的路径前缀
        full_config_dict: 包含模型和数据集配置的字典
        use_wandb: 是否使用Weights & Biases进行实验跟踪
        device: 计算设备(CPU/GPU)
        is_fisher: 是否使用Fisher信息损失
        output_dim: 输出维度（未使用）
    """
    model.to(device)  # 将模型移到指定设备
    model.train()  # 设置模型为训练模式
    val_losses, val_accuracies = [], []  # 存储验证损失和准确率历史
    best_val_loss = float("Inf")  # 初始化最佳验证损失为无穷大
    val_loss = float("Inf")  # 初始化当前验证损失为无穷大
    best_val_acc = 0.0  # 初始化最佳验证准确率为0
    epoch = 0  # 初始化epoch计数器

    # 主训练循环
    for epoch in range(max_epochs):
        # 遍历训练数据批次
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            X_train, Y_train = X_train.to(device), Y_train.to(device)  # 将数据移到指定设备

            X_train = encode_complex(X_train, method="rect")

            model.train()  # 确保模型处于训练模式
            # 前向传播并计算损失
            model(X_train, Y_train, compute_loss=True, epoch=epoch)
            model.step()  # 执行优化步骤
            # model.module.step()  # 用于分布式训练的代码（当前未使用）

        # 对于特定数据集和模型类型，更新学习率
        if full_config_dict['dataset_name'] == 'MNIST' or full_config_dict['model_type'] == 'DUQ':
            model.scheduler.step()

        # 按指定频率进行评估和记录
        if epoch % frequency == 0:
            # 计算训练集上的统计数据并记录到wandb（如果启用）
            if use_wandb:
                if is_fisher:
                    # 对于Fisher损失，计算和记录多个损失组件
                    train_accuracy, total_loss_, loss_mse_, loss_var_, loss_kl_, loss_fisher_ = compute_loss_accuracy(
                        model, train_loader, epoch, device=device, is_fisher=True)
                    wandb.log({'Train/total_loss_': round(total_loss_, 3), 'Train/loss_mse_': round(loss_mse_, 3),
                               'Train/loss_var_': round(loss_var_, 3), 'Train/loss_kl_': round(loss_kl_, 3),
                               'Train/loss_fisher_': round(loss_fisher_, 3), 'Train/Acc': round(train_accuracy * 100, 3),
                               'Train/epoch': epoch + 1})
                else:
                    # 对于标准损失，计算和记录损失和准确率
                    train_accuracy, total_loss_ = compute_loss_accuracy(model, train_loader, epoch, device=device)
                    wandb.log({'Train/total_loss_': round(total_loss_, 3), 'Train/Acc': round(train_accuracy * 100, 3),
                               'Train/epoch': epoch + 1})

            # 在验证集上计算性能
            val_accuracy, val_loss = compute_loss_accuracy(model, val_loader, epoch, device=device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # 在wandb中记录验证性能（如果启用）
            if use_wandb:
                wandb.log({'Val/total_loss_': round(val_loss * 100, 3), 'Val/Acc': round(val_accuracy * 100, 3),
                           'Val/epoch': epoch + 1})

            # 打印当前验证性能（蓝色文字）
            print("\033[34m Epoch {} -> Val loss {} | Val Acc. {}% | Best Val Acc. {}%\033[0m".format(
                epoch,
                round(val_losses[-1] * 100, 3),
                round(val_accuracies[-1] * 100, 3),
                round(best_val_acc * 100, 3)))

            # 如果当前验证准确率是最好的，保存模型
            if best_val_acc < val_accuracies[-1]:
                best_val_acc = val_accuracies[-1]
                torch.save(
                    {'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(),
                     'loss': best_val_loss}, f"{model_path}_best")
                print(f'Best model saved, Epoch: {epoch}')

            # 检测NaN损失并中断训练
            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            # 早停逻辑（未使用）
            # if int(epoch / frequency) > patience and val_accuracies[-patience] >= max(val_accuracies[-patience:]):
            #     print('Early Stopping.')
            #     break

    # 保存最终模型
    torch.save(
        {'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(),
         'loss': val_loss}, f"{model_path}_last")
    print(f'Last model saved, Epoch: {epoch}')

    return