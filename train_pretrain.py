# 导入必要的库和模块
import os
import platform
import argparse  # 用于解析命令行参数
import time
import math
import warnings  # 忽略警告信息
import pandas as pd
import torch
import torch.distributed as dist  # 分布式训练相关
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行模型
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样器
from contextlib import nullcontext  # 上下文管理器

from transformers import AutoTokenizer  # HuggingFace的tokenizer

# 导入自定义模块
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')  # 忽略所有警告


def Logger(content):
    """分布式训练环境下的日志打印函数（仅主进程打印）"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """带预热的余弦退火学习率调度函数
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """单个训练周期的完整流程
    Args:
        epoch: 当前epoch序号
        wandb: wandb日志对象
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 不自动求平均的交叉熵损失
    start_time = time.time()
    
    # 遍历训练数据
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度训练上下文
        with ctx:
            # 前向传播
            res = model(X)
            # 计算masked loss
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用loss mask
            loss += res.aux_loss  # 添加辅助损失（如MoE的负载均衡损失）
            loss = loss / args.accumulation_steps  # 梯度累积归一化

        # 反向传播（带梯度缩放）
        scaler.scale(loss).backward()

        # 梯度累积更新判断
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放以进行梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 参数更新
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 还原实际loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录到wandb（仅主进程）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''  # MoE模型特殊标记
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            # 处理分布式模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# 分布式训练启动命令示例：torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 输出目录
    parser.add_argument("--out_dir", type=str, default="out")
    # 训练轮次（设置为1可实现快速运行）
    parser.add_argument("--epochs", type=int, default=1)
    # 批量大小
    parser.add_argument("--batch_size", type=int, default=32)
    # 基础学习率
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # 训练设备
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练精度（bfloat16/float16）
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 是否使用wandb
    parser.add_argument("--use_wandb", action="store_true")
    # wandb项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    # 数据加载线程数
    parser.add_argument("--num_workers", type=int, default=1)
    # 是否启用分布式训练
    parser.add_argument("--ddp", action="store_true")
    # 梯度累积步数
    parser.add_argument("--accumulation_steps", type=int, default=8)
    # 梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 预热步数
    parser.add_argument("--warmup_iters", type=int, default=0)
    # 日志间隔
    parser.add_argument("--log_interval", type=int, default=100)
    # 模型保存间隔
    parser.add_argument("--save_interval", type=int, default=100)
    # 分布式训练本地rank（自动获取，无需手动设置）
    parser.add_argument('--local_rank', type=int, default=-1)
    # 模型维度
    parser.add_argument('--dim', default=512, type=int)
    # 模型层数
    parser.add_argument('--n_layers', default=8, type=int)
    # 最大序列长度
    parser.add_argument('--max_seq_len', default=512, type=int)
    # 是否使用MoE结构
    parser.add_argument('--use_moe', default=False, type=bool)
    # 训练数据路径
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    # 初始化模型配置
    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe
    )
    
    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 计算每个iteration处理的token数量
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    
    # 设置随机种子
    torch.manual_seed(1337)
    
    # 确定设备类型（cuda/cpu）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 混合精度训练上下文（CPU时使用普通上下文）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 判断是否分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 初始化分布式训练
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅主进程）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和tokenizer
    model, tokenizer = init_model(lm_config)
    
    # 创建训练数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,    # 启用内存锁页，加速数据传输
        drop_last=False,    # 保留不完整批次
        shuffle=False,      # 分布式训练时通过sampler控制shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 初始化梯度缩放器（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式数据并行包装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略不需要同步的参数
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    # 训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)