# 导入必要的库
import os
import platform
import argparse  # 命令行参数解析
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist  # 分布式训练
from contextlib import nullcontext  # 上下文管理器

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 分布式数据采样
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace工具
from model.model import MiniMindLM  # 自定义模型
from model.LMConfig import LMConfig  # 模型配置
from model.dataset import SFTDataset  # 微调数据集

warnings.filterwarnings('ignore')  # 忽略警告


def Logger(content):
    """分布式训练日志记录器（只在主进程打印）"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """余弦退火学习率调度器"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """单个训练周期的完整流程"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 带掩码的交叉熵损失
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据转移到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 学习率调度
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度训练上下文
        with ctx:
            res = model(X)  # 前向传播
            # 计算掩码损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用损失掩码
            loss += res.aux_loss  # 添加辅助损失（如MoE模型的专家负载平衡损失）
            loss = loss / args.accumulation_steps  # 梯度累积

        # 反向传播与梯度缩放
        scaler.scale(loss).backward()

        # 梯度累积步骤
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录到wandb（主进程）
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 模型保存（主进程）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            # 处理分布式模型状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载自定义分词器
    model = MiniMindLM(lm_config)  # 初始化模型
    
    # 加载预训练权重
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)  # 允许部分加载
    
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 使用NCCL后端
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)  # 设置当前GPU设备


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型（bfloat16/float16）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--ddp", action="store_true", help="启用分布式训练")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="预热步数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    parser.add_argument('--local_rank', type=int, default=-1, help="分布式训练本地rank")
    parser.add_argument('--dim', default=512, type=int, help="模型维度")
    parser.add_argument('--n_layers', default=8, type=int, help="Transformer层数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=False, type=bool, help="是否使用MoE结构")
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl", help="训练数据路径")

    args = parser.parse_args()

    # 模型配置
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
    
    # 计算每次迭代处理的token数
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)  # 固定随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # wandb运行名称
    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 混合精度上下文（CPU时使用普通上下文）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 分布式训练检测
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（主进程）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)

    # 数据集与数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 锁页内存加速传输
        drop_last=False,
        shuffle=False,    # 分布式训练时用sampler控制shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 分布式数据并行包装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略旋转位置编码缓存
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练循环
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)