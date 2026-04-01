#!/usr/bin/env python
"""增量置信度更新运行入口。

该脚本演示如何将 unKR 的嵌入模型与 UnifiedConfidenceUpdater 集成，
以对不确定知识图谱执行增量置信度更新。

典型用法::

    python scripts/run_incremental_update.py \\
        --data_dir  data/CN15K \\
        --model_name UKGE \\
        --unkr_checkpoint checkpoints/ukge_cn15k.ckpt \\
        --emb_dim 128 \\
        --device cpu

流程说明：
    1. 加载 IncrementalUKGDataset（base/inc 两阶段数据格式）。
    2. 初始化指定的 unKR 模型并加载预训练权重（可选）。
    3. 用 UnKRModelAdapter 包装模型，统一暴露 entity_emb / relation_emb /
       mlp_mean / mlp_var / forward / predict 接口。
    4. 初始化 UnifiedConfidenceUpdater。
    5. 以 single-batch 或 streaming 模式调用 updater.step() 执行增量更新。
    6. 打印更新摘要（新事实平均置信度、全局平均变动、局部最大变动、受影响旧事实数）。
"""

import argparse
import os
import sys

import torch

# 确保 src 目录在 PYTHONPATH 中（适配直接运行脚本的场景）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def get_args():
    parser = argparse.ArgumentParser(
        description="unKR + UnifiedConfidenceUpdater 增量更新脚本"
    )

    # ------------------------------------------------------------------
    # 数据与设备
    # ------------------------------------------------------------------
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="数据集根目录，需包含 base/ 和 inc/ 子目录。")
    parser.add_argument("--device",    type=str, default="cpu",
                        help='运算设备，如 "cpu" 或 "cuda:0"。')
    parser.add_argument("--seed",      type=int, default=42)

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    parser.add_argument("--model_name", type=str, default="UKGE",
                        choices=["UKGE", "UKGE_PSL", "BEUrRE", "FocusE",
                                 "GTransE", "PASSLEAF", "UKGsE", "UPGAT", "SSCDL"],
                        help="unKR 模型名称。")
    parser.add_argument("--unkr_checkpoint", type=str, default=None,
                        help="预训练权重路径（.ckpt 或 .pth），可选。")
    parser.add_argument("--emb_dim",    type=int, default=128,
                        help="嵌入向量维度。")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--num_pseudo",   type=int,   default=10,
                        help="UPGAT 专用：伪邻居预测时每对 (head, relation) 保留的候选尾实体数。")

    # ------------------------------------------------------------------
    # 增量更新超参数（与 update 库 config.py 对齐）
    # ------------------------------------------------------------------
    parser.add_argument("--inc_lr",     type=float, default=0.001,
                        help="增量更新的学习率。")
    parser.add_argument("--gamma",      type=float, default=0.8,
                        help="因果影响评估的衰减系数。")
    parser.add_argument("--inc_batch_size", type=int, default=1024,
                        help="streaming 模式下每批增量事实的大小。")
    parser.add_argument("--single_batch", action="store_true",
                        help="使用 single-batch（one-shot）模式处理所有增量事实。")
    parser.add_argument("--ablation_mode", type=str, default="full",
                        help=(
                            '消融实验模式，逗号分隔，例如 "wo_topo_prop,wo_anchor"。'
                            '设为 "full" 时启用所有模块（默认）。'
                        ))

    # Stage 1: 课程微调
    parser.add_argument("--anchor_steps",           type=int,   default=3)
    parser.add_argument("--finetune_steps",          type=int,   default=5)
    parser.add_argument("--mlp_anchor_coeff",        type=float, default=0.01)
    parser.add_argument("--lambda_ent_reg",          type=float, default=0.1)
    parser.add_argument("--alpha_labeled_supervision", type=float, default=1.0)
    parser.add_argument("--alpha_new_supervision",   type=float, default=0.3)
    parser.add_argument("--dynamic_update_interval", type=int,   default=2)

    # Stage 2: 因果影响
    parser.add_argument("--causal_num_hops",   type=int,   default=2)

    # Stage 3: 贝叶斯滤波
    parser.add_argument("--epsilon",           type=float, default=1e-4)

    # Stage 4: 局部表征精炼
    parser.add_argument("--influence_threshold", type=float, default=0.01)
    parser.add_argument("--lambda_reg",          type=float, default=0.001)
    parser.add_argument("--func_anchor_ratio",   type=float, default=0.9)
    parser.add_argument("--refine_steps",        type=int,   default=3)

    return parser.parse_args()


def build_model(model_name: str, num_ent: int, num_rel: int, args):
    """根据模型名称实例化对应的 unKR 模型。"""
    import importlib

    module = importlib.import_module(f"unKR.model.UKGModel.{model_name}")
    model_cls = getattr(module, model_name)

    # 构造最小化 args 命名空间，确保模型能正常初始化
    model_args = argparse.Namespace(
        num_ent=num_ent,
        num_rel=num_rel,
        emb_dim=args.emb_dim,
        gpu=args.device,
        # UPGAT 专用参数：伪邻居预测时每对 (h, r) 保留的候选尾实体数
        num_pseudo=getattr(args, "num_pseudo", 10),
    )
    return model_cls(model_args)


def load_checkpoint(model, ckpt_path: str, device: str):
    """加载预训练权重（支持 PyTorch Lightning .ckpt 和普通 .pth 格式）。"""
    print(f">>> 加载预训练权重: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)

    # PyTorch Lightning checkpoint 的权重存储在 "state_dict" 键下，
    # 且键名带有 "model." 前缀，需要去除。
    if "state_dict" in state_dict:
        raw = state_dict["state_dict"]
        cleaned = {}
        for k, v in raw.items():
            new_k = k[len("model."):] if k.startswith("model.") else k
            cleaned[new_k] = v
        state_dict = cleaned

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  警告: 以下键缺失（将使用随机初始化）: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  警告: 以下键未使用: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


def run(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # 1. 加载数据集
    # ------------------------------------------------------------------
    from unKR.data.IncrementalDataset import IncrementalUKGDataset

    print(f"\n{'='*60}")
    print(f"数据集路径: {args.data_dir}")
    print(f"模型名称:   {args.model_name}")
    print(f"运算设备:   {args.device}")
    print(f"{'='*60}\n")

    dataset = IncrementalUKGDataset(args.data_dir)
    print(f"\n数据集摘要: 实体总数={dataset.num_ent}, 关系总数={dataset.num_rel}, "
          f"Base 实体数={dataset.base_num_ent}")

    # ------------------------------------------------------------------
    # 2. 初始化 unKR 模型
    # ------------------------------------------------------------------
    unkr_model = build_model(
        args.model_name,
        num_ent=dataset.num_ent,
        num_rel=dataset.num_rel,
        args=args,
    )
    unkr_model = unkr_model.to(args.device)

    if args.unkr_checkpoint and os.path.exists(args.unkr_checkpoint):
        unkr_model = load_checkpoint(unkr_model, args.unkr_checkpoint, args.device)
    else:
        if args.unkr_checkpoint:
            print(f"警告: 未找到预训练权重 {args.unkr_checkpoint}，将使用随机初始化。")
        else:
            print("未指定预训练权重，将使用随机初始化。")

    # ------------------------------------------------------------------
    # 3. 用 UnKRModelAdapter 包装模型
    # ------------------------------------------------------------------
    from unKR.model.UKGModel.updater_adapter import UnKRModelAdapter

    adapter = UnKRModelAdapter(
        unkr_model,
        emb_dim=args.emb_dim,
        num_ent=dataset.num_ent,
        num_rel=dataset.num_rel,
        dropout_rate=args.dropout_rate,
    ).to(args.device)
    print(f"\n>>> 适配器初始化成功（底层模型: {args.model_name}）。")

    # ------------------------------------------------------------------
    # 4. 初始化 UnifiedConfidenceUpdater
    # ------------------------------------------------------------------
    from unKR.updater import UnifiedConfidenceUpdater

    updater = UnifiedConfidenceUpdater(
        model=adapter,
        dataset=dataset,
        lr=args.inc_lr,
        gamma=args.gamma,
        device=args.device,
        args=args,
        ablation_mode=args.ablation_mode,
    )
    print("\n>>> UnifiedConfidenceUpdater 初始化完成。")

    # ------------------------------------------------------------------
    # 5. 执行增量更新
    # ------------------------------------------------------------------
    mode_str = "Single-batch (one-shot)" if args.single_batch else "Streaming (mini-batch)"
    print(f"\n>>> 启动增量更新 [模式: {mode_str}]...")

    if args.single_batch:
        all_facts = dataset.inc_train
        print(f"    处理全部 {len(all_facts)} 条增量事实（单批次）...")
        new_mu, mean_chg, max_chg, affected = updater.step(all_facts)
        print("\n=== 增量更新摘要 ===")
        print(f"  新事实平均置信度 : {new_mu.mean().item():.4f}")
        print(f"  全局平均变动幅度 : {mean_chg:.6f}")
        print(f"  局部最大变动幅度 : {max_chg:.4f}")
        print(f"  受影响旧事实数   : {affected}")
    else:
        batches = list(dataset.get_incremental_batches(batch_size=args.inc_batch_size))
        print(f"    共 {len(batches)} 个批次，每批最多 {args.inc_batch_size} 条事实...")
        for idx, batch in enumerate(batches):
            new_mu, mean_chg, max_chg, affected = updater.step(batch)
            print(f"\n  Batch {idx + 1}/{len(batches)} 摘要:")
            print(f"    新事实平均置信度 : {new_mu.mean().item():.4f}")
            print(f"    全局平均变动幅度 : {mean_chg:.6f}")
            print(f"    局部最大变动幅度 : {max_chg:.4f}")
            print(f"    受影响旧事实数   : {affected}")

    print("\n=== 增量更新全部完成 ===")

    # 保存最终 belief state
    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    save_path = os.path.join(out_dir, f"final_belief_state_{dataset_name}.pt")
    torch.save(dataset.belief_state, save_path)
    print(f">>> 全局 Belief 状态已保存至 {save_path}")


if __name__ == "__main__":
    args = get_args()
    run(args)
