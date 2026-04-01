#!/usr/bin/env python
"""增量置信度更新运行入口。

该脚本演示如何将 unKR 的嵌入模型与 UnifiedConfidenceUpdater 集成，
以对不确定知识图谱执行增量置信度更新。

典型用法::

    python scripts/run_incremental_update.py \\
        --data_dir  data/CN15K \\
        --model_name UKGE \\
        --yaml_config config/cn15k/UKGE_cn15k.yaml \\
        --unkr_checkpoint checkpoints/ukge_cn15k.ckpt \\
        --emb_dim 128 \\
        --device cpu

流程说明：
    1. 加载 IncrementalUKGDataset（base/inc 两阶段数据格式）。
    2. 从 YAML 配置文件读取模型参数（--yaml_config 指定），初始化指定的 unKR 模型。
    3. 若提供 --unkr_checkpoint，加载预训练权重（自动处理 Lightning 前缀和
       嵌入尺寸不匹配：base checkpoint 的嵌入行数少于当前实体总数时，
       多余的行用均值初始化）。
    4. 用 UnKRModelAdapter 包装模型，统一暴露 entity_emb / relation_emb /
       mlp_mean / mlp_var / forward / predict 接口。
    5. 初始化 UnifiedConfidenceUpdater。
    6. 以 single-batch 或 streaming 模式调用 updater.step() 执行增量更新。
    7. 打印更新摘要（新事实平均置信度、全局平均变动、局部最大变动、受影响旧事实数）。
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
    parser.add_argument("--yaml_config", type=str, default=None,
                        help=(
                            "unKR YAML 配置文件路径（例如 config/cn15k/UKGE_cn15k.yaml）。"
                            "文件中的参数将作为模型初始化的默认值，"
                            "命令行参数（num_ent、num_rel、emb_dim、gpu）会覆盖 YAML 中的同名参数。"
                        ))
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
    """根据模型名称和配置实例化对应的 unKR 模型。

    参数加载优先级：
        命令行关键参数（num_ent、num_rel、emb_dim、gpu）> YAML 配置文件。

    YAML 配置文件提供了 GTransE（margin、alpha）、FocusE（margin、base_model）、
    BEUrRE（GUMBEL_BETA、num_neg）等模型初始化所需的特有参数。
    若不提供 --yaml_config，则这些模型在实例化时可能因缺少参数而报 AttributeError。
    """
    import importlib
    try:
        import yaml
        _has_yaml = True
    except ImportError:
        _has_yaml = False

    # 从 YAML 配置文件加载模型参数（作为基础参数集）
    model_params: dict = {}
    yaml_path = getattr(args, "yaml_config", None)
    if yaml_path:
        if not os.path.exists(yaml_path):
            print(
                f"警告: 未找到 YAML 配置文件 {yaml_path}。"
                f"GTransE/FocusE/BEUrRE 等模型需要 YAML 提供 margin、alpha、"
                f"GUMBEL_BETA 等参数，缺少时将引发 AttributeError。"
            )
        elif not _has_yaml:
            print("警告: 未安装 PyYAML，无法加载 YAML 配置（pip install pyyaml）。")
        else:
            print(f">>> 加载 YAML 配置: {yaml_path}")
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}
            model_params.update(yaml_cfg)

    # 命令行关键参数覆盖 YAML（确保数据集实际的实体/关系数优先）
    model_params.update({
        "num_ent":    num_ent,
        "num_rel":    num_rel,
        "emb_dim":    args.emb_dim,
        "gpu":        args.device,
        "num_pseudo": getattr(args, "num_pseudo", 10),
    })

    model_args = argparse.Namespace(**model_params)

    module = importlib.import_module(f"unKR.model.UKGModel.{model_name}")
    model_cls = getattr(module, model_name)
    return model_cls(model_args)


def load_checkpoint(model, ckpt_path: str, device: str):
    """加载预训练权重（支持 PyTorch Lightning .ckpt 和普通 .pth 格式）。

    自动处理：
    1. Lightning checkpoint 格式：权重存储在 ``state_dict`` 键下，且键名可能携带
       ``"model."`` 或 ``"lit_model.model."`` 前缀，本函数统一去除。
    2. 嵌入尺寸不匹配：预训练模型仅含 base_num_ent 个实体的嵌入，而当前模型（已含
       增量新实体）的嵌入表行数更多。对于尺寸不匹配的矩阵参数，将 checkpoint 的基础
       行复制到模型对应位置，新增行则用基础行均值初始化。
    """
    print(f">>> 加载预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 处理 PyTorch Lightning checkpoint 格式
    state_dict = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        state_dict = {}
        for k, v in raw.items():
            if k.startswith("lit_model.model."):
                state_dict[k[len("lit_model.model."):]] = v
            elif k.startswith("model."):
                state_dict[k[len("model."):]] = v
            else:
                state_dict[k] = v

    # 先用 strict=False 加载，处理多余/缺失键
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 处理矩阵参数尺寸不匹配：将 checkpoint 基础行复制到模型中，新行用均值初始化
    all_model_tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    for key, ckpt_tensor in state_dict.items():
        if key not in all_model_tensors:
            continue
        model_tensor = all_model_tensors[key]
        if model_tensor.shape == ckpt_tensor.shape:
            continue  # 尺寸匹配，已在 load_state_dict 中正确加载
        if model_tensor.dim() < 2 or model_tensor.shape[0] <= ckpt_tensor.shape[0]:
            continue  # 非矩阵参数，或模型比 checkpoint 小（不支持缩减）

        # 模型的行数多于 checkpoint（预训练仅含 base 实体）
        n_base = ckpt_tensor.shape[0]
        n_model = model_tensor.shape[0]
        ckpt_data = ckpt_tensor.to(device)
        mean_row = ckpt_data.mean(dim=0)
        n_new = n_model - n_base

        with torch.no_grad():
            model_tensor.data[:n_base] = ckpt_data
            model_tensor.data[n_base:] = mean_row.unsqueeze(0).expand(n_new, -1)
        print(
            f"  嵌入扩展: {key} {list(ckpt_tensor.shape)} → {list(model_tensor.shape)}"
            f"（新增 {n_new} 行使用均值初始化）"
        )

    if missing:
        print(f"  警告: 缺失键（参数保留为模型初始化值）: {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  警告: 未使用键: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
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
    # 2. 初始化 unKR 模型（含完整 num_ent，覆盖增量新实体）
    # ------------------------------------------------------------------
    unkr_model = build_model(
        args.model_name,
        num_ent=dataset.num_ent,
        num_rel=dataset.num_rel,
        args=args,
    )
    unkr_model = unkr_model.to(args.device)

    if args.unkr_checkpoint and os.path.exists(args.unkr_checkpoint):
        # load_checkpoint 内部会自动处理：
        #   - Lightning 前缀（model. / lit_model.model.）
        #   - 嵌入尺寸不匹配（base checkpoint 行数 < 当前模型行数）
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

    # 若适配器的嵌入表仍小于 dataset.num_ent（例如 build_model 中未完全扩展），
    # 显式调用 expand_embedding 补全。
    if adapter.entity_emb.weight.shape[0] < dataset.num_ent:
        adapter.expand_embedding(dataset.num_ent, dataset.num_rel)

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
