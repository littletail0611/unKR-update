"""两阶段增量置信度更新实验入口脚本。

使用方法::

    python run_incremental_unkr.py \\
        --data_dir ./datasets/cn15k_ind \\
        --model_name UKGE \\
        --emb_dim 128 \\
        --base_epochs 200 \\
        --device cuda

流程：

1. **阶段 1**：在 base 数据上训练所选 unKR 模型。
2. **阶段 1.5**：用 :class:`~unKR.model.UKGModel.updater_adapter.UnKRModelAdapter`
   包装 base 模型，冻结嵌入层，仅训练 MLP 双分支。
3. **阶段 2**：使用 :class:`~unKR.updater.UnifiedConfidenceUpdater` 在 inc 数据
   上执行增量 belief 更新，最终评估并保存结果。
"""

import sys
import os
import torch
import torch.optim as optim

# 添加 src 到模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from unKR.data.IncrementalDataset import UKGDataset
from unKR.model.UKGModel.updater_adapter import UnKRModelAdapter
from unKR.updater import UnifiedConfidenceUpdater
from unKR.utils.eval_utils import evaluate_model, evaluate_belief_state, Logger
from unKR.config.incremental_config import get_incremental_args


# ---------------------------------------------------------------------------
# 辅助：动态构建 unKR 模型所需的 args-like 命名空间
# ---------------------------------------------------------------------------

class _ModelArgs:
    """为 unKR 模型构造函数提供最小参数命名空间。"""

    def __init__(self, num_ent, num_rel, emb_dim, device,
                 num_neg=1, GUMBEL_BETA=0.5):
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.gpu = device
        self.num_neg = num_neg
        self.GUMBEL_BETA = GUMBEL_BETA


def _build_unkr_model(model_name, model_args):
    """根据 ``model_name`` 动态导入并初始化对应的 unKR 模型。

    Args:
        model_name: 模型名称字符串，如 ``"UKGE"``、``"BEUrRE"``。
        model_args: :class:`_ModelArgs` 命名空间。

    Returns:
        nn.Module: 已初始化的 unKR 模型实例。

    Raises:
        ValueError: 若 ``model_name`` 不在支持列表中。
    """
    supported = ["UKGE", "UKGE_PSL", "BEUrRE", "FocusE", "GTransE",
                 "UKGsE", "UPGAT", "PASSLEAF", "SSCDL"]
    if model_name not in supported:
        raise ValueError(f"不支持的模型 '{model_name}'，可选: {supported}")

    if model_name == "UKGE":
        from unKR.model.UKGModel.UKGE import UKGE
        return UKGE(model_args)
    elif model_name == "UKGE_PSL":
        from unKR.model.UKGModel.UKGE_PSL import UKGE_PSL
        return UKGE_PSL(model_args)
    elif model_name == "BEUrRE":
        from unKR.model.UKGModel.BEUrRE import BEUrRE
        return BEUrRE(model_args)
    elif model_name == "FocusE":
        from unKR.model.UKGModel.FocusE import FocusE
        return FocusE(model_args)
    elif model_name == "GTransE":
        from unKR.model.UKGModel.GTransE import GTransE
        return GTransE(model_args)
    elif model_name == "UKGsE":
        from unKR.model.UKGModel.UKGsE import UKGsE
        return UKGsE(model_args)
    elif model_name == "UPGAT":
        from unKR.model.UKGModel.UPGAT import UPGAT
        return UPGAT(model_args)
    elif model_name == "PASSLEAF":
        from unKR.model.UKGModel.PASSLEAF import PASSLEAF
        return PASSLEAF(model_args)
    elif model_name == "SSCDL":
        from unKR.model.UKGModel.SSCDL import ssCDL
        return ssCDL(model_args)


# ---------------------------------------------------------------------------
# 阶段 1：在 base 数据上训练 unKR 模型
# ---------------------------------------------------------------------------

def train_base_model(model, dataset, args):
    """在 base 数据上训练 unKR 模型。

    训练完成后将最佳权重保存到 ``args.checkpoint_dir``。

    Args:
        model: 已初始化的 unKR 模型。
        dataset: :class:`~unKR.data.IncrementalDataset.UKGDataset` 实例。
        args: 配置参数命名空间。

    Returns:
        str: 最佳权重文件路径。
    """
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_{dataset_name}_base.pth")

    print(f"=== 阶段 1: 在 Base 数据上训练 {args.model_name} (设备: {args.device}) ===")
    print(f">>> 权重将保存至: {ckpt_path}")

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr,
                           weight_decay=args.base_weight_decay)

    base_triples = dataset.base_train
    h_all = torch.tensor([t[0] for t in base_triples], dtype=torch.long)
    r_all = torch.tensor([t[1] for t in base_triples], dtype=torch.long)
    t_all = torch.tensor([t[2] for t in base_triples], dtype=torch.long)
    c_all = torch.tensor([t[3] for t in base_triples], dtype=torch.float)

    n = len(base_triples)
    best_valid_mse = float('inf')
    patience_counter = 0

    for epoch in range(args.base_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, n, args.base_batch_size):
            idx = perm[i:i + args.base_batch_size]
            bh = h_all[idx].to(args.device)
            br = r_all[idx].to(args.device)
            bt = t_all[idx].to(args.device)
            bc = c_all[idx].to(args.device)

            optimizer.zero_grad()
            triples = torch.stack([bh, br, bt], dim=1)
            score = model(triples)
            loss = torch.mean((score - bc) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % max(1, args.base_epochs // 20) == 0:
            model.eval()
            valid_triples = dataset.base_valid
            if valid_triples:
                with torch.no_grad():
                    vh = torch.tensor([t[0] for t in valid_triples], dtype=torch.long, device=args.device)
                    vr = torch.tensor([t[1] for t in valid_triples], dtype=torch.long, device=args.device)
                    vt = torch.tensor([t[2] for t in valid_triples], dtype=torch.long, device=args.device)
                    vc = torch.tensor([t[3] for t in valid_triples], dtype=torch.float, device=args.device)
                    v_triples = torch.stack([vh, vr, vt], dim=1)
                    v_score = model(v_triples)
                    valid_mse = torch.mean((v_score - vc) ** 2).item()
            else:
                valid_mse = avg_loss

            print(f"Epoch [{epoch+1}/{args.base_epochs}] | Loss: {avg_loss:.4f} | Valid MSE: {valid_mse:.4f}")

            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print("  --> [模型提升] 最佳权重已保存")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n触发早停! 验证集已连续 {args.patience} 轮未提升。")
                    break

    print("\n=== Base 模型训练完成 ===")
    return ckpt_path


# ---------------------------------------------------------------------------
# 阶段 1.5：训练适配器的 MLP 双分支
# ---------------------------------------------------------------------------

def train_mlp_branches(adapted_model, dataset, args):
    """冻结 unKR 嵌入层，仅训练 MLP 双分支以拟合 base 数据的置信度。

    Args:
        adapted_model: :class:`~unKR.model.UKGModel.updater_adapter.UnKRModelAdapter`
            实例。
        dataset: :class:`~unKR.data.IncrementalDataset.UKGDataset` 实例。
        args: 配置参数命名空间。
    """
    print("\n=== 阶段 1.5: 训练置信度预测 MLP 双分支 ===")

    # 冻结嵌入层
    adapted_model.entity_emb.weight.requires_grad_(False)
    adapted_model.relation_emb.weight.requires_grad_(False)

    # 仅训练 MLP
    mlp_params = (list(adapted_model.mlp_mean.parameters())
                  + list(adapted_model.mlp_var.parameters()))
    for p in mlp_params:
        p.requires_grad_(True)

    optimizer = optim.Adam(mlp_params, lr=args.mlp_lr)

    base_triples = [t for t in dataset.base_train if t[3] is not None]
    h_all = torch.tensor([t[0] for t in base_triples], dtype=torch.long, device=args.device)
    r_all = torch.tensor([t[1] for t in base_triples], dtype=torch.long, device=args.device)
    t_all = torch.tensor([t[2] for t in base_triples], dtype=torch.long, device=args.device)
    c_all = torch.tensor([t[3] for t in base_triples], dtype=torch.float, device=args.device)

    for epoch in range(args.mlp_epochs):
        adapted_model.train()
        z = adapted_model.entity_emb.weight.detach()  # 冻结嵌入
        mu, sigma_sq = adapted_model.predict(z[h_all], r_all, z[t_all])
        loss = adapted_model.heteroscedastic_loss(mu, sigma_sq, c_all)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, args.mlp_epochs // 5) == 0:
            print(f"  MLP Epoch [{epoch+1}/{args.mlp_epochs}] | loss={loss.item():.6f}")


# ---------------------------------------------------------------------------
# 阶段 2：增量 Belief 更新
# ---------------------------------------------------------------------------

def run_incremental_update(adapted_model, dataset, args):
    """执行增量 belief 更新并进行全面评估。

    Args:
        adapted_model: :class:`~unKR.model.UKGModel.updater_adapter.UnKRModelAdapter`
            实例（已完成阶段 1 和 1.5 训练）。
        dataset: :class:`~unKR.data.IncrementalDataset.UKGDataset` 实例。
        args: 配置参数命名空间。

    Returns:
        dict: 包含 ``inc_test``、``base_test``、``combined_test`` 三组指标的字典。
    """
    mode_str = "Single-batch (one-shot)" if args.single_batch else "Streaming (mini-batch)"
    print(f"\n=== 阶段 2: 增量 Belief 更新 [模式: {mode_str}] ===")

    updater = UnifiedConfidenceUpdater(
        model=adapted_model,
        dataset=dataset,
        lr=args.inc_lr,
        gamma=args.gamma,
        device=args.device,
        args=args,
        ablation_mode=args.ablation_mode,
    )

    # 更新前初始评估
    edge_index, edge_type, edge_conf = dataset.get_base_graph_data()
    edge_index = edge_index.to(args.device)
    edge_type = edge_type.to(args.device)
    edge_conf = edge_conf.to(args.device)

    adapted_model.eval()
    with torch.no_grad():
        init_z = adapted_model(edge_index, edge_type, edge_conf)
    init_metrics = evaluate_model(adapted_model, dataset.inc_valid, z=init_z, device=args.device)
    Logger.print_metrics("Pre-update Inc Valid", init_metrics)

    # 执行增量更新
    if args.single_batch:
        all_facts = dataset.inc_train
        print(f"\n>>> [Single-batch] 一次性处理 {len(all_facts)} 条增量事实...")
        new_mu, change_mean, change_max, affected = updater.step(all_facts)
        print(f"  - 新事实平均置信度: {new_mu.mean().item():.4f}")
        print(f"  - 全局平均变化: {change_mean:.6f} | 最大变化: {change_max:.4f}")
        print(f"  - 受影响旧事实数: {affected}")
    else:
        inc_batches = list(dataset.get_incremental_batches(batch_size=args.inc_batch_size))
        for batch_idx, batch_facts in enumerate(inc_batches):
            print(f"\n--- Batch {batch_idx + 1}/{len(inc_batches)} ---")
            new_mu, change_mean, change_max, affected = updater.step(batch_facts)
            print(f"  - 新事实平均置信度: {new_mu.mean().item():.4f}")
            print(f"  - 全局平均变化: {change_mean:.6f} | 最大变化: {change_max:.4f}")
            print(f"  - 受影响旧事实数: {affected}")

    print("\n=== 增量更新完毕，执行最终全面评估 ===")

    # 构建包含 inc 数据的联合图（供 forward 使用）
    inc_facts = dataset.inc_train
    if inc_facts:
        inc_h = torch.tensor([f[0] for f in inc_facts], dtype=torch.long, device=args.device)
        inc_r = torch.tensor([f[1] for f in inc_facts], dtype=torch.long, device=args.device)
        inc_t = torch.tensor([f[2] for f in inc_facts], dtype=torch.long, device=args.device)
        inc_c = torch.tensor(
            [dataset.belief_state.get((f[0], f[1], f[2]), 0.5) for f in inc_facts],
            dtype=torch.float, device=args.device
        )
        eval_ei = torch.cat([edge_index, torch.stack([inc_h, inc_t])], dim=1)
        eval_et = torch.cat([edge_type, inc_r])
        eval_ec = torch.cat([edge_conf, inc_c])
    else:
        eval_ei, eval_et, eval_ec = edge_index, edge_type, edge_conf

    adapted_model.eval()
    with torch.no_grad():
        final_z = adapted_model(eval_ei, eval_et, eval_ec)

    inc_test_metrics = evaluate_model(adapted_model, dataset.inc_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update on Inc Test", inc_test_metrics)

    base_test_metrics = evaluate_model(adapted_model, dataset.base_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update on Base Test", base_test_metrics)

    combined_test = dataset.base_test + dataset.inc_test
    combined_metrics = evaluate_model(adapted_model, combined_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update on Combined Test (Base + Inc)", combined_metrics)

    # 保存最终 belief state
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    belief_path = os.path.join(
        args.checkpoint_dir, f"{args.model_name}_{dataset_name}_belief_state.pt"
    )
    torch.save(dataset.belief_state, belief_path)
    print(f">>> 全局 Belief 状态已保存至 {belief_path}")

    return {
        "inc_test": inc_test_metrics,
        "base_test": base_test_metrics,
        "combined_test": combined_metrics,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    args = get_incremental_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"配置: model={args.model_name}, data_dir={args.data_dir}, "
          f"emb_dim={args.emb_dim}, device={args.device}")

    # 1. 加载数据集
    dataset = UKGDataset(data_dir=args.data_dir)

    model_args = _ModelArgs(
        num_ent=dataset.num_ent,
        num_rel=dataset.num_rel,
        emb_dim=args.emb_dim,
        device=args.device,
    )

    # 2. 构建 unKR 模型
    unkr_model = _build_unkr_model(args.model_name, model_args)
    unkr_model = unkr_model.to(args.device)

    # 阶段 1：base 训练
    ckpt_path = train_base_model(unkr_model, dataset, args)

    # 加载最佳权重
    unkr_model.load_state_dict(torch.load(ckpt_path, map_location=args.device), strict=False)
    unkr_model.eval()

    # 构建适配器
    adapted_model = UnKRModelAdapter(unkr_model, emb_dim=args.emb_dim,
                                     dropout_rate=args.dropout_rate).to(args.device)

    # 如果 inc 阶段有新实体，动态扩展嵌入层
    if dataset.num_ent > dataset.base_num_ent:
        adapted_model.expand_embedding(dataset.num_ent)

    # 阶段 1.5：训练 MLP 双分支
    train_mlp_branches(adapted_model, dataset, args)

    # 阶段 2：增量更新与评估
    results = run_incremental_update(adapted_model, dataset, args)

    print("\n=== 实验完成 ===")
    for stage, metrics in results.items():
        print(f"{stage}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, "
              f"RMSE={metrics['RMSE']:.4f}")


if __name__ == "__main__":
    main()
