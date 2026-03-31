"""增量更新实验参数配置。

基于 update 库的 ``config.py``，新增 ``--model_name`` 参数用于选择 unKR 模型。
"""

import argparse
import torch


def get_incremental_args(argv=None):
    """解析增量更新实验的命令行参数。

    Args:
        argv: 参数列表，默认为 ``None``（从 ``sys.argv`` 读取）。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="unKR 增量置信度更新实验")

    # ================= 1. 数据与环境配置 =================
    parser.add_argument("--data_dir", type=str, default="./datasets/nl27k_ind",
                        help="数据集根目录（需包含 base/ 和 inc/ 子目录）")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (cuda / cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证实验可复现")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="模型权重保存目录")

    # ================= 2. 模型选择 =================
    parser.add_argument("--model_name", type=str, default="UKGE",
                        choices=["UKGE", "UKGE_PSL", "BEUrRE", "FocusE", "GTransE",
                                 "UKGsE", "UPGAT", "PASSLEAF", "SSCDL"],
                        help="选择 unKR 底层模型")

    # ================= 3. 模型结构参数 =================
    parser.add_argument("--emb_dim", type=int, default=128,
                        help="实体和关系的嵌入维度 (d)")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="防过拟合的 Dropout 比例")

    # ================= 4. Base 模型离线训练参数 =================
    parser.add_argument("--base_epochs", type=int, default=200,
                        help="Base 模型最大训练轮数")
    parser.add_argument("--base_lr", type=float, default=0.001,
                        help="Base 模型学习率")
    parser.add_argument("--base_weight_decay", type=float, default=1e-4,
                        help="L2 正则化权重")
    parser.add_argument("--base_batch_size", type=int, default=2048,
                        help="Base 模型 Batch 大小")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停机制容忍次数")
    parser.add_argument("--mlp_epochs", type=int, default=50,
                        help="适配器 MLP 双分支训练轮数（阶段 1.5）")
    parser.add_argument("--mlp_lr", type=float, default=0.001,
                        help="适配器 MLP 双分支学习率（阶段 1.5）")

    # ================= 5. 增量更新参数 =================
    # [通用控制]
    parser.add_argument("--inc_batch_size", type=int, default=1024,
                        help="处理新事实的 Batch 大小")
    parser.add_argument("--single_batch", action="store_true", default=False,
                        help="将所有增量事实作为一个 Batch 一次性更新（默认流式处理）")
    parser.add_argument("--inc_lr", type=float, default=0.005,
                        help="增量更新器 (Updater) 的学习率")
    parser.add_argument("--mlp_anchor_coeff", type=float, default=0.01,
                        help="全局预测头 (MLP) 的正则化系数，防止全局预测能力崩溃")

    # [Stage 1: 先立锚再扩散]
    parser.add_argument("--anchor_steps", type=int, default=100,
                        help="Sub-stage 1.1: 纯有标签数据的立锚微调步数")
    parser.add_argument("--finetune_steps", type=int, default=50,
                        help="Sub-stage 1.2: 引入无标签数据的扩散微调步数")
    parser.add_argument("--lambda_ent_reg", type=float, default=1.0,
                        help="立锚期防止新实体表征坍塌的空间正则化权重")
    parser.add_argument("--alpha_labeled_supervision", type=float, default=1.0,
                        help="有标签数据的监督权重")
    parser.add_argument("--alpha_new_supervision", type=float, default=3.0,
                        help="无标签伪标签数据的监督权重")
    parser.add_argument("--dynamic_update_interval", type=int, default=5,
                        help="扩散期更新伪标签的频率（步数）")

    # [Stage 2: 局部因果推断与贝叶斯精炼]
    parser.add_argument("--causal_num_hops", type=int, default=2,
                        help="因果影响评估截取的局部子图跳数 (K-hop)")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="因果影响计算中的全局衰减因子")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="贝叶斯滤波防除零的平滑项")
    parser.add_argument("--influence_threshold", type=float, default=0.005,
                        help="判定旧事实受因果影响的阈值")
    parser.add_argument("--refine_steps", type=int, default=3,
                        help="受影响局部子图的联合微调步数")
    parser.add_argument("--lambda_reg", type=float, default=0.001,
                        help="受影响旧实体的 L2 正则化权重")
    parser.add_argument("--func_anchor_ratio", type=float, default=0.9,
                        help="功能性锚定损失的混合比例")

    # ================= 6. 消融实验 =================
    parser.add_argument("--ablation_mode", type=str, default="full",
                        help="消融实验模式（'full' 或逗号分隔的消融项，"
                             "如 'wo_geom_init,wo_topo_prop'）")

    # ================= 7. 调参模式 =================
    parser.add_argument("--tuning_mode", action="store_true", default=False,
                        help="启用调参模式（抑制控制台冗余输出）")

    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = get_incremental_args()
    print("当前增量更新配置参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
