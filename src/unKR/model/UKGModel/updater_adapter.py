"""适配器：将 unKR 模型包装成 update 库 UnifiedConfidenceUpdater 所需的统一接口。"""

import torch
import torch.nn as nn


class UnKRModelAdapter(nn.Module):
    """将 unKR 模型（UKGE、BEUrRE、FocusE、GTransE、UPGAT 等）适配为
    ``UnifiedConfidenceUpdater`` 所需的统一接口。

    核心思路：

    1. 复用 unKR 模型的 ``ent_emb`` / ``rel_emb`` 作为 ``entity_emb`` /
       ``relation_emb``。对于 BEUrRE（box embedding），从 ``min_embedding``
       提取点嵌入近似。
    2. 新增 ``mlp_mean`` / ``mlp_var`` 双分支 MLP，用于预测置信度均值和方差
       （与 update 库的 ``HeteroscedasticBaseModel`` 接口一致）。
    3. ``forward()`` 直接返回实体嵌入权重矩阵（unKR 模型非 GNN 模型，无消息传递）。
    4. ``predict()`` 通过拼接 ``(z_h, r_emb, z_t)`` 输入 MLP 来预测
       ``(mu, sigma_sq)``。

    Args:
        unkr_model: 已初始化的 unKR 模型实例。
        emb_dim: 嵌入维度。
        dropout_rate: Dropout 比例，默认 0.3。

    Attributes:
        unkr_model: 底层 unKR 模型。
        emb_dim: 嵌入维度。
        entity_emb: 实体嵌入层（映射自 unKR 模型）。
        relation_emb: 关系嵌入层（映射自 unKR 模型）。
        mlp_mean: 均值预测分支 MLP。
        mlp_var: 方差预测分支 MLP。
    """

    def __init__(self, unkr_model, emb_dim, dropout_rate=0.3):
        super().__init__()
        self.unkr_model = unkr_model
        self.emb_dim = emb_dim

        # ---- 实体嵌入映射 ----
        if hasattr(unkr_model, 'ent_emb') and isinstance(unkr_model.ent_emb, nn.Embedding):
            self.entity_emb = unkr_model.ent_emb
        elif hasattr(unkr_model, 'ent_emb') and isinstance(unkr_model.ent_emb, nn.Parameter):
            # UPGAT 使用 nn.Parameter 而非 nn.Embedding
            num_ent = unkr_model.ent_emb.shape[0]
            self.entity_emb = nn.Embedding(num_ent, emb_dim)
            self.entity_emb.weight = unkr_model.ent_emb  # 共享参数
        elif hasattr(unkr_model, 'min_embedding'):
            # BEUrRE：从 box 最小边界提取点嵌入近似
            num_ent = unkr_model.min_embedding.shape[0]
            actual_dim = min(emb_dim, unkr_model.min_embedding.shape[1])
            self.entity_emb = nn.Embedding(num_ent, actual_dim)
            with torch.no_grad():
                self.entity_emb.weight.copy_(unkr_model.min_embedding[:, :actual_dim])
        else:
            raise ValueError(
                f"无法从模型 {type(unkr_model).__name__} 中提取实体嵌入。"
                "请确保模型含有 ent_emb 或 min_embedding 属性。"
            )

        # ---- 关系嵌入映射 ----
        if hasattr(unkr_model, 'rel_emb') and isinstance(unkr_model.rel_emb, nn.Embedding):
            self.relation_emb = unkr_model.rel_emb
        elif hasattr(unkr_model, 'rel_emb') and isinstance(unkr_model.rel_emb, nn.Parameter):
            num_rel = unkr_model.rel_emb.shape[0]
            self.relation_emb = nn.Embedding(num_rel, emb_dim)
            self.relation_emb.weight = unkr_model.rel_emb
        elif hasattr(unkr_model, 'rel_trans_for_head'):
            # BEUrRE：用 rel_trans_for_head 作为关系嵌入近似
            num_rel = unkr_model.rel_trans_for_head.shape[0]
            actual_dim = min(emb_dim, unkr_model.rel_trans_for_head.shape[1])
            self.relation_emb = nn.Embedding(num_rel, actual_dim)
            with torch.no_grad():
                self.relation_emb.weight.copy_(unkr_model.rel_trans_for_head[:, :actual_dim])
        else:
            raise ValueError(
                f"无法从模型 {type(unkr_model).__name__} 中提取关系嵌入。"
                "请确保模型含有 rel_emb 或 rel_trans_for_head 属性。"
            )

        actual_emb_dim = self.entity_emb.weight.shape[1]

        # ---- 双分支置信度预测 MLP（与 HeteroscedasticBaseModel 接口一致）----
        self.mlp_mean = nn.Sequential(
            nn.Linear(3 * actual_emb_dim, actual_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(actual_emb_dim, 1),
            nn.Sigmoid(),
        )

        self.mlp_var = nn.Sequential(
            nn.Linear(3 * actual_emb_dim, actual_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(actual_emb_dim, 1),
            nn.Softplus(),
        )

    def forward(self, edge_index, edge_type, edge_conf):
        """兼容 updater 的 forward 接口。

        对于非 GNN 模型，直接返回实体嵌入权重矩阵，不做消息传递。

        Args:
            edge_index: 图的边索引，shape ``[2, E]``（本实现中未使用）。
            edge_type: 边关系类型，shape ``[E]``（本实现中未使用）。
            edge_conf: 边置信度，shape ``[E]``（本实现中未使用）。

        Returns:
            torch.Tensor: 实体嵌入矩阵，shape ``[num_ent, emb_dim]``。
        """
        return self.entity_emb.weight

    def predict(self, z_h, r_id, z_t):
        """兼容 updater 的 predict 接口。

        将头实体嵌入、关系嵌入和尾实体嵌入拼接后输入 MLP 预测置信度均值和方差。

        Args:
            z_h: 头实体嵌入，shape ``[batch, emb_dim]``。
            r_id: 关系 ID 张量，shape ``[batch]``。
            z_t: 尾实体嵌入，shape ``[batch, emb_dim]``。

        Returns:
            tuple: ``(mu, sigma_sq)``，均为 shape ``[batch]`` 的张量。
        """
        r_emb = self.relation_emb(r_id)
        h_r_t = torch.cat([z_h, r_emb, z_t], dim=-1)
        mu = self.mlp_mean(h_r_t).squeeze(-1)
        sigma_sq = self.mlp_var(h_r_t).squeeze(-1)
        return mu, sigma_sq

    def heteroscedastic_loss(self, mu, sigma_sq, target_conf):
        """异方差高斯负对数似然损失（与 update 库一致）。

        .. math::

            \\mathcal{L} = \\frac{(y - \\mu)^2}{2(\\sigma^2 + \\epsilon)}
                           + \\frac{1}{2}\\ln(\\sigma^2 + \\epsilon)

        Args:
            mu: 预测均值，shape ``[batch]``。
            sigma_sq: 预测方差，shape ``[batch]``。
            target_conf: 目标置信度，shape ``[batch]``。

        Returns:
            torch.Tensor: 标量损失值。
        """
        eps = 1e-6
        mse_term = ((target_conf - mu) ** 2) / (2 * (sigma_sq + eps))
        reg_term = 0.5 * torch.log(sigma_sq + eps)
        return torch.mean(mse_term + reg_term)

    def expand_embedding(self, new_num_ent):
        """动态扩展实体嵌入以容纳新实体。

        新实体的嵌入使用正态分布随机初始化（均值 0，标准差 0.1）。

        Args:
            new_num_ent: 扩展后的实体总数（须大于当前实体数）。
        """
        old_num_ent, emb_dim = self.entity_emb.weight.shape
        if new_num_ent <= old_num_ent:
            return

        new_weight = torch.zeros(new_num_ent, emb_dim, device=self.entity_emb.weight.device)
        new_weight[:old_num_ent] = self.entity_emb.weight.data
        nn.init.normal_(new_weight[old_num_ent:], mean=0.0, std=0.1)

        new_emb = nn.Embedding(new_num_ent, emb_dim)
        new_emb.weight = nn.Parameter(new_weight)
        self.entity_emb = new_emb

    def get_unkr_score(self, h_idx, r_idx, t_idx):
        """调用原始 unKR 模型的评分函数。

        Args:
            h_idx: 头实体索引，shape ``[batch]``。
            r_idx: 关系索引，shape ``[batch]``。
            t_idx: 尾实体索引，shape ``[batch]``。

        Returns:
            torch.Tensor: 模型原生置信度分数。
        """
        triples = torch.stack([h_idx, r_idx, t_idx], dim=-1)
        return self.unkr_model(triples)
