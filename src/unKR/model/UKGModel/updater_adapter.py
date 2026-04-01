"""模型适配器模块：将 unKR 的嵌入式模型包装成 UnifiedConfidenceUpdater 所需的统一接口。

正确的适配方式：

    entity_emb:   直接引用 unkr_model.ent_emb（nn.Embedding）。
    relation_emb: 直接引用 unkr_model.rel_emb（nn.Embedding）。
    mlp_mean:     dummy nn.Sequential()，让 Updater 的冻结/锚定逻辑不报错，
                  但不含任何参数，不做任何实质操作。
    mlp_var:      同上。
    forward():    直接返回 entity_emb.weight（unKR 是纯嵌入模型，无 GNN）。
    predict():    使用 unkr_model.score_func() 计算 mu；
                  sigma_sq 为固定常数 0.1。

支持的 unKR 模型（自动检测嵌入层命名）：
    - UKGE / UKGE_PSL / GTransE / FocusE / UKGsE / PASSLEAF / SSCDL
      使用 nn.Embedding 类型的 ent_emb 和 rel_emb。
    - UPGAT：使用 nn.Parameter 类型的 ent_emb 和 rel_emb。
    - BEUrRE：使用 min_embedding + delta_embedding 表示实体，
      通过取中心点 (min + delta/2) 来构建统一嵌入层。
"""

import torch
import torch.nn as nn

# 输出负距离分数（需要 sigmoid(-score) 映射到 [0, 1]）的模型
_DISTANCE_BASED_MODELS = {"GTransE", "FocusE"}

# 固定方差常数（作为不确定性估计的占位值）
_DEFAULT_SIGMA_SQ = 0.1


class UnKRModelAdapter(nn.Module):
    """将 unKR 嵌入模型包装成 UnifiedConfidenceUpdater 所需接口的适配器。

    UnifiedConfidenceUpdater 需要模型提供：

        entity_emb   (nn.Embedding)  — 实体嵌入
        relation_emb (nn.Embedding)  — 关系嵌入
        mlp_mean     (nn.Module)     — 均值预测头（本适配器用 dummy 空模块占位）
        mlp_var      (nn.Module)     — 方差预测头（本适配器用 dummy 空模块占位）
        forward(edge_index, edge_type, edge_conf) -> z  — 实体表征
        predict(z_h, r_id, z_t) -> (mu, sigma_sq)      — 置信度预测

    Args:
        unkr_model:   任意 unKR UKGModel 实例。
        emb_dim:      嵌入向量维度，默认自动从模型中推断。
        num_ent:      实体总数，默认自动从模型中推断（BEUrRE / UPGAT 需要）。
        num_rel:      关系总数，默认自动从模型中推断（UPGAT 需要）。
        dropout_rate: 保留参数，当前不使用（兼容旧调用方式）。
        sigma_sq:     predict() 返回的固定方差常数，默认 0.1。
    """

    def __init__(
        self,
        unkr_model: nn.Module,
        emb_dim: int = None,
        num_ent: int = None,
        num_rel: int = None,
        dropout_rate: float = 0.3,
        sigma_sq: float = _DEFAULT_SIGMA_SQ,
    ):
        super().__init__()
        self.unkr_model = unkr_model
        self._model_type = type(unkr_model).__name__
        self._sigma_sq = sigma_sq

        # 1. 直接引用底层模型的实体/关系嵌入（标准模型直接引用，特殊模型用包装层）
        self.entity_emb   = self._get_entity_emb(num_ent, emb_dim)
        self.relation_emb = self._get_relation_emb(num_rel, emb_dim)

        # 2. Dummy MLP 模块（空 nn.Sequential，无任何参数）
        #    UnifiedConfidenceUpdater 会执行：
        #      for param in self.model.mlp_mean.parameters(): ...
        #      for name, param in self.model.mlp_mean.named_parameters(): ...
        #    空模块使这些循环不迭代任何参数，不报错也不做任何实质操作。
        #    unKR 模型本身的 score_func 已经能直接输出置信度预测，
        #    不需要额外的 MLP 分支。
        self.mlp_mean = nn.Sequential()
        self.mlp_var  = nn.Sequential()

    # ------------------------------------------------------------------
    # 内部辅助：提取底层模型的嵌入层
    # ------------------------------------------------------------------

    def _get_entity_emb(self, num_ent: int, emb_dim: int) -> nn.Embedding:
        """提取底层模型的实体嵌入，统一为 nn.Embedding 类型。

        - 标准模型（ent_emb 为 nn.Embedding）：直接引用，共享同一 Parameter。
        - UPGAT（ent_emb 为 nn.Parameter）：包装成新 Embedding 并复制数据。
        - BEUrRE：取 box 中心点 (min + delta/2) 作为单点嵌入近似。
        """
        m = self.unkr_model

        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Embedding):
            # 直接引用：适配器的 entity_emb 与底层模型的 ent_emb 是同一对象，
            # updater 对 entity_emb.weight 的任何修改均自动反映到底层模型。
            return m.ent_emb

        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Parameter):
            # UPGAT：把 Parameter 包装成 Embedding
            n = num_ent or m.ent_emb.shape[0]
            d = emb_dim or m.ent_emb.shape[1]
            emb = nn.Embedding(n, d)
            with torch.no_grad():
                emb.weight.copy_(m.ent_emb.data)
            return emb

        if hasattr(m, "min_embedding") and hasattr(m, "delta_embedding"):
            # BEUrRE：取 box 中心点作为单点嵌入近似
            n = num_ent or m.min_embedding.shape[0]
            d = emb_dim or m.min_embedding.shape[1]
            emb = nn.Embedding(n, d)
            with torch.no_grad():
                center = m.min_embedding.data + m.delta_embedding.data / 2.0
                emb.weight.copy_(center)
            return emb

        raise RuntimeError(
            f"不支持的模型类型 {type(m).__name__}，无法提取 entity_emb。"
        )

    def _get_relation_emb(self, num_rel: int, emb_dim: int) -> nn.Embedding:
        """提取底层模型的关系嵌入，统一为 nn.Embedding 类型。"""
        m = self.unkr_model

        if hasattr(m, "rel_emb") and isinstance(m.rel_emb, nn.Embedding):
            # 直接引用，与底层模型共享同一 Parameter。
            return m.rel_emb

        if hasattr(m, "rel_emb") and isinstance(m.rel_emb, nn.Parameter):
            # UPGAT
            n = num_rel or m.rel_emb.shape[0]
            d = emb_dim or m.rel_emb.shape[1]
            emb = nn.Embedding(n, d)
            with torch.no_grad():
                emb.weight.copy_(m.rel_emb.data)
            return emb

        raise RuntimeError(
            f"不支持的模型类型 {type(m).__name__}，无法提取 relation_emb。"
        )

    # ------------------------------------------------------------------
    # 公开接口：forward / predict
    # ------------------------------------------------------------------

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_conf: torch.Tensor,
    ) -> torch.Tensor:
        """返回实体表征矩阵 z，形状 [num_ent, emb_dim]。

        unKR 是纯嵌入模型（无 GNN 图传播），直接返回实体嵌入权重矩阵。

        Args:
            edge_index: 形状 [2, E] 的边索引张量（头/尾实体 ID），当前未使用。
            edge_type:  形状 [E] 的关系 ID 张量，当前未使用。
            edge_conf:  形状 [E] 的置信度张量，当前未使用。

        Returns:
            z: 形状 [num_ent, emb_dim] 的实体表征张量。
        """
        return self.entity_emb.weight

    def predict(
        self,
        z_h: torch.Tensor,
        r_id: torch.Tensor,
        z_t: torch.Tensor,
    ):
        """使用底层 unKR 模型的 score_func 计算置信度均值与方差。

        Args:
            z_h:  头实体表征，形状 [B, emb_dim] 或 [B, 1, emb_dim]。
            r_id: 关系 ID，形状 [B]。
            z_t:  尾实体表征，形状 [B, emb_dim] 或 [B, 1, emb_dim]。

        Returns:
            mu:       形状 [B] 的置信度均值，值域 [0, 1]。
            sigma_sq: 形状 [B] 的置信度方差，固定常数 0.1。
        """
        # 处理可能的额外维度（[B, 1, D] -> [B, D]）
        if z_h.dim() == 3:
            z_h = z_h.squeeze(1)
        if z_t.dim() == 3:
            z_t = z_t.squeeze(1)

        # 获取关系嵌入（使用底层模型的 rel_emb，确保与 score_func 一致）
        r_emb = self.unkr_model.rel_emb(r_id)
        if r_emb.dim() == 3:
            r_emb = r_emb.squeeze(1)

        # 调用底层模型的 score_func（传 mode='single' 走单样本评分分支）
        score = self.unkr_model.score_func(z_h, r_emb, z_t, mode="single")

        # 处理可能的额外维度（某些模型返回 [B, 1]）
        if score.dim() > 1:
            score = score.squeeze(-1)

        # 距离模型（GTransE、FocusE）输出负距离，需要映射到 [0, 1]；
        # 概率模型（UKGE 等）已输出 [0, 1] 范围的值，直接限幅即可。
        if self._model_type in _DISTANCE_BASED_MODELS:
            mu = torch.sigmoid(-score)
        else:
            mu = score.clamp(0.0, 1.0)

        sigma_sq = torch.full_like(mu, self._sigma_sq)
        return mu, sigma_sq
