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
      通过取中心点 (min + delta/2) 来构建统一嵌入层；
      predict() 使用 Box Embedding 交集体积计算置信度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # BEUrRE：取 box 中心点作为单点嵌入近似。
            # entity_emb.weight 存储各实体 box 的中心坐标。
            # updater 直接操作此表（例如 _init_new_entities），
            # _predict_beurre() 则将这些中心值作为 min_embedding 代理配合
            # delta_embedding 的均值近似重建 box 并计算交集概率。
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
        """提取底层模型的关系嵌入，统一为 nn.Embedding 类型。

        BEUrRE 没有 rel_emb，用 rel_trans_for_head 作为代理关系嵌入。
        """
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

        if hasattr(m, "rel_trans_for_head"):
            # BEUrRE：没有标准 rel_emb，用 rel_trans_for_head 作为代理。
            # UnifiedConfidenceUpdater 只需要冻结这个 Embedding 的梯度，
            # 并不实际用其值做打分（predict() 中会单独处理 BEUrRE 的关系变换）。
            n = num_rel or m.rel_trans_for_head.shape[0]
            d = emb_dim or m.rel_trans_for_head.shape[1]
            emb = nn.Embedding(n, d)
            with torch.no_grad():
                emb.weight.copy_(m.rel_trans_for_head.data)
            return emb

        raise RuntimeError(
            f"不支持的模型类型 {type(m).__name__}，无法提取 relation_emb。"
        )

    # ------------------------------------------------------------------
    # 公开接口：forward / predict / expand_embedding
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

        对于 BEUrRE，使用 Box Embedding 交集体积计算置信度（特殊路径）。

        Args:
            z_h:  头实体表征，形状 [B, emb_dim] 或 [B, 1, emb_dim]。
            r_id: 关系 ID，形状 [B]。
            z_t:  尾实体表征，形状 [B, emb_dim] 或 [B, 1, emb_dim]。

        Returns:
            mu:       形状 [B] 的置信度均值，值域 [0, 1]。
            sigma_sq: 形状 [B] 的置信度方差，固定常数。
        """
        # 处理可能的额外维度（[B, 1, D] -> [B, D]）
        if z_h.dim() == 3:
            z_h = z_h.squeeze(1)
        if z_t.dim() == 3:
            z_t = z_t.squeeze(1)

        if self._model_type == "BEUrRE":
            return self._predict_beurre(z_h, r_id, z_t)

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

    def _predict_beurre(
        self,
        z_h: torch.Tensor,
        r_id: torch.Tensor,
        z_t: torch.Tensor,
    ):
        """BEUrRE 专用的置信度预测，通过 Box Embedding 交集体积计算。

        z_h / z_t 是 entity_emb.weight 中存储的 box 中心坐标（min + delta/2）。
        使用底层 BEUrRE 模型的关系仿射变换参数和 Gumbel softmax 交集计算。
        delta 使用底层模型的全局均值 delta 作为近似（因为 predict() 只接收
        嵌入向量而非实体 ID，无法精确查找每个实体的 delta）。

        Args:
            z_h:  头实体 box 中心坐标，形状 [B, emb_dim]。
            r_id: 关系 ID，形状 [B]。
            z_t:  尾实体 box 中心坐标，形状 [B, emb_dim]。

        Returns:
            mu:       形状 [B] 的置信度均值，值域 [0, 1]。
            sigma_sq: 形状 [B] 的置信度方差，固定常数。
        """
        m = self.unkr_model
        device = z_h.device

        # 用全局均值 delta 近似每个实体的 box 尺寸
        mean_delta = m.delta_embedding.data.mean(dim=0).to(device)
        delta_h = mean_delta.unsqueeze(0).expand_as(z_h)
        delta_t = mean_delta.unsqueeze(0).expand_as(z_t)

        # 应用关系仿射变换（头实体侧）
        r_trans_h = m.rel_trans_for_head[r_id].to(device)
        r_scale_h = F.relu(m.rel_scale_for_head[r_id].to(device))

        # 应用关系仿射变换（尾实体侧）
        r_trans_t = m.rel_trans_for_tail[r_id].to(device)
        r_scale_t = F.relu(m.rel_scale_for_tail[r_id].to(device))

        # 变换后的 box（用中心 - delta/2 近似 min_embed）
        h_min = (z_h - delta_h / 2.0) + r_trans_h
        h_delta = delta_h * r_scale_h

        t_min = (z_t - delta_t / 2.0) + r_trans_t
        t_delta = delta_t * r_scale_t

        h_max = h_min + h_delta
        t_max = t_min + t_delta

        # Gumbel softmax 交集计算
        gumbel_beta = m.gumbel_beta

        int_min = gumbel_beta * torch.logsumexp(
            torch.stack([h_min / gumbel_beta, t_min / gumbel_beta]), dim=0
        )
        int_min = torch.max(int_min, torch.max(h_min, t_min))

        int_max = -gumbel_beta * torch.logsumexp(
            torch.stack([-h_max / gumbel_beta, -t_max / gumbel_beta]), dim=0
        )
        int_max = torch.min(int_max, torch.min(h_max, t_max))

        int_delta = int_max - int_min

        # 计算 log 体积（仿照 BEUrRE.log_volumes）
        euler_gamma = m.euler_gamma
        eps = torch.finfo(torch.float32).tiny

        def _log_vol(delta: torch.Tensor) -> torch.Tensor:
            return torch.sum(
                torch.log(
                    F.softplus(
                        delta - 2.0 * euler_gamma * gumbel_beta
                    ).clamp_min(eps)
                ),
                dim=-1,
            )

        # 条件概率 P(h | t, r) ≈ Vol(intersection) / Vol(t_box)
        log_prob = _log_vol(int_delta) - _log_vol(t_delta)
        mu = torch.exp(log_prob).clamp(0.0, 1.0)

        sigma_sq = torch.full_like(mu, self._sigma_sq)
        return mu, sigma_sq

    def expand_embedding(self, num_ent: int, num_rel: int = None):
        """扩展嵌入表，以容纳增量阶段引入的新实体（或新关系）。

        当预训练 checkpoint 的实体数（base_num_ent）少于当前数据集的实体总数
        （num_ent）时，需要调用此方法扩展嵌入表，新实体行使用现有嵌入的均值初始化。

        对于 BEUrRE 模型，同时扩展底层模型的 min_embedding 和 delta_embedding。
        对于标准模型，同时更新底层模型的 ent_emb（两者本来是同一对象，但若之前
        通过 load_checkpoint 已做了手工扩展，此方法可作为补充的显式扩展调用）。

        Args:
            num_ent: 目标实体总数（>= 当前 entity_emb 的行数时才实际执行扩展）。
            num_rel: 目标关系总数（可选；>= 当前 relation_emb 的行数时才扩展）。
        """
        curr_n = self.entity_emb.weight.shape[0]
        if num_ent > curr_n:
            n_new = num_ent - curr_n
            emb_dim = self.entity_emb.weight.shape[1]
            device = self.entity_emb.weight.device

            if self._model_type == "BEUrRE":
                m = self.unkr_model
                with torch.no_grad():
                    # 扩展 min_embedding
                    mean_min = m.min_embedding.data.mean(dim=0)
                    new_min_rows = mean_min.unsqueeze(0).expand(n_new, -1).contiguous()
                    new_min = torch.cat([m.min_embedding.data, new_min_rows], dim=0)
                    m.min_embedding = nn.Parameter(new_min.to(device))

                    # 扩展 delta_embedding
                    mean_delta = m.delta_embedding.data.mean(dim=0)
                    new_delta_rows = mean_delta.unsqueeze(0).expand(n_new, -1).contiguous()
                    new_delta = torch.cat([m.delta_embedding.data, new_delta_rows], dim=0)
                    m.delta_embedding = nn.Parameter(new_delta.to(device))

                    # 从更新后的 min/delta 重建 entity_emb（box 中心）
                    center = m.min_embedding.data + m.delta_embedding.data / 2.0
                    new_emb = nn.Embedding(num_ent, emb_dim)
                    new_emb.weight = nn.Parameter(center.clone())
                self.entity_emb = new_emb.to(device)
            else:
                # 标准模型：entity_emb IS unkr_model.ent_emb（同一对象）
                mean_emb = self.entity_emb.weight.data.mean(dim=0)
                new_rows = mean_emb.unsqueeze(0).expand(n_new, -1).contiguous()
                new_weight = torch.cat([self.entity_emb.weight.data, new_rows], dim=0)
                new_emb = nn.Embedding(num_ent, emb_dim)
                with torch.no_grad():
                    new_emb.weight.copy_(new_weight)
                new_emb = new_emb.to(device)
                self.entity_emb = new_emb
                if hasattr(self.unkr_model, "ent_emb"):
                    self.unkr_model.ent_emb = new_emb

            print(f">>> expand_embedding: entity_emb {curr_n} → {num_ent}")

        if num_rel is not None:
            curr_nr = self.relation_emb.weight.shape[0]
            if num_rel > curr_nr:
                n_new = num_rel - curr_nr
                emb_dim = self.relation_emb.weight.shape[1]
                device = self.relation_emb.weight.device

                mean_rel = self.relation_emb.weight.data.mean(dim=0)
                new_rows = mean_rel.unsqueeze(0).expand(n_new, -1).contiguous()
                new_weight = torch.cat([self.relation_emb.weight.data, new_rows], dim=0)
                new_rel = nn.Embedding(num_rel, emb_dim)
                with torch.no_grad():
                    new_rel.weight.copy_(new_weight)
                new_rel = new_rel.to(device)
                self.relation_emb = new_rel
                if hasattr(self.unkr_model, "rel_emb") and isinstance(
                    self.unkr_model.rel_emb, nn.Embedding
                ):
                    self.unkr_model.rel_emb = new_rel

                print(f">>> expand_embedding: relation_emb {curr_nr} → {num_rel}")
