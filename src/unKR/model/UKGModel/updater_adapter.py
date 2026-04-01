"""模型适配器模块：将 unKR 的嵌入式模型包装成 UnifiedConfidenceUpdater 所需的统一接口。

UnifiedConfidenceUpdater 期望模型具备以下属性和方法：

属性：
    entity_emb   (nn.Embedding)   — 实体嵌入
    relation_emb (nn.Embedding)   — 关系嵌入
    mlp_mean     (nn.Sequential)  — 置信度均值预测头
    mlp_var      (nn.Sequential)  — 置信度方差预测头

方法：
    forward(edge_index, edge_type, edge_conf) -> z
        返回形状 [num_ent, emb_dim] 的实体表征张量。
    predict(z_h, r_id, z_t) -> (mu, sigma_sq)
        给定头/尾实体表征与关系 ID，返回置信度均值和方差。
    heteroscedastic_loss(mu, sigma_sq, target_conf) -> loss
        计算异方差损失。

支持的 unKR 模型（自动检测嵌入层命名）：
    - UKGE / UKGE_PSL / GTransE / FocusE / UKGsE / PASSLEAF / SSCDL
      使用 nn.Embedding 类型的 ent_emb 和 rel_emb。
    - UPGAT：使用 nn.Parameter 类型的 ent_emb 和 rel_emb，额外支持
      调用其图注意力传播逻辑（forward_GAT）。
    - BEUrRE：使用 min_embedding + delta_embedding 表示实体，
      通过取中心点 (min + delta/2) 来构建统一嵌入层。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnKRModelAdapter(nn.Module):
    """将 unKR 嵌入模型包装成 UnifiedConfidenceUpdater 所需接口的适配器。

    Args:
        unkr_model: 任意 unKR UKGModel 实例。
        emb_dim:    嵌入向量维度，默认自动从模型中推断。
        num_ent:    实体总数，默认自动从模型中推断。
        num_rel:    关系总数，默认自动从模型中推断。
        dropout_rate: MLP 预测头的 Dropout 概率，默认 0.3。
    """

    def __init__(
        self,
        unkr_model: nn.Module,
        emb_dim: int = None,
        num_ent: int = None,
        num_rel: int = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.unkr_model = unkr_model

        # ------------------------------------------------------------------
        # 自动探测嵌入层并建立统一的 entity_emb / relation_emb 属性
        # ------------------------------------------------------------------
        emb_dim = emb_dim or self._detect_emb_dim()

        self.entity_emb   = self._build_entity_emb(num_ent, emb_dim)
        self.relation_emb = self._build_relation_emb(num_rel, emb_dim)

        # ------------------------------------------------------------------
        # 双分支异方差预测头
        # ------------------------------------------------------------------
        self.mlp_mean = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )
        self.mlp_var = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, 1),
            nn.Softplus(),
        )

        self._emb_dim = emb_dim
        self._model_type = type(unkr_model).__name__

    # ------------------------------------------------------------------
    # 内部辅助：探测嵌入维度
    # ------------------------------------------------------------------

    def _detect_emb_dim(self) -> int:
        """按优先级探测底层模型的嵌入维度。"""
        m = self.unkr_model
        # 尝试 args.emb_dim
        if hasattr(m, "args") and hasattr(m.args, "emb_dim"):
            return m.args.emb_dim
        # nn.Embedding 类型的 ent_emb
        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Embedding):
            return m.ent_emb.embedding_dim
        # nn.Parameter 类型的 ent_emb（如 UPGAT）
        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Parameter):
            return m.ent_emb.shape[1]
        # BEUrRE：min_embedding
        if hasattr(m, "min_embedding"):
            return m.min_embedding.shape[1]
        raise RuntimeError(
            "无法自动推断嵌入维度，请通过 emb_dim 参数手动指定。"
        )

    # ------------------------------------------------------------------
    # 内部辅助：构建 entity_emb（统一为 nn.Embedding）
    # ------------------------------------------------------------------

    def _build_entity_emb(self, num_ent: int, emb_dim: int) -> nn.Embedding:
        """将底层模型的实体嵌入参数复制到一个新的 nn.Embedding 层。

        - 标准模型（ent_emb 为 nn.Embedding）：直接共享权重数据。
        - UPGAT（ent_emb 为 nn.Parameter）：复制数据，但在 forward 中会
          通过 hook 同步更新。
        - BEUrRE：取 (min + delta/2) 作为近似中心点嵌入。
        """
        m = self.unkr_model

        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Embedding):
            # 直接共享 Parameter 对象，使适配器和底层模型的实体嵌入在梯度和数据两个层面
            # 保持完全同步。这是有意为之：updater 对 entity_emb.weight 的原地修改会
            # 自动反映到原始模型中，无需额外同步步骤。
            n = num_ent or m.ent_emb.num_embeddings
            emb = nn.Embedding(n, emb_dim)
            emb.weight = m.ent_emb.weight
            return emb

        if hasattr(m, "ent_emb") and isinstance(m.ent_emb, nn.Parameter):
            # UPGAT：把 Parameter 数据包装成 Embedding
            n = num_ent or m.ent_emb.shape[0]
            emb = nn.Embedding(n, emb_dim)
            with torch.no_grad():
                emb.weight.copy_(m.ent_emb.data)
            return emb

        if hasattr(m, "min_embedding") and hasattr(m, "delta_embedding"):
            # BEUrRE：取 box 中心点作为单点嵌入近似
            n = num_ent or m.min_embedding.shape[0]
            emb = nn.Embedding(n, emb_dim)
            with torch.no_grad():
                center = m.min_embedding.data + m.delta_embedding.data / 2.0
                emb.weight.copy_(center)
            return emb

        raise RuntimeError(
            f"不支持的模型类型 {type(m).__name__}，无法构建 entity_emb。"
        )

    # ------------------------------------------------------------------
    # 内部辅助：构建 relation_emb（统一为 nn.Embedding）
    # ------------------------------------------------------------------

    def _build_relation_emb(self, num_rel: int, emb_dim: int) -> nn.Embedding:
        """将底层模型的关系嵌入参数复制到一个新的 nn.Embedding 层。"""
        m = self.unkr_model

        if hasattr(m, "rel_emb") and isinstance(m.rel_emb, nn.Embedding):
            # 同 entity_emb：共享 Parameter 对象，使适配器和底层模型的关系嵌入保持同步。
            n = num_rel or m.rel_emb.num_embeddings
            emb = nn.Embedding(n, emb_dim)
            emb.weight = m.rel_emb.weight
            return emb

        if hasattr(m, "rel_emb") and isinstance(m.rel_emb, nn.Parameter):
            # UPGAT
            n = num_rel or m.rel_emb.shape[0]
            emb = nn.Embedding(n, emb_dim)
            with torch.no_grad():
                emb.weight.copy_(m.rel_emb.data)
            return emb

        raise RuntimeError(
            f"不支持的模型类型 {type(m).__name__}，无法构建 relation_emb。"
        )

    # ------------------------------------------------------------------
    # 公开接口：forward / predict / heteroscedastic_loss
    # ------------------------------------------------------------------

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_conf: torch.Tensor,
    ) -> torch.Tensor:
        """计算所有实体的上下文化表征张量。

        对于纯嵌入模型（无 GNN 图传播），直接返回嵌入权重矩阵；
        对于有图注意力的 UPGAT 模型，调用其内部图注意力传播逻辑更新嵌入后返回。

        Args:
            edge_index: 形状 [2, E] 的边索引张量（头/尾实体 ID）。
            edge_type:  形状 [E] 的关系 ID 张量。
            edge_conf:  形状 [E] 的置信度张量（当前未使用，保持接口兼容）。

        Returns:
            z: 形状 [num_ent, emb_dim] 的实体表征张量。
        """
        m = self.unkr_model

        # UPGAT：构造邻接矩阵格式并调用 forward_GAT 更新嵌入
        if self._model_type == "UPGAT" and hasattr(m, "forward_GAT"):
            # UPGAT 的 forward_GAT 期望 adj_matrix = [node_list, edge_list]
            # node_list 为 [2, E]，edge_list 为 [E]
            adj_matrix = [edge_index, edge_type]
            # 构造一个虚拟 triple 以触发图更新，不使用其输出
            dummy_triple = torch.zeros((1, 3), dtype=torch.long, device=edge_index.device)
            try:
                m.forward_GAT(dummy_triple, adj_matrix, negs=None, mode="single")
            except Exception:
                pass
            # 同步更新后的 entity_emb 权重
            with torch.no_grad():
                self.entity_emb.weight.copy_(m.ent_emb.data)
            return self.entity_emb.weight

        # 所有其他模型：直接返回嵌入权重（适配器的 entity_emb 已与底层共享）
        return self.entity_emb.weight

    def predict(
        self,
        z_h: torch.Tensor,
        r_id: torch.Tensor,
        z_t: torch.Tensor,
    ):
        """给定头/尾实体表征和关系 ID，预测置信度均值与方差。

        Args:
            z_h:  形状 [B, emb_dim] 的头实体表征张量。
            r_id: 形状 [B] 的关系 ID 张量。
            z_t:  形状 [B, emb_dim] 的尾实体表征张量。

        Returns:
            mu:       形状 [B] 的置信度均值。
            sigma_sq: 形状 [B] 的置信度方差。
        """
        r_features = self.relation_emb(r_id)

        # 处理 z_h / z_t 可能携带的额外维度（[B, 1, D] -> [B, D]）
        if z_h.dim() == 3:
            z_h = z_h.squeeze(1)
        if z_t.dim() == 3:
            z_t = z_t.squeeze(1)
        if r_features.dim() == 3:
            r_features = r_features.squeeze(1)

        h_r_t = torch.cat([z_h, r_features, z_t], dim=-1)
        mu       = self.mlp_mean(h_r_t).squeeze(-1)
        sigma_sq = self.mlp_var(h_r_t).squeeze(-1)
        return mu, sigma_sq

    def heteroscedastic_loss(
        self,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
        target_conf: torch.Tensor,
    ) -> torch.Tensor:
        """计算异方差高斯负对数似然损失。

        Loss = E[ (target - mu)^2 / (2 * sigma_sq) + 0.5 * log(sigma_sq) ]

        Args:
            mu:          形状 [B] 的预测置信度均值。
            sigma_sq:    形状 [B] 的预测方差。
            target_conf: 形状 [B] 的目标置信度。

        Returns:
            loss: 标量损失值。
        """
        eps = 1e-6
        mse_term = ((target_conf - mu) ** 2) / (2 * (sigma_sq + eps))
        reg_term = 0.5 * torch.log(sigma_sq + eps)
        return torch.mean(mse_term + reg_term)

    def train_mlp_heads(
        self,
        dataset,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str = "cpu",
    ):
        """在 Base 图数据上预训练 MLP 预测头（冻结底层嵌入，仅训练 mlp_mean/mlp_var）。

        Args:
            dataset:    IncrementalUKGDataset 实例。
            epochs:     训练轮数。
            lr:         学习率。
            batch_size: 小批量大小。
            device:     运算设备字符串（"cpu" 或 "cuda:0" 等）。
        """
        import torch.optim as optim

        # 冻结底层嵌入
        for p in self.entity_emb.parameters():
            p.requires_grad = False
        for p in self.relation_emb.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(
            list(self.mlp_mean.parameters()) + list(self.mlp_var.parameters()),
            lr=lr,
        )

        edge_index, edge_type, edge_conf = dataset.get_base_graph_data()
        edge_index = edge_index.to(device)
        edge_type  = edge_type.to(device)
        edge_conf  = edge_conf.to(device)

        self.to(device)
        self.train()

        N = edge_index.shape[1]
        for epoch in range(epochs):
            perm = torch.randperm(N, device=device)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]

                h_ids = edge_index[0, idx]
                t_ids = edge_index[1, idx]
                r_ids = edge_type[idx]
                confs = edge_conf[idx]

                with torch.no_grad():
                    z = self.forward(edge_index, edge_type, edge_conf)

                z_h = z[h_ids]
                z_t = z[t_ids]

                mu, sigma_sq = self.predict(z_h, r_ids, z_t)
                loss = self.heteroscedastic_loss(mu, sigma_sq, confs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[MLP 预训练] Epoch {epoch + 1}/{epochs} — loss={total_loss / max(n_batches, 1):.6f}")

        # 解冻底层嵌入，恢复正常训练状态
        for p in self.entity_emb.parameters():
            p.requires_grad = True
        for p in self.relation_emb.parameters():
            p.requires_grad = True
