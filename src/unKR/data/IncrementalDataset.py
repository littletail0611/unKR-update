"""增量数据集模块：解析 base/inc 两阶段数据格式，构建图张量。

数据目录结构：
    data_dir/
    ├── base/
    │   ├── train.txt   (h\\tr\\tt\\tconf 四列，tab 分隔)
    │   ├── valid.txt
    │   └── test.txt
    └── inc/
        ├── train.txt   (h\\tr\\tt\\tconf 四列 或 h\\tr\\tt 三列混合)
        ├── valid.txt
        └── test.txt
"""

import os
import torch


class IncrementalUKGDataset:
    """兼容 update 库 base/inc 两阶段数据格式的不确定知识图谱增量数据集。

    属性：
        data_dir: 数据集根目录路径。
        add_inverse: 是否为每个关系自动注册反向关系并生成反向边。
            - True（默认）：为 GNN 模型生成双向边，每个关系额外注册 ``_inv`` 反向关系。
            - False：纯嵌入模型（UKGE 等）场景，不生成反向关系和反向边，
              避免 checkpoint 加载时出现无意义的嵌入扩展。
        ent2id: 实体名称到整数 ID 的映射。
        rel2id: 关系名称到整数 ID 的映射。
        num_ent: 当前实体总数（含增量阶段新实体）。
        num_rel: 当前关系总数（add_inverse=True 时含反向关系）。
        belief_state: 全局置信度状态字典，键为 (h_id, r_id, t_id) 三元组。
        new_entities: 增量阶段新出现实体的 ID 集合（OOKB 实体）。
        base_num_ent: Base 阶段的实体数量。
    """

    def __init__(self, data_dir: str, add_inverse: bool = True):
        self.data_dir = data_dir
        self.add_inverse = add_inverse

        self.ent2id: dict = {}
        self.rel2id: dict = {}
        self.num_ent: int = 0
        self.num_rel: int = 0
        self.belief_state: dict = {}
        self.new_entities: set = set()

        print(">>> 正在加载 Base 图谱数据...")
        self.base_train = self._load_file(os.path.join(data_dir, "base", "train"), is_inc=False)
        self.base_valid = self._load_file(os.path.join(data_dir, "base", "valid"), is_inc=False)
        self.base_test  = self._load_file(os.path.join(data_dir, "base", "test"),  is_inc=False)

        self.base_num_ent = self.num_ent
        print(f"Base 阶段加载完成: 实体数={self.base_num_ent}, 关系数={self.num_rel}")

        print(">>> 正在加载 Inc 增量数据...")
        self.inc_train = self._load_file(os.path.join(data_dir, "inc", "train"), is_inc=True)
        self.inc_valid = self._load_file(os.path.join(data_dir, "inc", "valid"), is_inc=True)
        self.inc_test  = self._load_file(os.path.join(data_dir, "inc", "test"),  is_inc=True)

        self.inc_labeled_mask = [f[3] is not None for f in self.inc_train]
        labeled_count = sum(self.inc_labeled_mask)
        print(
            f"Inc 阶段加载完成: 发现新实体数={len(self.new_entities)}, "
            f"有标注事实={labeled_count}/{len(self.inc_train)}"
        )

    # ------------------------------------------------------------------
    # ID 分配辅助方法
    # ------------------------------------------------------------------

    def _get_ent_id(self, ent: str, is_inc: bool) -> int:
        """获取实体 ID；增量阶段的新实体会记录到 new_entities 集合。"""
        if ent not in self.ent2id:
            self.ent2id[ent] = self.num_ent
            if is_inc:
                self.new_entities.add(self.num_ent)
            self.num_ent += 1
        return self.ent2id[ent]

    def _get_rel_id(self, rel: str) -> int:
        """获取关系 ID。

        当 ``add_inverse=True`` 时，同时自动为其注册一个反向关系 ID
        （命名约定：原关系名 + ``_inv``）。
        当 ``add_inverse=False`` 时，只分配正向关系 ID，不注册反向关系。
        """
        if rel not in self.rel2id:
            self.rel2id[rel] = self.num_rel
            self.num_rel += 1
            if self.add_inverse:
                inv_rel = rel + "_inv"
                self.rel2id[inv_rel] = self.num_rel
                self.num_rel += 1
        return self.rel2id[rel]

    # ------------------------------------------------------------------
    # 文件加载
    # ------------------------------------------------------------------

    def _load_file(self, filepath_prefix: str, is_inc: bool) -> list:
        """自动适配 .txt 或 .tsv 后缀加载三元组文件。

        对于增量文件（is_inc=True），支持混合格式：
          - 4 列行 ``h\\tr\\tt\\tconf``：有标注事实，置信度已知。
          - 3 列行 ``h\\tr\\tt``：无标注事实，置信度用 None 标记。
        非增量文件（base）仍要求 4 列。
        belief_state 仅在增量训练文件中由有标注事实填充。

        当 ``add_inverse=True`` 时，为每条正向边自动生成对应的反向边；
        当 ``add_inverse=False`` 时，不生成反向边。
        """
        triplets = []
        filepath = None

        for suffix in (".txt", ".tsv"):
            candidate = filepath_prefix + suffix
            if os.path.exists(candidate):
                filepath = candidate
                break

        if filepath is None:
            print(f"警告: 找不到文件 {filepath_prefix}.txt 或 .tsv")
            return triplets

        is_train = "train" in filepath_prefix

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) >= 4:
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)
                    c_val = float(parts[3])

                    triplets.append((h_id, r_id, t_id, c_val))
                    if self.add_inverse:
                        r_inv_id = r_id + 1
                        triplets.append((t_id, r_inv_id, h_id, c_val))

                    if is_train:
                        self.belief_state[(h_id, r_id, t_id)] = c_val
                        if self.add_inverse:
                            self.belief_state[(t_id, r_inv_id, h_id)] = c_val

                elif len(parts) == 3 and is_inc:
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)

                    triplets.append((h_id, r_id, t_id, None))
                    if self.add_inverse:
                        r_inv_id = r_id + 1
                        triplets.append((t_id, r_inv_id, h_id, None))

        return triplets

    # ------------------------------------------------------------------
    # 图数据访问
    # ------------------------------------------------------------------

    def get_base_graph_data(self):
        """返回 Base 训练图的边张量表示。

        Returns:
            edge_index: 形状 [2, E] 的长整型张量，边的头/尾实体 ID。
            edge_type:  形状 [E] 的长整型张量，边的关系 ID。
            edge_conf:  形状 [E] 的浮点张量，边的置信度分数。
        """
        if not self.base_train:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.float),
            )

        heads = [t[0] for t in self.base_train]
        tails = [t[2] for t in self.base_train]
        rels  = [t[1] for t in self.base_train]
        confs = [t[3] for t in self.base_train]

        edge_index = torch.tensor([heads, tails], dtype=torch.long)
        edge_type  = torch.tensor(rels,  dtype=torch.long)
        edge_conf  = torch.tensor(confs, dtype=torch.float)

        return edge_index, edge_type, edge_conf

    def get_incremental_batches(self, batch_size: int = 1024):
        """产出增量训练事实的批次迭代器。

        每个元素为 ``(h, r, t, conf_or_None)``，其中：
          - conf_or_None 为 float 时表示该事实有已知置信度（有标注）；
          - conf_or_None 为 None 时表示无标注，置信度需由模型预测。
        """
        for i in range(0, len(self.inc_train), batch_size):
            yield self.inc_train[i : i + batch_size]

    def update_belief(self, fact_tuple: tuple, new_confidence: float):
        """更新单条事实的置信度。"""
        self.belief_state[fact_tuple] = new_confidence
