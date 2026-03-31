# 增量数据加载模块：兼容 update 库的 base/inc 二阶段数据格式

import os
import torch


class UKGDataset:
    """加载 update 库格式的二阶段数据集。

    目录结构::

        data_dir/
        ├── base/
        │   ├── train.txt   (h\\tr\\tt\\tconf)
        │   ├── valid.txt
        │   └── test.txt
        └── inc/
            ├── train.txt   (h\\tr\\tt\\tconf 或 h\\tr\\tt)
            ├── valid.txt
            └── test.txt

    文件后缀自动识别 ``.txt`` / ``.tsv``。

    Attributes:
        data_dir: 数据集根目录。
        ent2id: 实体名 -> ID 映射。
        rel2id: 关系名 -> ID 映射（含反向关系 ``rel_inv``）。
        num_ent: 当前总实体数。
        num_rel: 当前总关系数（含反向关系）。
        belief_state: ``{(h, r, t): confidence}`` 全局信念状态字典。
        new_entities: inc 阶段新增实体 ID 集合。
        base_num_ent: base 阶段结束时的实体数量。
        base_train / base_valid / base_test: base 阶段三元组列表。
        inc_train / inc_valid / inc_test: inc 阶段三元组列表。
        inc_labeled_mask: inc_train 中有已知置信度的布尔掩码列表。
    """

    def __init__(self, data_dir="data/CN15K"):
        self.data_dir = data_dir

        self.ent2id = {}
        self.rel2id = {}
        self.num_ent = 0
        self.num_rel = 0
        self.belief_state = {}
        self.new_entities = set()

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
        print(f"Inc 阶段加载完成: 发现新实体数={len(self.new_entities)}, "
              f"有标注事实={labeled_count}/{len(self.inc_train)}")

    def _get_ent_id(self, ent, is_inc):
        if ent not in self.ent2id:
            self.ent2id[ent] = self.num_ent
            if is_inc:
                self.new_entities.add(self.num_ent)
            self.num_ent += 1
        return self.ent2id[ent]

    def _get_rel_id(self, rel):
        """获取关系 ID，同时自动注册反向关系 ID（命名约定：原关系名 + ``_inv``）。

        反向关系 ID 始终等于正向关系 ID + 1，由此保证 ``r_inv = r_id + 1``。
        """
        if rel not in self.rel2id:
            self.rel2id[rel] = self.num_rel
            self.num_rel += 1
            inv_rel = rel + "_inv"
            self.rel2id[inv_rel] = self.num_rel
            self.num_rel += 1
        return self.rel2id[rel]

    def _load_file(self, filepath_prefix, is_inc):
        """自动适配 ``.txt`` 或 ``.tsv`` 后缀加载三元组文件。

        对增量文件（``is_inc=True``）支持混合格式：

        * 4 列行 ``h\\tr\\tt\\tconf``：有标注事实，置信度已知。
        * 3 列行 ``h\\tr\\tt``：无标注事实，置信度未知（第 4 元素存为 ``None``）。

        每条正向三元组同时生成一条反向三元组，正反向均写入结果列表。
        ``belief_state`` 仅由 base/inc 训练文件的有标注事实填充。
        """
        triplets = []
        filepath = None

        if os.path.exists(filepath_prefix + ".txt"):
            filepath = filepath_prefix + ".txt"
        elif os.path.exists(filepath_prefix + ".tsv"):
            filepath = filepath_prefix + ".tsv"
        else:
            print(f"警告: 找不到文件 {filepath_prefix}.txt 或 .tsv")
            return triplets

        is_train = "train" in filepath_prefix

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 4:
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)
                    c_val = float(parts[3])

                    triplets.append((h_id, r_id, t_id, c_val))
                    r_inv_id = r_id + 1
                    triplets.append((t_id, r_inv_id, h_id, c_val))

                    if is_train:
                        self.belief_state[(h_id, r_id, t_id)] = c_val
                        self.belief_state[(t_id, r_inv_id, h_id)] = c_val

                elif len(parts) == 3 and is_inc:
                    h_id = self._get_ent_id(parts[0], is_inc)
                    r_id = self._get_rel_id(parts[1])
                    t_id = self._get_ent_id(parts[2], is_inc)

                    triplets.append((h_id, r_id, t_id, None))
                    r_inv_id = r_id + 1
                    triplets.append((t_id, r_inv_id, h_id, None))

        return triplets

    def get_base_graph_data(self):
        """返回 base 图谱的结构数据。

        Returns:
            tuple: ``(edge_index, edge_type, edge_confidence)``，均为 ``torch.Tensor``。

            * ``edge_index``: shape ``[2, E]``，头尾实体索引。
            * ``edge_type``: shape ``[E]``，关系索引。
            * ``edge_confidence``: shape ``[E]``，置信度。
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
        edge_type = torch.tensor(rels, dtype=torch.long)
        edge_confidence = torch.tensor(confs, dtype=torch.float)

        return edge_index, edge_type, edge_confidence

    def update_belief(self, fact_tuple, new_confidence):
        """更新单条事实的信念置信度。

        Args:
            fact_tuple: ``(h, r, t)`` 整数元组。
            new_confidence: 新的置信度值。
        """
        self.belief_state[fact_tuple] = new_confidence

    def get_incremental_batches(self, batch_size=1024):
        """生成增量训练事实的批次迭代器。

        每个批次元素为 ``(h, r, t, conf_or_None)``，其中
        ``conf_or_None`` 为 ``None`` 时表示无标注事实，置信度需由模型预测。

        Args:
            batch_size: 每批次大小，默认 1024。

        Yields:
            list: 长度不超过 ``batch_size`` 的三元组列表。
        """
        for i in range(0, len(self.inc_train), batch_size):
            yield self.inc_train[i:i + batch_size]
