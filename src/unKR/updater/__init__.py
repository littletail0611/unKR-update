"""增量置信度更新器模块。

从 update 库（littletail0611/update）的 updater.py 第 624-1201 行
原封不动地引入 UnifiedConfidenceUpdater，并通过 UnKRModelAdapter 将
unKR 嵌入模型包装成其所需的接口。

快速使用示例::

    from unKR.data.IncrementalDataset import IncrementalUKGDataset
    from unKR.model.UKGModel.updater_adapter import UnKRModelAdapter
    from unKR.updater import UnifiedConfidenceUpdater

    dataset = IncrementalUKGDataset("data/CN15K")
    unkr_model = UKGE(args)
    adapter = UnKRModelAdapter(unkr_model)
    updater = UnifiedConfidenceUpdater(adapter, dataset, device="cpu")
    new_mu, mean_chg, max_chg, cnt = updater.step(dataset.inc_train)
"""

from unKR.updater.unified_confidence_updater import UnifiedConfidenceUpdater

__all__ = ["UnifiedConfidenceUpdater"]
