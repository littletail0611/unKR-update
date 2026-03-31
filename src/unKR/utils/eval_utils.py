# 工具类：评估指标 (MSE, MAE 等)、日志记录

import torch
import numpy as np

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def calculate_metrics(y_true, y_pred):
    """计算回归任务的核心评估指标：MSE、MAE、RMSE。

    Args:
        y_true: 真实置信度，可为 ``torch.Tensor``、``np.ndarray`` 或 ``list``。
        y_pred: 预测置信度，类型同上。

    Returns:
        dict: ``{"MSE": float, "MAE": float, "RMSE": float}``
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0}

    if _SKLEARN_AVAILABLE:
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
    else:
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))

    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


def evaluate_model(model, test_data, z, device="cpu"):
    """在给定的测试集上评估模型预测性能。

    只评估有已知置信度（第 4 个元素不为 ``None``）的事实。

    Args:
        model: 实现 ``predict(z_h, r_idx, z_t)`` 接口的模型
            （如 :class:`~unKR.model.UKGModel.updater_adapter.UnKRModelAdapter`）。
        test_data: 三元组列表，元素为 ``(h, r, t, conf_or_None)``。
        z: GNN 聚合后的实体特征张量，shape ``[num_ent, emb_dim]``。
        device: 运行设备，默认 ``"cpu"``。

    Returns:
        dict: ``{"MSE": float, "MAE": float, "RMSE": float}``
    """
    if not test_data:
        return calculate_metrics([], [])

    labeled_data = [fact for fact in test_data if fact[3] is not None]
    if not labeled_data:
        return calculate_metrics([], [])

    model.eval()

    h_idx = torch.tensor([fact[0] for fact in labeled_data], dtype=torch.long).to(device)
    r_idx = torch.tensor([fact[1] for fact in labeled_data], dtype=torch.long).to(device)
    t_idx = torch.tensor([fact[2] for fact in labeled_data], dtype=torch.long).to(device)
    y_true = torch.tensor([fact[3] for fact in labeled_data], dtype=torch.float)

    with torch.no_grad():
        z_h = z[h_idx]
        z_t = z[t_idx]
        mu_pred, _ = model.predict(z_h, r_idx, z_t)

    return calculate_metrics(y_true, mu_pred.cpu())


def evaluate_belief_state(dataset, test_data):
    """评估系统当前维护的全局 belief state 与真实标签的差异。

    Args:
        dataset: 含 ``belief_state`` 字典的
            :class:`~unKR.data.IncrementalDataset.UKGDataset` 实例。
        test_data: 三元组列表，元素为 ``(h, r, t, conf)``。

    Returns:
        dict: ``{"MSE": float, "MAE": float, "RMSE": float}``
    """
    if not test_data:
        return calculate_metrics([], [])

    y_true = []
    y_pred = []

    for h_id, r_id, t_id, c_true in test_data:
        if c_true is None:
            continue
        fact_tuple = (h_id, r_id, t_id)
        c_pred = dataset.belief_state.get(fact_tuple, 0.5)
        y_true.append(c_true)
        y_pred.append(c_pred)

    return calculate_metrics(y_true, y_pred)


class Logger:
    """简单的指标日志打印工具。"""

    @staticmethod
    def print_metrics(stage_name, metrics):
        """打印评估指标。

        Args:
            stage_name: 阶段名称字符串，显示在标题中。
            metrics: ``{"MSE": float, "MAE": float, "RMSE": float}`` 字典。
        """
        print(f"[{stage_name} Evaluation]")
        print(f"  - MSE:  {metrics['MSE']:.4f}")
        print(f"  - MAE:  {metrics['MAE']:.4f}")
        print(f"  - RMSE: {metrics['RMSE']:.4f}")
        print("-" * 30)
