"""
客户端选择器基类模块。
定义了所有选择器必须实现的接口，并提供了 ClientInfo 数据类。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

# 注册表字典（更名为 _REGISTRY，避免与装饰器函数重名）
_REGISTRY = {}


def register_selector(name):
    """装饰器：将选择器类注册到全局字典"""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_selector(name):
    """根据名称获取选择器类（供外部使用）"""
    return _REGISTRY.get(name)


class SelectionMetrics(Enum):
    LOSS = "loss"
    GRADIENT_NORM = "gradient_norm"
    DATA_SIZE = "data_size"
    COMPUTATION_TIME = "computation_time"
    UPLOAD_TIME = "upload_time"
    LAST_SELECTED = "last_selected"
    ACCURACY = "accuracy"


@dataclass
class ClientInfo:
    client_id: int
    data_size: int = 0
    last_loss: float = float('inf')
    last_accuracy: float = 0.0
    last_test_loss: float = float('inf')
    gradient_norm: float = 0.0
    computation_time: float = 0.0
    upload_time: float = 0.0
    capability: float = 1.0
    bandwidth: float = 10.0
    last_selected_round: int = -1
    participation_count: int = 0
    is_available: bool = True
    data_distribution: Optional[np.ndarray] = None


class BaseSelector(ABC):
    # Most historical baselines in this codebase consume current pre-selection
    # metrics. Lightweight selectors can override this to avoid unnecessary
    # client-side metric collection.
    requires_preselection_metrics = True

    def __init__(self, total_clients: int, clients_per_round: int, **kwargs):
        self.total_clients = total_clients
        self.clients_per_round = clients_per_round
        self.round = 0
        self.client_info = {}
        self.selected_clients = []
        self.needs_round_selection = True

    def init_client_info(self, client_ids: List[int]):
        for client_id in client_ids:
            self.client_info[client_id] = ClientInfo(client_id=client_id)

    @abstractmethod
    def select(self, available_clients: List[int],
               client_metrics: Optional[Dict[int, Dict[str, Any]]] = None) -> List[int]:
        pass

    def get_clients_for_round(self, round_idx: int) -> List[int]:
        return self.selected_clients

    def update(self, selected_clients: List[int], results: Dict[int, Dict[str, Any]]):
        self.round += 1
        for client_id, result in results.items():
            if client_id in self.client_info:
                info = self.client_info[client_id]
                info.last_selected_round = self.round
                info.participation_count += 1
                if 'loss' in result:
                    info.last_loss = result['loss']
                if 'accuracy' in result:
                    info.last_accuracy = result['accuracy']
                if 'test_loss' in result:
                    info.last_test_loss = result['test_loss']
                if 'computation_time' in result:
                    info.computation_time = result['computation_time']
                if 'upload_time' in result:
                    info.upload_time = result['upload_time']
                if 'gradient_norm' in result:
                    info.gradient_norm = result['gradient_norm']