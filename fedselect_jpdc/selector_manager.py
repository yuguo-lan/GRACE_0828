from typing import List, Dict, Any, Optional
from .selection import BaseSelector, get_selector


class SelectorManager:
    def __init__(self, selector_name: str, total_clients: int, clients_per_round: int,
                 selector_params: Optional[Dict[str, Any]] = None):
        selector_class = get_selector(selector_name)
        if selector_class is None:
            raise ValueError(f"Unknown selector: {selector_name}")
        # 将 selector_params 展开作为关键字参数传递给选择器
        self.selector: BaseSelector = selector_class(
            total_clients=total_clients,
            clients_per_round=clients_per_round,
            **(selector_params or {})
        )
        self.client_info = {}
        self.total_clients = total_clients

    def init_clients(self, client_ids: List[int]):
        self.selector.init_client_info(client_ids)

    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[int]:
        return self.selector.select(available_clients, client_metrics)

    def update(self, selected_clients: List[int], results: Dict[int, Dict[str, Any]]):
        self.selector.update(selected_clients, results)

    @property
    def requires_preselection_metrics(self) -> bool:
        return bool(getattr(self.selector, "requires_preselection_metrics", True))
