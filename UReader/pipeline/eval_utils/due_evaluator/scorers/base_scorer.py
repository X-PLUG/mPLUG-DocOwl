import abc
from typing import List


class BaseScorer(abc.ABC):
    """Abstract class for scorers."""

    @abc.abstractmethod
    def add(self, out_items: List[dict], ref_items: List[dict]):
        pass

    @abc.abstractmethod
    def score(self):
        pass

    @abc.abstractclassmethod
    def support_feature_scores(cls) -> bool:
        pass

    @abc.abstractclassmethod
    def metric_name(cls) -> str:
        pass
