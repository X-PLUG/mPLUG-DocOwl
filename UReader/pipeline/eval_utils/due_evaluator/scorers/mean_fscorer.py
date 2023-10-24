from typing import List

from .fscorer import FScorer
from .base_scorer import BaseScorer


class MeanFScorer(BaseScorer):
    def __init__(self):
        self.__scores: List[float] = []

    def add(self, out_items: List[str], ref_items: List[str]):
        fscorer = FScorer()
        fscorer.add(out_items, ref_items)
        self.__scores.append(fscorer.f_score())

    def support_feature_scores(cls) -> bool:
        return False

    def metric_name(cls) -> str:
        return "MEAN-F1"

    def score(self) -> float:
        if self.__scores:
            return sum(self.__scores) / len(self.__scores)
        return 0.0
