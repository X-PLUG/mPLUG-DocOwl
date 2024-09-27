import logging
from typing import List
from operator import itemgetter

from .base_scorer import BaseScorer

logger = logging.getLogger(__name__)


class AccuracyScorer(BaseScorer):
    """Accuracy Scorer."""

    def __init__(self, threshold: float = 0.5):
        self.__scores: List[float] = []
        self.threshold = threshold

    @property
    def scores(self):
        return self.__scores

    def check_denotation(self, out: list, ref: list) -> bool:
        return out == ref

    def add(self, out_items: List[dict], ref_items: List[dict]):
        """Add more items for computing corpus level scores.

        Args:
            out_items: outs from a single document (line)
            ref_items: reference of the evaluated document (line)

        """
        out_ann = sorted(out_items['annotations'], key=itemgetter('key'))
        ref_ann = sorted(ref_items['annotations'], key=itemgetter('key'))
        assert [a['key'] for a in out_ann] == [a['key'] for a in ref_ann]

        for out, ref in zip(out_ann, ref_ann):
            o_values = [v['value'] for v in out['values']]
            r_values = [v['value'] for v in ref['values']]
            score = int(self.check_denotation(o_values, r_values))
            self.__scores.append(score)

    def score(self) -> float:
        if self.__scores:
            return sum(self.__scores) / len(self.__scores)
        return 0.0

    @classmethod
    def support_feature_scores(cls) -> bool:
        return False

    @classmethod
    def metric_name(cls) -> str:
        return "Accuracy"
