# -*- coding: utf-8 -*-

"""F1 Scorer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from due_evaluator.scorers.base_scorer import BaseScorer


@dataclass(eq=False, frozen=True)
class Annotation:
    key: str
    value: str
    value_variants: List[str] = field(default_factory=list)

    def __eq__(self, other):
        if self.key == other.key:
            if self.value == other.value:
                return True
            elif self.value in other.value_variants:
                return True
            elif other.value in self.value_variants:
                return True
        return False


class FScorer(BaseScorer):
    """Corpus level F1 Score evaluator."""

    def __init__(self):
        """Initialize class."""
        self.__precision = []
        self.__recall = []

    @classmethod
    def from_scorers(cls, scorers: List['FScorer']) -> 'FScorer':
        """Get new scorers that is the ensamble of the scorers.

        Args:
            scorers: list of scorers

        Returns:
            FScorer: a new FScorer

        """
        new_scorer = cls()
        for scorer in scorers:
            new_scorer.__precision.extend(scorer.__precision)
            new_scorer.__recall.extend(scorer.__recall)
        return new_scorer
    
    def flatten_annotations(self, annotations: List[Dict[str, Any]]) -> List[Annotation]:
        flatten_items = []
        for annotation in annotations:
            for value in annotation['values']:
                flatten_items.append(Annotation(
                    key=annotation['key'],
                    value=value['value'],
                    value_variants=value['value_variants'] if 'value_variants' in value else []))
        return flatten_items
        

    def add(self, out_items: Dict[str, Any], ref_items: Dict[str, Any]):
        """Add more items for computing corpus level scores.

        Args:
            out_items: outs from a single document (line)
            ref_items: reference of the evaluated document (line)

        """
        prediction_annotations = self.flatten_annotations(out_items['annotations'])
        ref_annotations = self.flatten_annotations(ref_items['annotations'])

        ref_annotations_copy = ref_annotations.copy()
        indicators = []
        for prediction in prediction_annotations:
            if prediction in ref_annotations_copy:
                indicators.append(1)
                ref_annotations_copy.remove(prediction)
            else:
                indicators.append(0)
        self.__add_to_precision(indicators)

        indicators = []
        prediction_annotations_copy = prediction_annotations.copy()
        for ref in ref_annotations:
            if ref in prediction_annotations_copy:
                indicators.append(1)
                prediction_annotations_copy.remove(ref)
            else:
                indicators.append(0)
        self.__add_to_recall(indicators)

    def __add_to_precision(self, item: List[int]):
        if isinstance(item, list):
            self.__precision.extend(item)
        else:
            self.__precision.append(item)

    def __add_to_recall(self, item: List[int]):
        if isinstance(item, list):
            self.__recall.extend(item)
        else:
            self.__recall.append(item)

    def precision(self) -> float:
        """Compute precision.

        Returns:
            float: corpus level precision

        """
        if self.__precision:
            precision = sum(self.__precision) / len(self.__precision)
        else:
            precision = 0.0
        return precision

    @property
    def precision_support(self):
        return self.__precision

    @property
    def recall_support(self):
        return self.__recall

    def recall(self) -> float:
        """Compute recall.

        Returns:
            float: corpus level recall

        """
        if self.__recall:
            recall = sum(self.__recall) / len(self.__recall)
        else:
            recall = 0.0
        return recall

    def f_score(self) -> float:
        """Compute F1 score.

        Returns:
            float: corpus level F1 score.

        """
        precision = self.precision()
        recall = self.recall()
        if precision or recall:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        return fscore

    def false_negative(self) -> int:
        """Return the number of false negatives.

        Returns:
            int: number of false negatives.

        """
        return len(self.__recall) - sum(self.__recall)

    def false_positive(self) -> int:
        """Return the number of false positives.

        Returns:
            int: number of false positives.

        """
        return len(self.__precision) - sum(self.__precision)

    def true_positive(self) -> int:
        """Return number of true positives.

        Returns:
            int: number of true positives.

        """
        return sum(self.__precision)

    def condition_positive(self) -> int:
        """Return number of condition positives.

        Returns:
            int: number of condition positives.

        """
        return len(self.__precision)

    def score(self):
        return self.f_score()

    @classmethod
    def support_feature_scores(cls) -> bool:
        return True

    @classmethod
    def metric_name(cls) -> str:
        return "F1"
