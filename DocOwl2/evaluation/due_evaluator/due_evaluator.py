import sys
from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional, Set, Tuple, TypeVar, Union, Generic
from copy import deepcopy

from due_evaluator.scorers import AnlsScorer, BaseScorer, FScorer, MeanFScorer, WtqScorer, GevalScorer, GroupAnlsScorer

TScorer = TypeVar("TScorer", bound=BaseScorer)


class DueEvaluator:
    """Due Evaluator."""

    def __init__(
        self,
        reference: List[List[dict]],
        answers: List[List[dict]],
        property_set: Optional[Set[str]] = None,
        ignore_case: bool = False,
        path: Optional[str] = None,
        metric: Optional[str] = 'F1',
    ):
        """Initialize DueEvaluator.

        Arguments:
            reference: reference
            answers: answers to be evaluated
            separator: property name and property value separator
            property_set: if given, the score will be computed taking into account only these properties.
            ignore_case: if true, compute scores ignoring casing.
            path: Optional, the path to the evaluated files.

        """
        self.reference = reference
        self.answers = answers
        self.property_set = property_set
        self.ignore_case = ignore_case
        self.metric = metric
        self.__path = path
        self.__general_scorer, self.__property_scorers = self._evalute()

    @property
    def general_scorer(self) -> BaseScorer:
        """Get general scorer.

        Returns:
            FScorer: the general scorer.

        """
        return self.__general_scorer

    @property
    def property_scorers(self) -> DefaultDict[str, BaseScorer]:
        """Get a scorer for each property.

        Returns:
            Fscorer: the general scorer.

        """
        return self.__property_scorers

    @property
    def path(self) -> Optional[str]:
        """Return the path of the evaluated file or None--in case when not ealuating a file.

        Returns:
            Optional[str]: the path of the evaluated file or None.

        """
        return self.__path

    def create_scorer(self) -> BaseScorer:
        scorer: BaseScorer
        if self.metric == 'F1':
            scorer = FScorer()
        elif self.metric == 'ANLS':
            scorer = AnlsScorer()
        elif self.metric == 'MEAN-F1':
            scorer = MeanFScorer()
        elif self.metric == 'WTQ':
            scorer = WtqScorer()
        elif self.metric == 'GROUP-ANLS':
            scorer = GroupAnlsScorer()
        elif self.metric == 'GEVAL':
            scorer = GevalScorer()
        else:
            raise ValueError(self.metric)
        return scorer

    def filter_properties(self, doc: dict, values: Union[str, List[str], Set[str]]) -> List[str]:
        """Filter the list of properties by provided property name(s).

        Args:
            doc: document with annotations
            values: a property name(s)

        Returns:
            doc: with filtered annotations

        """
        if isinstance(values, str):
            values = [values]

        doc_copy = deepcopy(doc)
        doc_copy['annotations'] = [a for a in doc_copy['annotations'] if a['key'] in values]
        return doc_copy

    def _evalute(self) -> Tuple[BaseScorer, DefaultDict[str, BaseScorer]]:
        """Evaluate the output file.

        Returns:
            tuple: general fscorer and a dict with fscorer per label.

        """
        label_scorers: DefaultDict[str, BaseScorer] = defaultdict(self.create_scorer)
        general_scorer = self.create_scorer()
        reference_labels: Set[str] = set()
        for ans_items, ref_items in zip(self.answers, self.reference):

            if self.ignore_case:
                ans_items = self.uppercase_items(ans_items)
                ref_items = self.uppercase_items(ref_items)

            if general_scorer.support_feature_scores():
                reference_labels |= set(a['key'] for a in ref_items['annotations'])

                for label in set(item['key'] for item in ref_items['annotations'] + ans_items['annotations']):
                    if self.property_set and label not in self.property_set:
                        continue
                    label_out = self.filter_properties(ans_items, label)
                    label_ref = self.filter_properties(ref_items, label)
                    label_scorers[label].add(label_out, label_ref)

            if general_scorer.support_feature_scores() and self.property_set:
                ans_items = self.filter_properties(ans_items, self.property_set)
                ref_items = self.filter_properties(ref_items, self.property_set)

            general_scorer.add(ans_items, ref_items)

        for label in list(label_scorers.keys()):
            if label not in reference_labels:
                del label_scorers[label]

        return general_scorer, label_scorers

    def uppercase_items(self, document: dict) -> List[str]:
        """Upperecase annotation values.

        Args:
            document: document with annotations that should be uppercased.

        Returns:
            document: with with uppercased annotations.

        """
        for item in document['annotations']:
            for value_dict in item['values']:
                if 'value' in value_dict:
                    value_dict['value'] = value_dict['value'].upper()
                if 'value_variants' in value_dict:
                    value_dict['value_variants'] = [variant.upper() for variant in value_dict['value_variants']]
                if 'children' in value_dict:
                    value_dict['children'] = self.uppercase_items({'annotations': value_dict['children']})['annotations']
        return document

    def line_by_line(self):
        """Compute scores line by line.

        Returns:
            List: list with scorers.

        """
        scores = []
        for ans_items, ref_items in zip(self.answers, self.reference):
            fscorer = self.create_scorer()
            if self.ignore_case:
                ans_items = self.uppercase_items(ans_items)
                ref_items = self.uppercase_items(ref_items)
            fscorer.add(ans_items, ref_items)
            scores.append(fscorer.score())
        return scores
