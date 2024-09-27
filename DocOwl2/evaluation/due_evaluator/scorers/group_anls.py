from typing import Any, List, Dict
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
import textdistance

from due_evaluator.scorers.fscorer import FScorer
from due_evaluator.scorers.base_scorer import BaseScorer


@dataclass(eq=False, frozen=True)
class FuzzyAnnotation:
    key: str
    value: str
    value_variants: List[str] = field(default_factory=list)

    def __eq__(self, other):
        def _is_float(val):
            try:
                float(val)
            except ValueError:
                return False
            return True

        def _comp(val, pos) -> float:
            if _is_float(val) or _is_float(pos):
                return float(val == pos)
            return textdistance.levenshtein.normalized_similarity(val, pos)

        def _is_acceptable(val, possible_vals, threshold=.5):
            best_score = max([_comp(val, pos) for pos in possible_vals] + [0.])
            return best_score >= threshold     

        if self.key == other.key:
            if _is_acceptable(other.value, [self.value]):
                return True
            elif _is_acceptable(self.value, other.value_variants):
                return True
            elif _is_acceptable(other.value, self.value_variants):
                return True
        return False


class FuzzyFScorer(FScorer):
    def flatten_annotations(self, annotations: List[Dict[str, Any]]) -> List[FuzzyAnnotation]:
        flatten_items = []
        for annotation in annotations:
            for value in annotation['values']:
                flatten_items.append(FuzzyAnnotation(
                    key=annotation['key'],
                    value=value['value'],
                    value_variants=value['value_variants'] if 'value_variants' in value else []))
        return flatten_items


class GroupAnlsScorer(BaseScorer):
    def __init__(self):
        self.__inner_scorer = FuzzyFScorer()

    def pseudo_documents(self, doc: dict) -> List[dict]:
        docs = []
        for ann in doc['annotations']:
            for val in ann['values']:
                assert 'children' in val
                docs.append({
                    'name': '',
                    'annotations': val['children']
                })
        return docs

    def best_permutation(self, out_items: List[dict], ref_items: List[dict]):
        out_items = self.pseudo_documents(out_items)
        ref_items = self.pseudo_documents(ref_items)
        target_length = max(len(out_items), len(ref_items))
        out_items = self.pad(out_items, target_length)
        ref_items = self.pad(ref_items, target_length)
        matrix = []
        for o in out_items:
            row = []
            for ri, r in enumerate(ref_items):
                 fscorer = FuzzyFScorer()
                 fscorer.add(o, r)
                 row.append(1 - fscorer.f_score())
            matrix.append(row)

        row_ind, col_ind = linear_sum_assignment(np.array(matrix))
        best_out = [out_items[i] for i in row_ind]
        best_ref = [ref_items[i] for i in col_ind]
        return (best_out, best_ref)
    
    def pad(self, items: List[dict], target_length: int):
        for _ in range(target_length - len(items)):
            items.append({'name': '', 'annotations': []})
        return items

    def add(self, out_items: List[str], ref_items: List[str]):
        if len(self.pseudo_documents(out_items)) == 0 and len(self.pseudo_documents(ref_items)) == 0:
            return
        out_perm, ref_perm = self.best_permutation(out_items, ref_items)
        for o, r in zip(out_perm, ref_perm):
            self.__inner_scorer.add(o, r)

    def support_feature_scores(cls) -> bool:
        return False

    def metric_name(cls) -> str:
        return "GROUP-ANLS"

    def score(self) -> float:
        return self.__inner_scorer.score()
