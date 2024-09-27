from typing import List
import tempfile
from collections import defaultdict
import os

from due_evaluator.scorers.fscorer import FScorer
from due_evaluator.scorers.base_scorer import BaseScorer


GEVAL_BINARY = os.getenv('GEVAL_BINARY', '/data/shared/bin/geval')
GEVAL_METRIC = os.getenv('GEVAL_METRIC', 'MultiLabel-F1:cN')


class GevalScorer(BaseScorer):
    def __init__(self):
        self.__ref = tempfile.NamedTemporaryFile('w+t')
        self.__out = tempfile.NamedTemporaryFile('w+t')
        self.__ref_data = defaultdict(set)
        self.__out_data = defaultdict(set)

    @staticmethod
    def add_to_geval_data(data, line):
        name = line['name']
        for annotation in line['annotations']:
            for idx, val in enumerate(annotation['values'], 1):
                for child in val['children']:
                    new_name = child['key'] + '__' + str(idx) if '__' in child['key'] else child['key']
                    if child['values'] and child['values'] != ['']:
                        new_value = '|'.join([v['value'].replace(' ', '_') for v in child['values']])
                        data[name].add(f'{new_name}={new_value}')

    def save_geval_files(self):
        for name in sorted(self.__ref_data.keys()):
            self.__ref.write(' '.join(self.__ref_data[name]) + '\n')
            self.__out.write(' '.join(self.__out_data[name]) + '\n')

    def add(self, out_items: List[str], ref_items: List[str]):
        self.add_to_geval_data(self.__out_data, out_items)
        self.add_to_geval_data(self.__ref_data, ref_items)

    def support_feature_scores(cls) -> bool:
        return False

    def metric_name(cls) -> str:
        return "GEVAL"
    
    def run_geval(self):
        self.__ref.flush()
        self.__out.flush()
        try:
            return float(os.popen(f'{GEVAL_BINARY} -o {self.__out.name} -e {self.__ref.name} --metric {GEVAL_METRIC}').read())
        except:
            return -1

    def score(self) -> float:
        self.save_geval_files()
        return self.run_geval()
