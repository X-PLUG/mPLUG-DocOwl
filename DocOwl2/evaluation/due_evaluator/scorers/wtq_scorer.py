"""
Based on the official implementation from:
https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py
"""

import logging
from typing import List
from operator import itemgetter
import re
from math import isnan, isinf
from abc import ABCMeta, abstractmethod
import unicodedata

from due_evaluator.scorers.accuracy_scorer import AccuracyScorer

logger = logging.getLogger(__name__)


def normalize(x):
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.
        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' +  str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = unicode(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.
        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.
        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


class WtqScorer(AccuracyScorer):
    """WTQ Scorer."""

    def __init__(self, threshold: float = 0.5):
        self.__scores: List[float] = []
        self.threshold = threshold

    @property
    def scores(self):
        return self.__scores

    def to_value(self, original_string, corenlp_value=None):
        """Convert the string to Value object.
        Args:
            original_string (str): Original string
            corenlp_value (str): Optional value returned from CoreNLP
        Returns:
            Value
        """
        if isinstance(original_string, Value):
            # Already a Value
            return original_string
        if not corenlp_value:
            corenlp_value = original_string
        # Number?
        amount = NumberValue.parse(corenlp_value)
        if amount is not None:
            return NumberValue(amount, original_string)
        # Date?
        ymd = DateValue.parse(corenlp_value)
        if ymd is not None:
            if ymd[1] == ymd[2] == -1:
                return NumberValue(ymd[0], original_string)
            else:
                return DateValue(ymd[0], ymd[1], ymd[2], original_string)
        # String.
        return StringValue(original_string)

    def to_value_list(self, original_strings, corenlp_values=None):
        """Convert a list of strings to a list of Values
        Args:
            original_strings (list[str])
            corenlp_values (list[str or None])
        Returns:
            list[Value]
        """
        assert isinstance(original_strings, (list, tuple, set))
        if corenlp_values is not None:
            assert isinstance(corenlp_values, (list, tuple, set))
            assert len(original_strings) == len(corenlp_values)
            return list(set(to_value(x, y) for (x, y)
                    in zip(original_strings, corenlp_values)))
        else:
            return list(set(self.to_value(x) for x in original_strings))

    def check_denotation(self, predicted_values: list, target_values: list):
        """Return True if the predicted denotation is correct.
        
        Args:
            predicted_values (list[Value])
            target_values (list[Value])

        Returns:
            bool
        """
        target_values = self.to_value_list(target_values)
        predicted_values = self.to_value_list(predicted_values)

        # Check size
        if len(target_values) != len(predicted_values):
            return False
        # Check items
        for target in target_values:
            if not any(target.match(pred) for pred in predicted_values):
                return False
        return True

    def add(self, out_items: List[dict], ref_items: List[dict]):
        """Add more items for computing corpus level scores.

        Args:
            out_items: outs from a single document (line)
            ref_items: reference of the evaluated document (line)

        """
        out_ann = sorted(out_items['annotations'], key=itemgetter('key'))
        ref_ann = sorted(ref_items['annotations'], key=itemgetter('key'))
        assert [a['key'][:100] for a in out_ann] == [a['key'][:100] for a in ref_ann]

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
        return "WTQ"
