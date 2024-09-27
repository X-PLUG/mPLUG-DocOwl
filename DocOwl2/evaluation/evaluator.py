import collections
import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from icecream import ic
import re

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
import editdistance

"""
this script support:
ANLS for DocVQA

RelaxedAccuracy for ChartQA

ContainAccuracy for MultimodalOCR LLM zero-shot text-recognition


"""



def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
        target: Target string.
        prediction: Predicted string.
        theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
        ANLS score.
    """

    edit_distance = editdistance.eval(target, prediction)
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1.0 - normalized_ld if normalized_ld < theta else 0.0

def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
    target: Target string.
    prediction: Predicted string.
    max_relative_change: Maximum relative change.

    Returns:
    Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return float(relative_change <= max_relative_change)
    else:
        return float(prediction.lower() == target.lower())


def exact_match(target: str, prediction: str):
    return float(target == prediction)


def iou_match(target: list, prediction: list, threshold=0.5):
    """
    target/prediction: normalized bbox (list(float)), xyxy
    """
    g_x1, g_y1, g_x2, g_y2 = target
    p_x1, p_y1, p_x2, p_y2 = prediction
    
    g_w = g_x2 - g_x1
    p_w = p_x2 - p_x1
    g_h = g_y2 - g_y1
    p_h = p_y2 - p_y1

    W = (min(g_x2, p_x2)-max(g_x1, p_x1))
    H = (min(g_y2, p_y2)-max(g_y1, p_y1))
    Intersection = W*H
    

    if Intersection <= 0:
        return 0.0

    Union = g_w*g_h + p_w*p_h -Intersection
    # ic(W, H, Intersection, Union)

    if Intersection / Union >= threshold:
        return 1.0
    else:
        return 0.0


def remove_special_chars_and_lower(s):
    pattern = r"[^a-zA-Z0-9\s]"
    # print('raw:', s)
    s = re.sub(pattern, "", s)
    # print('new:', s)
    return s.lower()

def contain_match(target:str, prediction:str):
    def has_word(sentence, word):
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, sentence)
        if match:
            return True
        else:
            return False
    # print(prediction, target, float(has_word(prediction, target)))
    return float(has_word(prediction, target))


def cider(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> float:
    """Compute CIDEr score."""
    coco_tokenizer = PTBTokenizer()
    scorer = Cider()
    score, scores = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores.tolist()]
    return score, scores

def rouge(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> float:
    """Compute CIDEr score."""
    coco_tokenizer = PTBTokenizer()
    scorer = Rouge()
    score, scores = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores.tolist()]
    return score, scores

def meteor(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> float:
    """Compute CIDEr score."""
    coco_tokenizer = PTBTokenizer()
    scorer = Meteor()
    score, scores = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores]
    return score, scores

def bleu(
    ngram: int,
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]) -> float:
    """Compute Bleu score."""
    assert ngram <= 4
    coco_tokenizer = PTBTokenizer()

    scorer = Bleu(4)
    score, scores = scorer.compute_score(
      gts=coco_tokenizer.tokenize({
          str(i): [{"caption": t} for t in target]
          for i, target in enumerate(targets)
      }),
      res=coco_tokenizer.tokenize({
          str(i): [{"caption": prediction}]
          for i, prediction in enumerate(predictions)
      }))
    
    
    score = score[ngram-1]
    scores = scores[ngram-1]
    # ic(score)
    # ic(scores)
    score = float(score) * 100.0
    scores = [float(s) * 100.0 for s in scores]
    return score, scores


def metric_calculate(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str],
    metric_fn: Callable[[str, str], Any],
    normalize_fn: Callable[[str], str] = lambda v: v):
    """Aggregate target-prediction pair metrics over a dataset."""
    assert len(targets) == len(predictions)
    total = 0
    scores = []
    for prediction, target in zip(predictions, targets):
        p = normalize_fn(prediction)
        score = max(metric_fn(normalize_fn(t), p) for t in target)
        scores.append(score)
        total += score
    score = (100.0 * total) / len(targets)
    return score, scores

def doc_evaluate(
    metric: str,
    targets: Sequence[Sequence[str]],
    predictions: Sequence[str]):
    """Calculates evaluation metrics.

    Args:
    metrcs: metric names
    targets: list of list of strings.
    predictions: list of strings.

    Returns:
    dictionary with metric names as keys and metric value as values.
    """
    results = {}

    assert metric in ['ExactAccuracy', 'RelaxedAccuracy', 'ANLS', 'ContainAccuracy', 
                        'CIDEr', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'RougeL', 'Meteor',
                        'IOU@0.5']
    if metric=='ExactAccuracy': # case sensitive
        score, scores = metric_calculate(targets, predictions, metric_fn=exact_match)
    elif metric=='IOU@0.5': 
        score, scores = metric_calculate(targets, predictions, metric_fn=iou_match)
    elif metric == 'ANLS':
        score, scores = metric_calculate(targets, predictions, metric_fn=anls_metric, normalize_fn=lambda v: v.lower())
    elif metric == 'RelaxedAccuracy':
        score, scores = metric_calculate(targets, predictions, metric_fn=relaxed_correctness)
    elif metric == 'ContainAccuracy':
        score, scores = metric_calculate(targets, predictions, metric_fn=contain_match, normalize_fn=remove_special_chars_and_lower)
    elif metric == 'CIDEr':
        score, scores = cider(targets, predictions)
    elif metric == 'BLEU1':
        score, scores = bleu(1, targets, predictions)
    elif metric == 'BLEU2':
        score, scores = bleu(2, targets, predictions)
    elif metric == 'BLEU3':
        score, scores = bleu(3, targets, predictions)
    elif metric == 'BLEU4':
        score, scores = bleu(4, targets, predictions)
    elif metric == 'RougeL':
        score, scores = rouge(targets, predictions)
    elif metric == 'Meteor':
        score, scores = meteor(targets, predictions)
    return score, scores 


