import copy
import json
import os
import random

import tqdm
import jsonlines
from icecream import ic


import collections
import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from icecream import ic
import re

from .due_evaluator.due_evaluator import DueEvaluator
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
import editdistance

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))
    print('save %d samples to %s' % (len(data), filename))
def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
      for line in jsonlines.Reader(f):
        data.append(line)
    return data


"""
this script support:
ANLS for DocVQA

RelaxedAccuracy for ChartQA

ContainAccuracy for MultimodalOCR LLM zero-shot text-recognition


"""
DUE_BENCHMARK = './benchmark_files/DUE/'
DUE_FOR_DONUT = './benchmark_files/DONUT/'

def dataset2metrics(dataset_name):
    if dataset_name in ['DocVQA', 'InfographicsVQA']:
        return ['ANLS']
    elif dataset_name in ['KleisterCharity', 'DeepForm']:
        return ['F1']
    elif dataset_name in ['TabFact']:
        return ['F1']
    elif dataset_name in ['PWC']:
        return ['GROUP-ANLS']
    elif dataset_name in ['WikiTableQuestions']:
        return ['WTQ']
    else:
        print('unsupported dataset:', dataset_name)

def eval_due(dataset_name, pred_path, gt_path):
    metrics = dataset2metrics(dataset_name)
    preds = read_jsonl(pred_path)
    gts = read_jsonl(gt_path)
    print('pred %d, gt %d' % (len(preds), len(gts)))
    for metric in metrics:
        evaluator = DueEvaluator(reference=gts,
                                answers=preds,
                                ignore_case=True,
                                metric=metric)
        general_scorer, label_scorers = evaluator._evalute()
        ic('Overall %s:%.4f' % (metric, general_scorer.score()))
        """for label, scorer in label_scorers.items():
             print('%s %s:%.4f' % (label, metric, scorer.score()))"""

def add_tabfact_missing_img(due_preds):
    ref_path = DUE_BENCHMARK + 'TabFact/test/document.jsonl'
    new_due_preds = []
    i = -1
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            i+=1
            if due_preds[i]['name'] == line['name']: 
                """# copy raw statement from anno file, avoid small revisions
                img = {'name':line['name'], 'annotations':[]}
                for i, anno in enumerate(line['annotations']):
                    pred_value = due_preds[i]['annotations']['values'][0]['value']
                    img['annotations'].append({'key':anno['key'], 'values':[{'value':pred_value}]})
                new_due_preds.append(img)"""
                new_due_preds.append(due_preds[i])
                continue
            else:
                print('add random prediction for missing img:', line['name'])
                img = {'name':line['name'], 'annotations':[]}
                for anno in line['annotations']:
                    img['annotations'].append({'key':anno['key'], 'values':[{'value':random.choice(['0', '1'])}]})
                new_due_preds.append(img)
                i-=1

    return new_due_preds

def postprocess_llm_vqa(dataset_name, split, llm_pred_path, eval_flag=True):
    """
    reformat results by LLM for due-benchmark evaluation 

    """
    assert dataset_name in ['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepFormQA', 'KleisterCharityQA', 'TabFact']
    ic(dataset_name)
    if dataset_name == 'DeepFormQA':
        dataset_categories = ['advertiser', 'flight_from', 'flight_to', 'gross_amount', 'contract_num']
    elif dataset_name == 'KleisterCharityQA':
        dataset_categories = ['address__post_town',
                         'address__postcode',
                         'address__street_line',
                         'charity_name',
                         'charity_number',
                         'income_annually_in_british_pounds',
                         'report_date',
                         'spending_annually_in_british_pounds']
    
    preds = []
    with open(llm_pred_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            preds.append({
                            'name':line['image'][0],
                            'question': line['conversations'][1]['value'],
                            'answer':str(line['model_answer']).replace('\n', '')})

    donut_path = DUE_FOR_DONUT + dataset_name + '/' + split + '/metadata.jsonl'
    donut_data = read_jsonl(donut_path)
    ic(len(donut_data), len(preds))
    assert len(donut_data) == len(preds)
    for i in range(len(donut_data)):
        preds[i]['name'] = donut_data[i]['file_name'].split('/')[-1].split('.pdf')[0]
        # for ie task, covert category question to the category
        if dataset_name in ['DeepFormQA', 'KleisterCharityQA']:
            cate_question = json.loads(donut_data[i]['ground_truth'])['gt_parses'][0]['question']
            for cate in dataset_categories:
                if cate in cate_question:
                    preds[i]['question'] = cate
                    break
        # for qa task, copy question is necessary, question in preds can have some minor revisions
        # keep quesiton consistent with gt file is necessary for due eveluation
        else:
            preds[i]['question'] = json.loads(donut_data[i]['ground_truth'])['gt_parses'][0]['question']

        if dataset_name == 'TabFact':
            if preds[i]['answer'].lower() == 'true':
                preds[i]['answer'] = '1'
            else:
                assert preds[i]['answer'].lower() == 'false'
                preds[i]['answer'] = '0'
    # reorganize preds to 1 line means QA pairs or category-value pairs of 1 image
    due_preds = []
    img = {}
    for i in range(len(preds)):
        pred = preds[i]
        if 'name' not in img: # start img
            img['name'] = pred['name']
            img['annotations'] = []
        elif pred['name'] != img['name']: # save previous img results and init a new one
            due_preds.append(copy.deepcopy(img))
            img = {}
            img['name'] = pred['name']
            img['annotations'] = []

        # for ie task, if the answer is none, drop the category-value pair
        if dataset_name not in ['DeepFormQA', 'KleisterCharityQA'] or pred['answer'] != 'None':
            img['annotations'].append({'key':pred['question'], 'values':[{'value':pred['answer']}]})
        
        if i == len(preds)-1:
            due_preds.append(copy.deepcopy(img))
    if dataset_name == 'TabFact':
        due_preds = add_tabfact_missing_img(due_preds)

    save_path = llm_pred_path.replace('.jsonl', '_due.jsonl')
    save_jsonl(due_preds, save_path)

    # evaluate 
    if eval_flag:
        if dataset_name  == 'DeepFormQA':
            eval_dataset_name = 'DeepForm'
        elif dataset_name == 'KleisterCharityQA':
            eval_dataset_name = 'KleisterCharity'
        else:
            eval_dataset_name = dataset_name

        if split == 'validation':
            gt_path = DUE_BENCHMARK + eval_dataset_name +'/dev/document.jsonl'
        else:
            gt_path = DUE_BENCHMARK + eval_dataset_name +'/'+split+'/document.jsonl'
        eval_due(eval_dataset_name, save_path, gt_path)

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

def textcaps_textvqa_eval(result_path, dataset='TextVQA', split='test'):
    # question_ids_path = dataset_dir + split+'_q_ids.json'
    question_ids_path = './benchmark_files/text_vqa_test_q_ids.json'
    if dataset == 'TextVQA':
        if not os.path.exists(question_ids_path):
            # qa_path = dataset_dir + 'TextVQA_0.5.1_'+split+'.json'
            qa_path = './benchmark_files/text_vqa_0.5.1_test.json'
            raw_qa_data = json.load(open(qa_path, 'r', encoding='utf-8'))
            raw_qa_data = raw_qa_data['data']

            # collect QAs of an identical image
            print('collecting QAs......')
            img2qas = {}
            que_num = 0
            for qa in tqdm(raw_qa_data):
                if dataset == 'TextVQA':
                    imgid = qa['image_id']
                    question = qa['question']
                    q_id = qa['question_id']
                    if imgid not in img2qas:
                        img2qas[imgid] = {}
                    img2qas[imgid][question] = q_id
                    que_num+=1
            ic(que_num)
            json.dump(img2qas, open(question_ids_path, 'w', encoding='utf-8'))
            print('save question ids to ', question_ids_path)
    
        q_ids = json.load(open(question_ids_path, 'r', encoding='utf-8'))

    # load LLM result
    llm_results = []
    if dataset == 'TextVQA':
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                img = line['image'][0]
                imgid = img.split('/')[-1].replace('.jpg', '')
                question = line['conversations'][1]['value']
                if dataset == 'TextVQA':
                    q_id = q_ids[imgid][question]
                    # gt_answer = str(line['gt_answer']).replace('\n', '')
                    model_answer = str(line['model_answer']).replace('\n', '')
                    # ic(imgid, question, model_answer)
                    if model_answer.endswith('.'):
                        model_answer = model_answer[:-1]
                    llm_results.append({'question_id':q_id, 'answer':model_answer})
    else:
        img2captions = {}
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                img = line['image'][0]
                imgid = img.split('/')[-1].replace('.jpg', '')
                model_answer = str(line['model_answer']).replace('\n', '')
                # ic(imgid, model_answer)
                if imgid not in img2captions:
                    img2captions[imgid] = []
                img2captions[imgid].append(model_answer)
        
        for imgid, captions in img2captions.items():
            llm_results.append({'image_id':imgid, 'caption':random.choice(captions)})

    ic(len(llm_results))
    save_path = result_path.replace('.jsonl', '_official_eval.json')
    json.dump(llm_results, open(save_path, 'w', encoding='utf-8'))
    print('save results to ', save_path)

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
                        'CIDEr', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'RougeL', 'Meteor']
    if metric=='ExactAccuracy': # case sensitive
        score, scores = metric_calculate(targets, predictions, metric_fn=exact_match)
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

if __name__ == '__main__':
    """predictions=["abc", "abc", "Abc", "100%", "100%", "100%", "100%", "Don't"]
    targets=[["abc"], ["Abc"], ["abc"], ["96%"], ["94%"], ["0.96"], ["0.94"], ["Won't"]]
    
    ic(predictions)
    ic(targets)

    for metric in ['ExactAccuracy', 'ANLS', 'RelaxedAccuracy']:
        score, scores = doc_evaluate(metric=metric, targets=targets, predictions=predictions)
        ic(metric, score, scores)"""
    
    # predictions = ['i love china', 'hello world']
    predictions = ['i love china Beijing', 'hello world python']
    targets = [['I love China Beijing and Shanghai'],  ['hello world, hello python.']]
    ic(predictions, targets)
    for metric in ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'RougeL', 'Meteor', 'CIDEr']:
        score, scores = doc_evaluate(metric=metric, targets=targets, predictions=predictions)
        ic(metric, score, scores)



def llm_answer_eval(metric_names=['ContainAccuracy'], result_path='', save_each_eval=True):
    ic(result_path)
    gts = []
    preds = []
    imgs = []
    ques = []
    missed = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            img = line['image'][0]
            question = line['conversations'][1]['value']
            # gt_answer = str(line['gt_answer']).replace('\n', '')
            gt_answer = str(line['conversations'][2]['value']).replace('\n', '')
            model_answer = str(line['model_answer']).replace('\n', '')
            if gt_answer.endswith('.'):
                gt_answer = gt_answer[:-1]
            
            model_answer = model_answer.replace('<Generate Error>', '')
            if 'Error>' in model_answer:
                # print('====answer miss:', img)
                missed.append(line)
                continue
            imgs.append(img)
            gts.append([gt_answer])
            preds.append(model_answer)
            ques.append(question)
    ic(len(gts), len(preds), len(missed))

    miss_save_path = result_path.replace('.jsonl', '_missed.jsonl')
    save_jsonl(missed, miss_save_path)

    #@ metric, eval_result = text_recognition_eval(gts=gts, preds=preds, imgs=imgs)
    metric2scores = {}
    for metric_name in metric_names:
        score, scores = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)
        ic(metric_name, score)
        metric2scores[metric_name] = scores
     
    if save_each_eval:
        save_path = result_path.replace('.jsonl', '_metrics.jsonl')
        eval_result = []
        for i in range(len(imgs)):
            # assert len(scores) == len(imgs)
            eval_result.append({
                                'metric2score': [{'metric':metric, 'score': scores[i]} for metric, scores in metric2scores.items()],
                                'image':imgs[i], 
                                'question': ques[i],
                                'gt': gts[i][0],
                                'pred': preds[i]})
        save_jsonl(eval_result, save_path)