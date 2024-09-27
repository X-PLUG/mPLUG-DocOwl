import json
from icecream import ic
import jsonlines
import copy
import random
import os
from due_evaluator.due_evaluator import DueEvaluator


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

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
      for line in jsonlines.Reader(f):
        data.append(line)
    return data

def save_jsonl(data, path):
    with open(path,'w')as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) +'\n')
    print('save %d samples(imgs) to %s ' % (len(data), path))



def add_tabfact_missing_img(due_preds, meta_dir):
    ref_path = meta_dir + 'TabFact/test/document.jsonl'
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


def llm_duebenchmark_eval(dataset_name, split, llm_pred_path, meta_dir):
    """
    reformat results by LLM for due-benchmark evaluation 

    """
    assert dataset_name in ['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepForm', 'KleisterCharity', 'TabFact']
    ic(dataset_name)
    if dataset_name == 'DeepForm':
        dataset_categories = ['advertiser', 'flight_from', 'flight_to', 'gross_amount', 'contract_num']
    elif dataset_name == 'KleisterCharity':
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
            assert len(line['messages']) == 2
            assert line['messages'][0]['role'] == 'user'
            question = line['messages'][0]['content'].replace('<|image|>', '')
            preds.append({
                            'name':line['image'][0],
                            'question': question,
                            'answer':str(line['model_answer']).strip().replace('\n', '')})

    meta_path = os.path.join(meta_dir, dataset_name, split, 'metadata.jsonl')
    meta_data = read_jsonl(meta_path)
    ic(len(meta_data), len(preds))
    assert len(meta_data) == len(preds)
    for i in range(len(meta_data)):
        preds[i]['name'] = meta_data[i]['file_name'].split('/')[-1].split('.pdf')[0]
        # for ie task, covert category question to the category
        if dataset_name in ['DeepForm', 'KleisterCharity']:
            cate_question = json.loads(meta_data[i]['ground_truth'])['gt_parses'][0]['question']
            for cate in dataset_categories:
                if cate in cate_question:
                    preds[i]['question'] = cate
                    break
        # for qa task, copy question is necessary, question in preds can have some minor revisions
        # keep quesiton consistent with gt file is necessary for due eveluation
        else:
            preds[i]['question'] = json.loads(meta_data[i]['ground_truth'])['gt_parses'][0]['question']

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
        if dataset_name not in ['DeepForm', 'KleisterCharity'] or pred['answer'] != 'None':
            img['annotations'].append({'key':pred['question'], 'values':[{'value':pred['answer']}]})
        
        if i == len(preds)-1:
            due_preds.append(copy.deepcopy(img))
    if dataset_name == 'TabFact':
        due_preds = add_tabfact_missing_img(due_preds, meta_dir)

    save_path = llm_pred_path.replace('.jsonl', '_due.jsonl')
    save_jsonl(due_preds, save_path)

    gt_path = os.path.join(meta_dir, dataset_name, split, 'document.jsonl')
    eval_due(dataset_name, save_path, gt_path)

