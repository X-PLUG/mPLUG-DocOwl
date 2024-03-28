import jsonlines
import json
from icecream import ic
import re
from evaluator import doc_evaluate
import os
from tqdm import tqdm
import random
from pathlib import Path

def parser_line(line):
    image = line['image'][0]
    assert len(line['messages']) == 2
    assert line['messages'][0]['role'] == 'user'
    question = line['messages'][0]['content'].replace('<|image|>', '')
    
    predicted_answer = line['model_answer'].replace('\n', '').strip()
    gt_answer = line['gt_answer'].replace('\n', '').strip()

    return image, question, predicted_answer, gt_answer


def parser_ground_line(line):
    task_name = line['task_name'] # e.g. paragraph_bbox2t_sft
    obj=task_name.split('_')[0]
    image = line['image'][0]
    assert 'messages' in line
    assert len(line['messages']) == 2

    assert line['messages'][0]['role'] == 'user'
    question = line['messages'][0]['content'].replace('<|image|>', '')

    task_name = line['task_name']
    if 't2bbox' in task_name:
        gt_answer = line['gt_answer'].strip().replace('<bbox>', '').replace('</bbox>','')
        gt_answer = [max(min(int(x)/999, 1.0), 0.0) for x in gt_answer.split(',')]

        model_answer = line['model_answer'].strip().replace('<bbox>', '').replace('</bbox>','')
        try:
            model_answer = [max(min(int(x)/999, 1.0), 0.0) for x in model_answer.split(',')]
        except Exception as e:
            model_answer = [0.0,0.0,0.0,0.0]
        try:
            assert len(model_answer) == 4
        except AssertionError as e:
            # ic(line)
            model_answer = [0.0,0.0,0.0,0.0]
            # exit(0)
    else:
        assert 'bbox2t' in task_name
        model_answer = line['model_answer'].strip().replace('<ocr>', '').replace('</ocr>','')
        model_answer = model_answer.strip()

        gt_answer = line['gt_answer'].strip().replace('<ocr>', '').replace('</ocr>','')
        gt_answer = gt_answer.strip()

    return image, question, model_answer, gt_answer, obj


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))
    print('save %d samples to %s' % (len(data), filename))


def llm_benchmark_eval(metric_names=['ContainAccuracy'], result_path='', save_each_eval=True):
    if not Path(result_path).exists():
        ic('not exists',result_path)
        return
    ic(result_path)
    gts = []
    preds = []
    imgs = []
    ques = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            img, question, model_answer, gt_answer = parser_line(line)
            if gt_answer.endswith('.'):
                gt_answer = gt_answer[:-1]

            imgs.append(img)
            gts.append([gt_answer])
            preds.append(model_answer)
            ques.append(question)

    ic(len(gts), len(preds))

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

def llm_text_localization_eval(metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4'], result_path='', save_each_eval=True):
    if not Path(result_path).exists():
        ic('not exists',result_path)
        return
    ic(result_path)

    gts = []
    preds = []
    imgs = []
    ques = []
    objs = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            img, question, model_answer, gt_answer, obj = parser_ground_line(line)
            # model_answer = model_answer.strip()
            if isinstance(gt_answer, str) and isinstance(model_answer, str):
                if gt_answer.endswith('.'):
                    gt_answer = gt_answer[:-1]
                
            imgs.append(img)
            gts.append([gt_answer])
            preds.append(model_answer)
            ques.append(question)
            objs.append(obj)

    ic(len(gts), len(preds))
    metric2scores = {}
    metric2score = {}
    for metric_name in metric_names:
        score, scores = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)
        # ic(metric_name, score)
        metric2scores[metric_name] = scores
        metric2score[metric_name]=str(round(score,2))
    
    # calculate metric of each type of object (word, phrase, line, paragraph)
    obj2metrics = {}
    for metric_name in metric_names:
        scores = metric2scores[metric_name]
        obj2scores = {}
        for i, obj in enumerate(objs):
            score = scores[i]
            if obj not in obj2scores:
                obj2scores[obj] = []
            obj2scores[obj].append(score)
        for obj, scores in obj2scores.items():
            num=len(scores)
            if metric_name == 'IOU@0.5':
                score = round(100*sum(scores)/len(scores), 2)
            else:
                score = round(sum(scores)/len(scores), 2)
            # ic(metric_name, obj, num, score)
            
            if obj == 'word' and metric_name in ['BLEU2', 'BLEU3', 'BLEU4']:
                continue
            if obj == 'phrase' and metric_name in ['BLEU1', 'BLEU3', 'BLEU4']:
                continue
            if obj == 'line' and metric_name in ['BLEU1', 'BLEU2', 'BLEU4']:
                continue
            if obj == 'paragraph' and metric_name in ['BLEU1', 'BLEU2', 'BLEU3']:
                continue
            obj2metrics[obj+'_'+metric_name] = score
        # print('---------------------------')
    ic(obj2metrics)

    if 'BLEU1' in metric_names: # recognition evaluation
        ave = round(sum(obj2metrics.values())/len(obj2metrics.values()), 2)
        ic(ave)
    else: # grounding evaluation
        ave = metric2score['IOU@0.5']
        ic(ave)
     
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

def llm_textcaps_textvqa_eval(result_path, dataset='TextVQA', split='test', meta_dir=''):
    if dataset == 'TextVQA':
        question_ids_path = os.path.join(meta_dir, dataset,  split+'_q_ids.json')
        if not os.path.exists(question_ids_path):
            qa_path = os.path.join(meta_dir, dataset, 'TextVQA_0.5.1_'+split+'.json')
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

        llm_results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                img = line['image'][0]
                imgid = img.split('/')[-1].replace('.jpg', '')
                assert line['messages'][0]['role'] == 'user'
                question = line['messages'][0]['content'].replace('<|image|>', '')
                if dataset == 'TextVQA':
                    q_id = q_ids[imgid][question]
                    # gt_answer = str(line['gt_answer']).replace('\n', '')
                    model_answer = str(line['model_answer'].strip()).replace('\n', '')
                    # ic(imgid, question, model_answer)
                    if model_answer.endswith('.'):
                        model_answer = model_answer[:-1]
                    llm_results.append({'question_id':q_id, 'answer':model_answer})
    else:
        llm_results = []
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
    print('save LLM predictions in the official format to ', save_path)
    if split == 'test':
        print('!!!!!! upload this file to official website for evaluation !!!!!')

    
