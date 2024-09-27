import json
import jsonlines
from tqdm import tqdm
import os
from icecream import ic
from evaluation.benchmarks_eval import (llm_text_localization_eval, llm_textcaps_textvqa_eval,llm_benchmark_eval)
from evaluation.due_benchmarks_eval import llm_duebenchmark_eval
from evaluation.dude_eval import postprocess_llm_vqa as llm_dude_eval
from evaluation.mpdocvqa_eval import postprocess_llm_vqa as llm_mpdocvqa_eval
from evaluation.newsvideoqa_eval import postprocess_llm_vqa as llm_newsvideoqa_eval
import argparse

import torch
from transformers import AutoTokenizer, AutoModel

class DocOwl2Infer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer

def read_jsonl(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            lines.append(line)
    return lines


def save_jsonl(data, filename, print_log=True):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))
        
    if print_log:
        print('save %d samples to %s' % (len(data), filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='docowl2 benchmark evaluation')
    parser.add_argument('--model_path', type=str, help='the directory path of model')
    parser.add_argument('--dataset', type=str, choices=['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 
                                                        'DeepForm', 'KleisterCharity', 'TabFact',
                                                        'ChartQA', 'TextVQA', 'TextCaps', 'VisualMRC',
                                                        'MP-DocVQA', 'DUDE', 'NewsVideoQA'])
    parser.add_argument('--downstream_dir', type=str, help='the directory path of DocDownstream-1.0 or DocDownstream-2.0')
    parser.add_argument('--save_dir', type=str, help='the directory to save predictions of the model')
    parser.add_argument('--split', type=str, choices=['val','test'])

    args = parser.parse_args()

    model_path = args.model_path
    dataset = args.dataset
    downstream_dir = args.downstream_dir
    save_dir = args.save_dir
    split = args.split

    if dataset not in ['MP-DocVQA', 'DUDE', 'NewsVideoQA']:
        try:
            assert split == 'test'
        except Exception as e:
            print("For single-image datasets of DocDownstream 1.0 ('DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepForm', 'KleisterCharity', 'TabFact',\
            'ChartQA', 'TextVQA', 'TextCaps', 'VisualMRC'), evaluate the test set. For multi-image datasets of  DocDownstream 2.0\
            ('MP-DocVQA', 'DUDE', 'NewsVideoQA') both val and test are supported. ")
            exit(0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_path = os.path.join(downstream_dir, split, dataset+'_'+split+'.jsonl')
    save_path = os.path.join(save_dir, dataset+'_'+split+'_pred.jsonl')

    if os.path.exists(save_path):
        print(save_path+' exists, skip inference. ')
    else:
        docowl = DocOwl2Infer(ckpt_path=model_path)
        print('load model from ', model_path)
        # infer the test samples one by one
        test_samples = read_jsonl(test_path)
        infer_results = []
        for sample in tqdm(test_samples):
            images = []
            for img in sample['image']:
                image = os.path.join(downstream_dir, img)
                assert os.path.exists(image)
                images.append(image)
            question = sample['messages'][0]
            answer = sample['messages'][1]
            assert question['role'] == 'user'
            assert answer['role'] == 'assistant'
            query = question['content'].replace('<|image|>', '')
            gt_answer = answer['content']
            model_answer = docowl.inference(images, query)
            sample['model_answer'] = model_answer
            sample['gt_answer'] = gt_answer
            ic(model_answer, gt_answer)
            infer_results.append(sample)
        save_jsonl(infer_results, save_path)
    
    # calculate metrics
    pred_path = save_path

    if not os.path.exists(pred_path):
        print('not exists:', pred_path)
        exit(0)
    
    meta_dir = os.path.join(downstream_dir, 'meta')

    if dataset in ['DeepForm', 'DocVQA', 'InfographicsVQA', 'KleisterCharity', 'WikiTableQuestions']:
        llm_duebenchmark_eval(dataset_name=dataset, split='test', llm_pred_path=pred_path, meta_dir=meta_dir)
    elif dataset in ['TabFact']:
        llm_benchmark_eval(metric_names=['ExactAccuracy'], result_path=pred_path, save_each_eval=True)
    elif dataset in ['ChartQA']:
        llm_benchmark_eval(metric_names=['RelaxedAccuracy'], result_path=pred_path, save_each_eval=True)
    elif dataset in ['TextCaps', 'TextVQA']:
        llm_textcaps_textvqa_eval(result_path=pred_path, dataset=dataset, split='test', meta_dir=meta_dir)
    elif dataset in ['VisualMRC']:
        llm_benchmark_eval(metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'Meteor', 'RougeL', 'CIDEr'], result_path=pred_path, save_each_eval=True)
    elif dataset in ['MP-DocVQA']:
        llm_mpdocvqa_eval(dataset_name=dataset, split=split, llm_pred_path=pred_path, meta_dir=os.path.join(meta_dir, dataset))
    elif dataset in ['DUDE']:
        llm_dude_eval(dataset_name=dataset, split=split, llm_pred_path=pred_path, meta_dir=os.path.join(meta_dir, dataset))
    elif dataset in ['NewsVideoQA']:
        llm_newsvideoqa_eval(dataset_name=dataset, split=split, llm_pred_path=pred_path, meta_dir=os.path.join(meta_dir, dataset))

    print('==============================================')
    



        

