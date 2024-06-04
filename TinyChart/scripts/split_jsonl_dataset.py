import json
import os
import argparse
from collections import defaultdict

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(data, jsonl_path):
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='all.json')
    parser.add_argument('--output', default='./output/')
    
    args = parser.parse_args()

    all_data = read_jsonl(args.input)

    dataset2jsonl = defaultdict(list)

    for item in all_data:
        int_id = item['id'].split('_')[-1]
        dataset_name_split = '_'.join(item['id'].split('_')[:-1])
        
        if '-two_col-' in dataset_name_split:
            dataset_name_split = dataset_name_split.replace('-two_col-', '-')
        if '-multi_col-' in dataset_name_split:
            dataset_name_split = dataset_name_split.replace('-multi_col-', '-')
        
        dataset2jsonl[dataset_name_split].append(item)

    for dataset_name_split, data in dataset2jsonl.items():
        data.sort(key=lambda x: int(x['id'].split('_')[-1]))
        write_jsonl(data, os.path.join(args.output, f'{dataset_name_split}.jsonl'))