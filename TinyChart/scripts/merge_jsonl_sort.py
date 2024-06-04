import os
import json
import argparse

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
    parser.add_argument('--input', default='temp/')
    parser.add_argument('--output', default='chartqa_val.json')
    
    args = parser.parse_args()
    files = os.listdir(args.input)
    files.sort()
    data = []
    for file in files:
        if file != 'all.jsonl':
            data.extend(read_jsonl(os.path.join(args.input, file)))
    # data.sort(key=lambda x: int(x['id'].split('_')[-1]))
    write_jsonl(data, args.output)
