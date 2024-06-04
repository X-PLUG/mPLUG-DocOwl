import os
import json
import argparse
import pandas as pd
from collections import defaultdict
from tinychart.eval.eval_metric import chartqa_evaluator, chartqapot_evaluator
from tinychart.eval.eval_metric import chartqa_oracle_merger_evaluator, chartqa_rule_merger_evaluator
from tinychart.eval.eval_chart2text import chart2text_evaluator
from tinychart.eval.eval_chart2table import chart2table_evaluator

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
    parser.add_argument('--input', default='./output/')

    args = parser.parse_args()

    result_files = os.listdir(args.input)
    result_files = [f for f in result_files if f.endswith('.jsonl')]
    result_files.sort()
    direct_result, pot_result = None, None

    dataset2metric = defaultdict(float)
    for result_file in result_files:
        # print(result_file)
        dataset_name = '.'.join(result_file.split('.')[:-1])
        file = os.path.join(args.input, result_file)
        result_data = read_jsonl(file)
        if 'chartqa-' in dataset_name:
            direct_result, direct_acc = chartqa_evaluator(result_data, key='model_answer')
            write_jsonl(direct_result, file)
            dataset2metric[dataset_name] = round(direct_acc * 100, 2)
            print(f'Direct Accuracy: {direct_acc}')
        elif 'chartqagptpot-' in dataset_name or 'chartqatemplatepot-' in dataset_name:
            pot_result, pot_acc, error_rate = chartqapot_evaluator(result_data)
            write_jsonl(pot_result, file)
            dataset2metric[dataset_name] = round(pot_acc * 100, 2)
            print(f'PoT Accuracy: {pot_acc}')
            print(f'PoT Error Rate: {error_rate}')
        elif 'chart2text' in dataset_name:
            metric = chart2text_evaluator(result_data, temp_dir=args.input)
            dataset2metric[dataset_name] = metric
            print(f'{dataset_name} bleu: {metric}')
        elif 'opencqa-absqa' in dataset_name:
            metric = chart2text_evaluator(result_data, temp_dir=args.input)
            dataset2metric[dataset_name] = metric
            print(f'{dataset_name} bleu: {metric}')
        elif 'chartqa2table-' in dataset_name:
            metric = chart2table_evaluator(result_data)
            dataset2metric[dataset_name] = metric
            print(f'{dataset_name} F1: {metric}')
        else:
            print(f'Skip unknown dataset: {result_file}')

    if direct_result is not None and pot_result is not None:
        print("Calculate merging direct and pot results with simple divider")
        oracle_results, oracle_acc = chartqa_oracle_merger_evaluator(direct_result, pot_result)
        dataset2metric['merged-oracle'] = round(oracle_acc * 100, 2)
        print(f'Oracle Merged Accuracy: {oracle_acc}')
        write_jsonl(oracle_results, os.path.join(args.input, 'merged-oracle.jsonl'))
        rule_results, rule_acc = chartqa_rule_merger_evaluator(direct_result, pot_result)
        dataset2metric['merged-rule'] = round(rule_acc * 100, 2)
        print(f'Rule Merged Accuracy: {rule_acc}')
        write_jsonl(rule_results, os.path.join(args.input, 'merged-rule.jsonl'))
    
    # save metrics into tsv with key as the first row
    df = pd.DataFrame(dataset2metric, index=[0])
    # if there is a metrics.tsv exists, add one in the name to avoid overwrite
    tsv_name = os.path.join(args.input, 'metrics.tsv')
    if os.path.exists(tsv_name):
        # avoid overwrite. if there is metrics.1.tsv, name it metrics.2.tsv...
        i = 1
        tsv_name = os.path.join(args.input, f'metrics.{i}.tsv')
        while os.path.exists(tsv_name):
            i += 1
            tsv_name = os.path.join(args.input, f'metrics.{i}.tsv')
    df.to_csv(tsv_name, sep='\t', index=False)
    print(f'Metrics saved at: {tsv_name}')
    print(df)
