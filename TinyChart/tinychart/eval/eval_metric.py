import os
import json
import os
import math
import copy
import argparse
import numpy as np

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def RelaxedAccuracy(pred, gt):
    try:
        gt = float(gt)
        pred = float(pred)
        if gt == 0.0:
            if pred == gt:
                return 1.0
            else:
                return 0.0
        else:
            if abs(pred-gt) / gt <= 0.05:
                return 1.0
            else:
                return 0.0
    except:
        if str(gt) == str(pred):
            return 1.0
        else:
            return 0.0

def evaluate_cmds(cmds):
    for cmd in cmds:
        exec(cmd)
    answer = eval('Answer')
    if (isinstance(answer, list) or isinstance(answer, np.ndarray)) and len(answer) == 1:
        answer = answer[0]
    if isinstance(answer, list) or isinstance(answer, np.ndarray):
        new_answer = answer[0]
        for i in range(1, len(answer)-1):
            new_answer = new_answer + ', ' + answer[i]
        new_answer += ' and ' + answer[-1]
        answer = new_answer
    if isinstance(answer, bool) or isinstance(answer, np.bool_):
        if answer == True:
            answer = 'Yes'
        elif answer == False:
            answer = 'No'
    return answer

def parse_model_output(cmdstr):
    lines = cmdstr.split('\n')
    new_lines = []
    for line in lines:
        if '<step>' in line or '</step>' in line:
            line = line.replace('<step>', '').replace('</step>', '')
            new_lines.append(line)
    return new_lines

def chartqa_evaluator(data, key='final_model_answer'):
    acc = 0
    for item in data:
        item['relaxed_acc'] = RelaxedAccuracy(item[key], item['gt_answer'].split('<pot_note>')[0])
        if item['relaxed_acc'] == 1.0:
            acc += 1
    accuracy = acc/len(data)
    return data, accuracy

def chartqapot_evaluator(output_data):
    correct_items = []
    wrong_items = []
    error_items = []
    output_data = copy.deepcopy(output_data)
    acc = 0
    for item in output_data:
        # cmds = parse_gpt_cmd(gpt_item['eval_cmd'])
        eval_cmds = parse_model_output(item['model_answer'])
        try:
            answer = evaluate_cmds(eval_cmds)
            item['final_model_answer'] = str(answer)
        except:
            error_items.append(item)
            item['final_model_answer'] = 'Execute <error>'
            item['relaxed_acc'] = 0.0
            continue
        item['gt_answer'] = item['gt_answer'].split('<cot_note>')[0]
        item['relaxed_acc'] = RelaxedAccuracy(str(answer), item['gt_answer'])
        
        if item['relaxed_acc'] == 1.0:
            correct_items.append(item)
        else:
            wrong_items.append(item)
    total = len(output_data)
    accuracy = len(correct_items)/total
    error_rate = len(error_items)/total
    
    return output_data, accuracy, error_rate

def rule_based_divider(question):
    calculate_words = [
        'sum', 'difference', 'times', 'summation', 'exceed', 
        'below', 'addition', 'fewer', 'subtract', ' mode ', 
        'ratio', 'division', 'average', 'mean', 'bigger', 
        'greater', ' less ', 'tallest', 'number', 'divide', 
        ' add ', 'absolute', 'dividing', 'differ', ' minus ', 
        'how many colors', 'lowest', 'what is the value', 'higher', 
        'longer', ' biggest ', 'lowest'
    ]
        
    for w in calculate_words:
        if w in question.lower():
            return 'pot'
    return 'direct'

def chartqa_rule_merger_evaluator(direct_data, pot_data):
    direct_data, _ = chartqa_evaluator(direct_data, key='model_answer')
    assert len(direct_data) == len(pot_data), 'direct and pot num inconsistent'
    acc_count = 0
    merged_data = []
    for datum1, datum2 in zip(direct_data, pot_data):
        if rule_based_divider(datum1['question']) == 'pot' and '<error>' not in datum2['final_model_answer'] and datum2['final_model_answer'] not in ['inf', '-inf', 'nan', 'np.nan', 'np.inf', '-np.inf']:
            acc_count += datum2['relaxed_acc']
            merged_data.append(datum2)
        else:
            acc_count += datum1['relaxed_acc']
            merged_data.append(datum1)
    accuracy = acc_count/len(direct_data)
    return merged_data, accuracy

def chartqa_oracle_merger_evaluator(direct_data, pot_data):
    direct_data, _ = chartqa_evaluator(direct_data, key='model_answer')   
    assert len(direct_data) == len(pot_data), 'direct and pot num inconsistent'
    acc_count = 0
    merged_data = []
    for datum1, datum2 in zip(direct_data, pot_data):
        if datum1['relaxed_acc'] != 1.0:
            acc_count += datum2['relaxed_acc']
            merged_data.append(datum2)
        else:
            acc_count += datum1['relaxed_acc']
            merged_data.append(datum1)
    accuracy = acc_count/len(direct_data)
    return merged_data, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', default='../eval_iter12000_0226/ChartQA_test_12000_pred.jsonl')
    parser.add_argument('--pot', default='../eval_iter12000_0226/ChartQA_test_pot_12000_eval.jsonl')
    parser.add_argument('--output', default='../eval_iter12000_0226/ChartQA_test_pot_12000_merged.jsonl')
    
    args = parser.parse_args()
    
    merged = oracle_merger(args.direct, args.pot)
    merged = rule_based_merger(args.direct, args.pot)
    
    write_jsonl(merged, args.output)  