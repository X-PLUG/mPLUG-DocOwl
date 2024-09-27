"""
this evaluation script is revised based on DUDE official evaluation script
"""

import argparse
import json
import logging
import os

import numpy as np
from munkres import Munkres, make_cost_matrix
from icecream import ic
import jsonlines


question_ids_to_exclude = []

def save_json(file_path, data):
    with open(file_path, "w+") as json_file:
        json.dump(data, json_file)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def parse_answers(pred_answers):
    if len(pred_answers) == 0:
        logging.warning("Mistaken unanswerable prediction")
        pred_answers = ""

    if isinstance(pred_answers, list):
        if len(pred_answers) > 1:
            logging.warning("Mistaken list prediction, assuming first")
        pred_answers = pred_answers[0]
    return pred_answers


def get_NLS(gt_answers, pred_answers, threshold):
    values = []

    pred_answers = parse_answers(pred_answers)

    for answer in gt_answers:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred_answers.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred_answers.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < threshold:
        question_result = 0

    return question_result


def get_best_matches_hungarian_munkers(anchor_list, matching_list):

    match_dict = {}
    match_matrix = []
    for anchor_item in anchor_list:
        NLS_dict = {}
        NLS_list = []
        for matching_item in matching_list:
            NLS = get_NLS([anchor_item], matching_item, threshold=0.5)
            NLS_dict[str(matching_item) + " "] = NLS
            NLS_list.append(NLS)

        match_dict[anchor_item] = NLS_dict
        match_matrix.append(NLS_list)

    return match_dict, match_matrix


def get_NLSL(gt_list, pred_list):
    if len(gt_list) < len(pred_list):
        anchor_list, matching_list = gt_list, pred_list

    else:
        anchor_list, matching_list = pred_list, gt_list

    match_dict, cost_matrix = get_best_matches_hungarian_munkers(anchor_list, matching_list)
    num_answers = max(len(set(gt_list)), len(pred_list))

    m = Munkres()
    m_cost_matrix = make_cost_matrix(cost_matrix)
    indexes = m.compute(m_cost_matrix)
    values = [cost_matrix[row][column] for row, column in indexes]
    NLSL = np.sum(values) / num_answers

    return NLSL


def validate_data(gtFilePath, submFilePath):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """

    gtJson = json.load(open(gtFilePath, "rb"))
    submJson = json.load(open(submFilePath, "rb"))

    if "data" not in gtJson:
        raise Exception("The GT file is not valid (no data key)")

    if "dataset_name" not in gtJson:
        raise Exception("The GT file is not valid (no dataset_name key)")

    if gtJson["dataset_name"] != "NewsVideoQA":
        raise Exception("The GT file is not valid dataset_name should be DUDE Dataset")

    if isinstance(submJson, list) is False:
        raise Exception("The Det file is not valid (root item must be an array)")

    if len(submJson) != len(gtJson["data"]):
        raise Exception(
            "The Det file is not valid (invalid number of answers. Expected:"
            + str(len(gtJson["data"]))
            + " Found:"
            + str(len(submJson))
            + ")"
        )

    gtQuestions = sorted([str(r["questionId"]) for r in gtJson["data"]])
    res_id_to_index = {str(r["questionId"]): ix for ix, r in enumerate(submJson)}
    detQuestions = sorted([str(r["questionId"]) for r in submJson])

    if (gtQuestions == detQuestions) is False:
        print(len(gtQuestions), len(detQuestions))
        print(len(set(gtQuestions).intersection(detQuestions)))
        print(gtQuestions[0], detQuestions[0])
        raise Exception("The Det file is not valid. Question IDs must match GT")

    for gtObject in gtJson["data"]:

        try:
            q_id = str(gtObject["questionId"])
            res_ix = res_id_to_index[q_id]

        except:
            raise Exception(
                "The Det file is not valid. Question "
                + str(gtObject["questionId"])
                + " not present"
            )

        else:
            detObject = submJson[res_ix]

            if "answers" not in detObject:
                raise Exception(
                    "Question " + str(gtObject["questionId"]) + " not valid (no answer key)"
                )
    return gtJson, submJson


def evaluate_method(gtJson, submJson, anls_threshold=0.5):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    

    res_id_to_index = {str(r["questionId"]): ix for ix, r in enumerate(submJson)}

    perSampleMetrics = {}

    totalANLS = 0
    row = 0


    for gtObject in gtJson["data"]:
        q_id = str(gtObject["questionId"])
        res_ix = res_id_to_index[q_id]
        detObject = submJson[res_ix]

        if q_id in question_ids_to_exclude:
            question_result = 0
            info = "Question EXCLUDED from the result"

        else:
            info = ""
            
            question_result = get_NLS(
                [gtObject["answer"]], detObject["answers"], anls_threshold
            )

            totalANLS += question_result

        

        perSampleMetrics[str(gtObject["questionId"])] = {
            "anls": question_result,
            "question": gtObject["question"],
            "gt_answer": [gtObject["answer"]],
            "answer_prediction": detObject["answers"],
            "answer_confidence": detObject.get("answers_confidence", -1),
            "info": info,
        }
        row = row + 1

    methodMetrics = {
        "anls": 0
        if len(gtJson["data"]) == 0
        else totalANLS / (len(gtJson["data"]) - len(question_ids_to_exclude))
    }

    resDict = {
        "result": methodMetrics,
        "per_sample_result": perSampleMetrics,
    }
    return resDict


def display_results(results):
    print("\nOverall ANLS: {:1.4f}\n".format(results["result"]["anls"]))
    print("")

def postprocess_llm_vqa(dataset_name, split, llm_pred_path, meta_dir, save_results=True):
    """
    reformat results by LLM for NewsVideoQA official evaluation 

    """
    assert dataset_name in ['NewsVideoQA']
    ic(dataset_name)
    
    preds = []
    with open(llm_pred_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            assert 'messages' in line
            if len(line['messages']) == 3:
                assert line['messages'][2]['role'] == 'user'
                question = line['messages'][2]['content']
            elif len(line['messages']) == 2:
                assert line['messages'][0]['role'] == 'user'
                question = line['messages'][0]['content'].replace('<|image|>', '')

            preds.append({
                            'clip_id':line['image'][0].split('/')[-1].split('_')[0],
                            'question': question,
                            'answer':str(line['model_answer']).strip().replace('\n', '')
                            # 'answer': 'None'
                        })
    # gt_path
    gt_path = os.path.join(meta_dir, 'newsvideoqa_version_2_'+split+'_release.json')
    gts = json.load(open(gt_path, 'r', encoding='utf-8'))
    gts = gts['data']
    
    # align preds with gts
    ic(len(preds), len(gts))
    # assert len(preds) == len(gts)
    """
    organize preds in the following format
    [
    {
        "questionId": "xxx",
        "answers": ["Yes"],
        "answers_confidence": [1]
    },
    {
        "questionId": "xxx",
        "answers": ["2"],
        "answers_confidence": [1]
    },
    ...]
    """
    reformat_preds = []
    for i in range(len(preds)):
        llm_pred_clipid = preds[i]['clip_id']
        llm_pred_question = preds[i]['question']
        llm_pred_answer = preds[i]['answer']

        gt_questionid = gts[i]['questionId']
        gt_question = gts[i]['question']
        gt_clipid = gts[i]['uni_clipped_id']

        # ic(llm_pred_clipid, gt_clipid)
        assert llm_pred_clipid == gt_clipid
        assert llm_pred_question == gt_question
        reformat_preds.append({
                'questionId':gt_questionid,
                "answers": [llm_pred_answer],
                "answers_confidence": [1]
            })

    reformat_pred_path = llm_pred_path.replace('.jsonl', '_reformat.json')
    json.dump(reformat_preds, open(reformat_pred_path, 'w', encoding='utf-8'))

    if split == 'test':
        submissions = []
        for x in reformat_preds:
            submissions.append({'questionId':x['questionId'], 'answer':x['answers'][0]})
        submission_path = os.path.join('/'.join(reformat_pred_path.split('/')[:-1]), dataset_name+'_submission.json')
        json.dump(submissions, open(submission_path, 'w', encoding='utf-8'))
        print('save submission results to ', submission_path)
    else:
        gtJson, submJson = validate_data(gt_path, reformat_pred_path)

        # Evaluate method
        results = evaluate_method(gtJson, submJson)
        display_results(results)

        if save_results:
            save_path = llm_pred_path.replace('.jsonl', '_metrics.json')
            save_json(save_path, results)
            print("All results including per-sample result has been correctly saved to ", save_path)


if __name__ == "__main__":
    postprocess_llm_vqa(dataset_name='NewsVideoQA', split='val', 
                        llm_pred_path='/nas-alinlp/anwenhu/DocSFTv2/NewsVideoQA/val.jsonl', save_results=True)

    