from .tools import llm_answer_eval, postprocess_llm_vqa, textcaps_textvqa_eval

if __name__ == '__main__':

    llm_answer_eval(metric_names=['RelaxedAccuracy'], result_path='evaluate_results/test_ChartQA.jsonl', save_each_eval=True)
    llm_answer_eval(metric_names=['ExactAccuracy'], result_path='evaluate_results/test_TabFact.jsonl', save_each_eval=True)
    llm_answer_eval(metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'Meteor', 'RougeL', 'CIDEr'], result_path='evaluate_results/test_VisualMRC.jsonl', save_each_eval=True)


    postprocess_llm_vqa(dataset_name='DeepFormQA', split='test',
                        llm_pred_path='./evaluate_results/test_DeepForm.jsonl',
                         eval_flag=True)
    postprocess_llm_vqa(dataset_name='DocVQA', split='test',
                            llm_pred_path='./evaluate_results/test_DocVQA.jsonl',
                            eval_flag=True)
    postprocess_llm_vqa(dataset_name='InfographicsVQA', split='test',
                            llm_pred_path='evaluate_results/test_InfographicsVQA.jsonl',
                            eval_flag=True)
    postprocess_llm_vqa(dataset_name='KleisterCharityQA', split='test',
                        llm_pred_path='evaluate_results/test_KleisterCharity.jsonl',
                         eval_flag=True)
    postprocess_llm_vqa(dataset_name='WikiTableQuestions', split='test',
                        llm_pred_path='evaluate_results/test_WikiTableQuestions.jsonl',
                         eval_flag=True)

    # need to submit evaluate_results/***_official_eval.json
    textcaps_textvqa_eval(result_path='evaluate_results/test_TextCaps.jsonl', dataset='TextCaps', split='test')
    textcaps_textvqa_eval(result_path='evaluate_results/test_TextVQA.jsonl', dataset='TextVQA', split='test')




