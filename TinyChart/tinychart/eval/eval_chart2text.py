import os
import re
import sys
import sacrebleu
import numpy as np

def chart2text_evaluator(data, temp_dir='/output/temp'):
    if temp_dir[-1] == '/':
        temp_dir = temp_dir[:-1]
    cands = []
    refs = []
    for item in data:
        cands.append(item['model_answer'])
        refs.append(item['gt_answer'])

    bleu = sacrebleu.corpus_bleu(cands, [refs], lowercase=True).score

    return bleu