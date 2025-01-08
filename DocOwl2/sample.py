import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time

class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer


docowl = DocOwlInfer(ckpt_path='mPLUG/DocOwl2')

images = [
        './examples/docowl2_page0.png',
        './examples/docowl2_page1.png',
        './examples/docowl2_page2.png',
        './examples/docowl2_page3.png',
        './examples/docowl2_page4.png',
        './examples/docowl2_page5.png',
    ]

answer = docowl.inference(images, query='what is this paper about? provide detailed information.')

answer = docowl.inference(images, query='what is the third page about? provide detailed information.')