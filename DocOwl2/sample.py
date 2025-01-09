import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
import subprocess

def download_imgs(download_dir='examples'):
    os.makedirs(download_dir, exist_ok=True)
    # download only if it is not existed.
    urls = [
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page0.png?download=true",
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page1.png?download=true",
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page2.png?download=true",
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page3.png?download=true",
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page4.png?download=true",
        "https://huggingface.co/mPLUG/DocOwl2/resolve/main/examples/docowl2_page5.png?download=true"
    ]
    for i, url in enumerate(urls):
        file_path = f"{download_dir}/docowl2_page{i}.png"
        if not os.path.exists(file_path):
            subprocess.run(["wget", "-O", file_path, url])
            
class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer

current_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = f'{current_dir}/examples'
download_imgs(download_dir)
images = [
        f'{download_dir}/docowl2_page0.png',
        # f'{download_dir}/docowl2_page1.png', # to avoid cuda out of memory, I commented out these.
        # f'{download_dir}/docowl2_page2.png',
        # f'{download_dir}/docowl2_page3.png',
        # f'{download_dir}/docowl2_page4.png',
        # f'{download_dir}/docowl2_page5.png'
    ]

# Free GPU memory
torch.cuda.empty_cache()

docowl = DocOwlInfer(ckpt_path='mPLUG/DocOwl2')

answer = docowl.inference(images, query='what is this paper about? provide detailed information.')
torch.cuda.empty_cache()

# answer = docowl.inference(images, query='what is the third page about? provide detailed information.')
# torch.cuda.empty_cache()