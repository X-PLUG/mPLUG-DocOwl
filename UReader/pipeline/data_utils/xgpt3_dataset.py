from pathlib import Path
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
import json
import os
import logging
import random
import re
import time
import traceback
import warnings
from io import BytesIO

import h5py
import numpy as np
import torch
from icecream import ic
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset

from pipeline.utils import get_args

from .processors.builder import build_processors
from datasets import load_dataset
import requests
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import traceback
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


from PIL import Image, ImageFile, ImageSequence
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
import json
import os 
import logging
import random
import time
from PIL import Image, ImageFile, ImageSequence
from io import BytesIO
import re
from icecream import ic
from datasets import load_dataset
import requests

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
import traceback
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def read_image_fn(fp, page_id = None):
    if page_id is not None:
        #image=list(ImageSequence.all_frames(Image.open(fp)))[page_id]
        image = Image.open(fp)
        image.seek(page_id)
        image = image.copy()
    else:
        image = Image.open(fp)
    image = image.convert('RGB')
    return image

def base64decode(s: str):
    import base64
    """
    Decode base64 `str` to original `bytes`.
    If the input is not a valid base64 string, return None.

    Args:
        s(str): A base64 `str` that can be used in text file.

    Returns:
        Optional[bytes]: The original decoded data with type `bytes`.
            If the input is not a valid base64 string, return None.
    """
    # return base64.b64decode(s)
    _base64_regex = re.compile(r'^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{3}=|[A-Za-z\d+/]{2}==)?$')
    s = s.translate(base64._urlsafe_decode_translation)
    if not _base64_regex.fullmatch(s):
        return None
    try:
        return base64.urlsafe_b64decode(s)
    except base64.binascii.Error:
        return None

class ImageIO():
    # 单独分离出来的图像IO 为了和inference代码块同步图像读取逻辑以支持inference数据集 
    # TODO 暂时先不合并到训练Dataset中 需要进一步验证稳定性
    def __init__(self):
        pass

    def _load_img(self, images, image_root=None):
        if isinstance(images, str):
            images = [images]
        image_pils = []
        for image_url in images:
            # 支持tiff页面号指定 xxx.tiff.2
            # TODO https://github.com/Belval/pdf2image to support pdf. Now please convert pdf to png/jpg first
            url_tmp =  image_url.split('.')
            if url_tmp[-1].isdigit():
                image_url = '.'.join(url_tmp[:-1])
                page_id = int(url_tmp[-1])
            else:
                page_id = None
                
            if image_url.startswith("oss://"):
                raise NotImplementedError
            elif image_url.startswith("http://") or image_url.startswith("https://"):
                # HTTP 读取
                # We need to actually check for a real protocol, otherwise it's impossible to
                # use a local file like http_huggingface_co.png.
                image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
            elif os.path.exists(image_url):
                image  = read_image_fn(image_url, page_id=page_id)
            elif image_root is not None and Path(image_root, image_url).exists():
                image  = read_image_fn(Path(image_root, image_url), page_id=page_id)
            else:
                image_bytes = base64decode(image_url)
                if image_bytes is not None:
                    image = read_image_fn(BytesIO(image_bytes))
                else:
                    pass
            image_pils.append(image)
        return image_pils

class MultiModalDataset(Dataset):
    """MultiModal dataset"""

    def __init__(self, input_files, tokenizer, processors, max_length=2048,
                media_tokens=['<image>', '<|video|>'], image_root = 'ureader_images'):
        args = get_args()
        self.image_root = image_root 
        self.dataset = []
        if isinstance(input_files, str):
            input_files = [input_files]
        for input_file in input_files:
            self.dataset += load_jsonl(input_file)

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.processors = processors
        # current only support image
        self.media_tokens = {k: -int(i+1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<image>': 1+64}

        print("num_media_token: ", self.media_lengths)
        self.image_io = ImageIO()
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)


    def process_data(self, data, processor=None):
        # Process Image if exists
        args = get_args()
        if 'image' in data and len(data['image']) > 0:
            if 'image_data' in data:
                # 如果已经传入了图片内容 则跳过读取
                images = data['image_data']
            else:
                # 否则就读图片
                image_urls = data['image']
             
                images = self.image_io._load_img(image_urls, image_root = self.image_root)
            num_img_token = data['text'].count('<image>')
            if num_img_token!=len(images):
                # 因为在处理单图切多图时需要分裂<image> 所以这里需要确保原本数目是正确的
                warnings.warn('The number of <image> is different from the number of input images')
            
            if processor:
                if hasattr(processor,'build_text'):
                    data['text'] = processor.build_text(data=data)
                process_results = [processor(image=image, text=None) for image in images]
                if len(process_results)>0 and len(process_results[0][0].shape) == 4:
                    # 图片被切分成了多块 默认是doc场景
                    text_list = data['text'].split('<image>')
                    images = []
                    patch_positions = []
                    text = text_list[0]
                    for ri, (image_input, text_input, patch_position) in enumerate(process_results):
                        images.append(image_input)
                        patch_positions.append(patch_position)
                        if args.patch_pos_embed_type == 'pre':
                            # 对于pre处理 v2t最终输出的是一张图的token
                            text += '<image>'
                        else:
                            # 对于post处理 v2t最终输出的是多图
                            text += '<image>'*image_input.shape[0]
                        text += text_list[ri+1]
                    images = torch.cat(images, dim=0)
                    patch_positions = torch.cat(patch_positions, dim=0)
                else:
                    # 如果没有切片 则正常stack 并创建patch position = num_image (0,0)的patch id以保持一致
                    images = [_[0] for _ in process_results]
                    images = torch.stack(images, dim=0)
                    patch_positions = torch.zeros(images.shape[0],2).long()
                    text = data['text']
            else:
                raise NotImplementedError
        else:
            images = None
            patch_positions = None
            text = data['text']
        
        # Process Text
        text = {
            "prompt": data.get('prompt', ""),
            "text": text
        }
        if processor:
            text = processor(image=None, text=text)[1]
        return images, text, patch_positions

    def __getitem__(self, index):
        data = self.dataset[index]
        task_type = data.get('task_type', 'dummy_default').split('_')[-1] # Get processor type
        while True:
            try:
                image, text, patch_positions = self.process_data(
                    data, self.processors[task_type])

                text_input = self._extract_text_token_from_conversation(
                    text, self.max_length, index)
             
            except Exception as e:
                time.sleep(0.1)
                index = 0 if index == (len(self) - 1) else index + 1
                data = self.dataset[index]
                task_type = data.get('task_type', 'dummy_default').split('_')[-1]
                traceback.print_exc()
                ic()
                continue
            break
        batch_data = {
            "image": image,
            "patch_positions":patch_positions,
            "text": text_input
        }

        return batch_data

    def _extract_text_token_from_conversation(self, data, max_length, index):
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        conversation = data["completion"]
        # For Text only data
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            pattern = '|'.join(map(re.escape, ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx-1] == 'AI: ':
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[:max_length-enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[:max_length+1-enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = '|'.join(
                map(re.escape, list(self.media_tokens.keys()) + ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = prompt_chunk + \
                        self.tokenizer(chunk_str, add_special_tokens=False)[
                            'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [self.media_tokens[chunk_str]
                                          ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:

                        if chunk_strs[idx-1] == 'AI: ':
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[:max_length-enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[:max_length +
                                                        1-enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)
        
        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * (max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + 1, (index, prompt_length, enc_length, padding_length, max_length + 1)
        assert len(label_chunk) == max_length + 1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length-1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        
        return {'input_ids': enc_chunk, "prompt_length": prompt_length, 'seq_length': enc_length, 
                "non_padding_mask": non_padding_mask, 'non_media_mask': non_media_mask, 'prompt_mask': prompt_mask}