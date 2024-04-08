"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from mplug_docowl.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)

from mplug_docowl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,WORKER_HEART_BEAT_INTERVAL
from mplug_docowl.conversation import conv_templates, SeparatorStyle
from mplug_docowl.model.builder import load_pretrained_model
from mplug_docowl.mm_utils import load_image_from_base64, process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_docowl.processor import DocProcessor


from transformers import TextIteratorStreamer
from threading import Thread
from icecream import ic


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

        
class ModelWorker:
    def __init__(self, 
                 model_path, model_base, model_name,
                 resolution, anchors, add_global_img,
                 load_8bit, load_4bit, device):

        if model_path.endswith("/"):
            model_path = model_path[:-1]
        
        self.model_name = get_model_name_from_path(model_path)

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")

        self.tokenizer, self.model, _, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        
        self.resolution=resolution
        self.token_num_each_img = (self.resolution/14)*(self.resolution/14)/self.model.get_model().vision2text.conv_patch
        self.doc_image_processor = DocProcessor(image_size=resolution, anchors=anchors, add_global_img=add_global_img, add_textual_crop_indicator=True)
       

        self.is_multimodal = True


    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                
                images = [load_image_from_base64(image) for image in images]
                # docowl only support 1 image, so only keep the last image
                image = images[-1]
                assert prompt.count(DEFAULT_IMAGE_TOKEN) == 1

                images, patch_positions, prompt = self.doc_image_processor(images=image, query=prompt)
                images = images.to(self.model.device, dtype=torch.float16)
                patch_positions = patch_positions.to(self.model.device)

                replace_token = DEFAULT_IMAGE_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                num_image_tokens = prompt.count(replace_token) * (self.token_num_each_img+1)
            else:
                images = None
                patch_positions = None
            image_args = {"images": images, "patch_positions":patch_positions}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        # max_context_length = getattr(model.config, 'max_position_embeddings', 4096)
        max_context_length = 4096
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        # do_sample = True if temperature > 0.001 else False
        do_sample = False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        ic(max_context_length, input_ids.shape[-1], num_image_tokens, max_new_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode()
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            # top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            # yield json.dumps({"text": generated_text, "error_code": 0}).encode()
            # replace < >  to [ ] to avoide <doc>,<md>,<ocr>,<bbox> are removed by web code
            yield json.dumps({"text": generated_text.replace('<','[').replace('>',']'), "error_code": 0}).encode()
            


    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
