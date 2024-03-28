import torch
from PIL import Image
from transformers import TextStreamer
import os

from mplug_docowl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_docowl.conversation import conv_templates, SeparatorStyle
from mplug_docowl.model.builder import load_pretrained_model
from mplug_docowl.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_docowl.processor import DocProcessor
from icecream import ic
import time


class DocOwlInfer():
    def __init__(self, ckpt_path, anchors='grid_9', add_global_img=True, load_8bit=False, load_4bit=False):
        model_name = get_model_name_from_path(ckpt_path)
        ic(model_name)
        self.tokenizer, self.model, _, _ = load_pretrained_model(ckpt_path, None, model_name, load_8bit=load_8bit, load_4bit=load_4bit, device="cuda")
        self.doc_image_processor = DocProcessor(image_size=448, anchors=anchors, add_global_img=add_global_img, add_textual_crop_indicator=True)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def inference(self, image, query):
        image_tensor, patch_positions, text = self.doc_image_processor(images=image, query='<|image|>'+query)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        patch_positions = patch_positions.to(self.model.device)

        # ic(image_tensor.shape, patch_positions.shape, text)

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles # ("USER", "ASSISTANT")

        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ic(prompt)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        
        # ic(input_ids)

        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                streamer=self.streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        return outputs.replace('</s>', '')



if __name__ == '__main__':

    ckpt_dir = './'
    model_dir = 'DocOwl1.5-Chat'
    model_path = os.path.join(ckpt_dir, model_dir)
    docowl = DocOwlInfer(ckpt_path=model_path, anchors='grid_9', add_global_img=True)
    print('load model from ', model_path)
    # exit(0)

    qas = [
        # docvqa case
        {"image_path":"./DocDownstream-1.0/imgs/DUE_Benchmark/DocVQA/pngs/rnbx0223_193.png", 
        "question":"What is the Compound Annual Growth Rate (CAGR) for total assets?"},
        {"image_path":"./DocDownstream-1.0/imgs/DUE_Benchmark/DocVQA/pngs/rnbx0223_193.png", 
        "question":"What is the Compound Annual Growth Rate (CAGR) for net worth?"},
    ]
    

    for qa in qas:
        image= qa['image_path'] 
        query = qa['question']
        start_time = time.time()
        ## give relatively longer answer
        answer = docowl.inference(image, query)
        end_time = time.time()
        cost_seconds = end_time-start_time

        ## answer with detailed explanation
        # query = qa['question']+'Answer the question with detailed explanation.'
        # answer = docowl.inference(image, query)
        ic(image)
        ic(query, answer)
        ic(cost_seconds)
        # ic(query_simple, answer_simple)
        print('==================')


