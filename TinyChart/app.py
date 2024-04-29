import argparse
import hashlib
import json
import os
import time
from threading import Thread
import logging
import gradio as gr
import torch

from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import (
    KeywordsStoppingCriteria,
    load_image_from_base64,
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from io import BytesIO
import base64
import torch
from transformers import StoppingCriteria
from tinychart.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tinychart.conversation import SeparatorStyle, conv_templates, default_conversation
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds

from transformers import TextIteratorStreamer
from pathlib import Path

DEFAULT_MODEL_PATH = "mPLUG/TinyChart-3B-768"
DEFAULT_MODEL_NAME = "TinyChart-3B-768"


block_css = """

#buttons button {
    min-width: min(120px,100%);
}
"""
title_markdown = """
# TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning
🔗 [[Code](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart)] | 📚 [[Paper](https://arxiv.org/abs/2404.16635)]

**Note:** 
1. Currently, this demo only supports English chart understanding and may not work well with other languages.
2. To use Program-of-Thoughts answer, please append "Answer with detailed steps." to your question.
"""
tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""

def regenerate(state, image_process_mode):
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def clear_history():
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None)


def add_text(state, text, image, image_process_mode):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None)

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            # text = text + "\n<image>"
            text = "<image>\n"+text
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def load_demo():
    state = default_conversation.copy()
    return state

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


@torch.inference_mode()
def get_response(params):
    prompt = params["prompt"]
    ori_prompt = prompt
    images = params.get("images", None)
    num_image_tokens = 0
    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError(
                    "Number of images does not match number of <image> tokens in prompt"
                )

            images = [load_image_from_base64(image) for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [
                    image.to(model.device, dtype=torch.float16) for image in images
                ]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, "mm_use_im_start_end", False):
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            if hasattr(model.get_vision_tower().config, "tome_r"):
                num_image_tokens = (
                    prompt.count(replace_token) * model.get_vision_tower().num_patches - 26 * model.get_vision_tower().config.tome_r
                )
            else:
                num_image_tokens = (
                    prompt.count(replace_token) * model.get_vision_tower().num_patches
                )
        else:
            images = None
        image_args = {"images": images}
    else:
        images = None
        image_args = {}

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False
    logger.info(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    keywords = [stop_str]

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
    )

    max_new_tokens = min(
        max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
    )

    if max_new_tokens < 1:
        yield json.dumps(
            {
                "text": ori_prompt
                + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0,
            }
        ).encode() + b"\0"
        return

    # local inference
    # BUG: If stopping_criteria is set, an error occur: 
    # RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
    generate_kwargs = dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        # stopping_criteria=[stopping_criteria],
        use_cache=True,
        **image_args,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    logger.debug(ori_prompt)
    logger.debug(generate_kwargs)
    generated_text = ori_prompt
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[: -len(stop_str)]
        yield json.dumps({"text": generated_text, "error_code": 0}).encode()

    if '<step>' in generated_text and '</step>' in generated_text and '<comment>' in generated_text and '</comment>' in generated_text:
        program = generated_text
        program = '<comment>#' + program.split('ASSISTANT: <comment>#')[-1]
        print(program)
        try:
            execuate_result = evaluate_cmds(parse_model_output(program))
            if is_float(execuate_result):
                execuate_result = round(float(execuate_result), 4)
            generated_text += f'\n\nExecute result: {execuate_result}'
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
        except:
            generated_text += f'\n\nIt seems the execution of the above code encounters bugs. I\'m trying to answer this question directly...'
            ori_generated_text = generated_text + '\nDirect Answer: '

            direct_prompt = ori_prompt.replace(' Answer with detailed steps.', '')
            direct_input_ids = (
                tokenizer_image_token(direct_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )

            generate_kwargs = dict(
                inputs=direct_input_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                **image_args,
            )
            thread = Thread(target=model.generate, kwargs=generate_kwargs)
            thread.start()
            generated_text = ori_generated_text
            for new_text in streamer:
                generated_text += new_text
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[: -len(stop_str)]
                yield json.dumps({"text": generated_text, "error_code": 0}).encode()
                
        

def http_bot(state, temperature, top_p, max_new_tokens):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot())
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation

        template_name = 'phi'

        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]

    # Make requests
    # pload = {"model": model_name, "prompt": prompt, "temperature": float(temperature), "top_p": float(top_p),
    #          "max_new_tokens": min(int(max_new_tokens), 1536), "stop": (
    #         state.sep
    #         if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
    #         else state.sep2
    #     ), "images": state.get_images()}

    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": (
            state.sep
            if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
            else state.sep2
        ), "images": state.get_images()}

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot())

    # for stream
    output = get_response(pload)
    for chunk in output:
        if chunk:
            data = json.loads(chunk.decode().replace('\x00',''))

            if data["error_code"] == 0:
                output = data["text"][len(prompt) :].strip()
                state.messages[-1][-1] = output + "▌"
                yield (state, state.to_gradio_chatbot())
            else:
                output = data["text"] + f" (error_code: {data['error_code']})"
                state.messages[-1][-1] = output
                yield (state, state.to_gradio_chatbot())
                return
            time.sleep(0.03)

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot())


def build_demo():
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title="TinyChart", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row(elem_id="Model ID"):
                    gr.Dropdown(
                        choices=[DEFAULT_MODEL_NAME],
                        value=DEFAULT_MODEL_NAME,
                        interactive=True,
                        label="Model ID",
                        container=False,
                    )
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                cur_dir = Path(__file__).parent
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/images/market.png",
                            "What is the highest number of companies in the domestic market? Answer with detailed steps.",
                        ],
                        [
                            f"{cur_dir}/images/college.png",
                            "What is the difference between Asians and Whites degree distribution? Answer with detailed steps."
                        ],
                        [
                            f"{cur_dir}/images/immigrants.png",
                            "How many immigrants are there in 1931?",
                        ],
                        [
                            f"{cur_dir}/images/sails.png",
                            "By how much percentage wholesale is less than retail? Answer with detailed steps."
                        ],
                        [
                            f"{cur_dir}/images/diseases.png",
                            "Is the median value of all the bars greater than 30? Answer with detailed steps.",
                        ],
                        [
                            f"{cur_dir}/images/economy.png",
                            "Which team has higher economy in 28 min?"
                        ],
                        [
                            f"{cur_dir}/images/workers.png",
                            "Generate underlying data table for the chart."
                        ],
                        [
                            f"{cur_dir}/images/sports.png",
                            "Create a brief summarization or extract key insights based on the chart image."
                        ],
                        [
                            f"{cur_dir}/images/albums.png",
                            "Redraw the chart with Python code."
                        ]
                    ],
                    inputs=[imagebox, textbox],
                )

                with gr.Accordion("Parameters", open=False) as _:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=1024,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as _:
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=True)

        gr.Markdown(tos_markdown)

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        clear_btn.click(
            clear_history, None, [state, chatbot, textbox, imagebox], queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        demo.load(load_demo, None, [state], queue=False)
    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", default=None)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(gr.__version__)
    args = parse_args()
    model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit
    )

    demo = build_demo()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
