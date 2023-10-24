import os
# import wget
# resources = os.getenv('resources_new')
# resources_filename = wget.download(resources)

# os.system('tar zxvf {}'.format(resources_filename))

# os.system('ls -l')

import argparse
import datetime
import json
import os
import time
import torch

import gradio as gr
import requests
from pipeline.utils import add_config_args, set_args
from sconf import Config


if __name__ == "__main__":
    from serve.serve_utils import init
    io = init()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = cur_dir[:-9] + "log"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency-count", type=int, default=100)
    parser.add_argument("--base-model",type=str, default='checkpoints/ureader')
    parser.add_argument("--load-8bit", action="store_true", help="using 8bit mode")
    parser.add_argument("--bf16", action="store_true", help="using 8bit mode")
    parser.add_argument("--mm_config", type=str, default='configs/sft/release.yaml')
    args = parser.parse_args()
    config = Config(args.mm_config)
    add_config_args(config, args)
    set_args(args)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    from serve.web_server import mPLUG_Owl_Server, build_demo
    model = mPLUG_Owl_Server(
        base_model=args.base_model,
        log_dir=log_dir,
        load_in_8bit=args.load_8bit,
        bf16=args.bf16,
        device=device,
        io=io,
        config=config
    )
    demo = build_demo(model)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False).launch(server_name=args.host, debug=args.debug, server_port=args.port, share=False)