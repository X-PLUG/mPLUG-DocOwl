import os
import numpy as np
from icecream import ic
from pipeline.data_utils.registry import Registry, build_from_cfg
# from .data_utils.registry import Registry, build_from_cfg

PROCESSORS = Registry('processors')

def build_processors(processors_cfg):
    processors = dict()
    for task, processor in processors_cfg.items():
        processors[task] = build_from_cfg(processor, PROCESSORS)
        ic(type(processors[task]))
    return processors
