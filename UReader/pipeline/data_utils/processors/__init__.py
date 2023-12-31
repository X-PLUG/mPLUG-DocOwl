# Copyright (c) Alibaba. All rights reserved.
from .builder import PROCESSORS, build_processors
from .default_processor import DefaultProcessor
from .caption_processor import CaptionProcessor
from .doc_processor import DocPretrainProcessor, DocSFTProcessor
__all__ = [
    'PROCESSORS', 'build_processors',
    'DefaultProcessor', 'CaptionProcessor',
    'DocPretrainProcessor', 'DocSFTProcessor'
]