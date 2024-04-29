import time
from typing import Dict, Optional, Sequence, List
import copy

import transformers
import tokenizers
import torch

from tinychart.data.process import register_preprocess
from tinychart.mm_utils import tokenizer_image_token
from tinychart import conversation as conversation_lib
from tinychart.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN

from packaging import version

# IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@register_preprocess('v1')
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    # conv = conversation_lib.default_conversation.copy()
    conv = conversation_lib.conv_phi_v0.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # total_len = len(target)

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        # cur_len = 1
        # cur_len = 1 + 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                # round_len = len(tokenizer_image_token(rou, tokenizer)) - 2 + 1
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids)
                # round_len = len(tokenizer(rou).input_ids) - 2 + 1
                # instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
            #     round_len -= 1
                # instruction_len -= 1
            instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        # target[cur_len:] = IGNORE_INDEX
        # import pdb;pdb.set_trace()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:

                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("conversation: ", conversations)
                print(target)
                print(input_ids)
                time.sleep(5)
                target[:] = IGNORE_INDEX
    return dict(
        input_ids=input_ids,
        labels=targets,
    )
