import json
import re
from typing import Dict, List, Tuple, Union

import torch

from config.config import TrainConfig, ChatConfig, APIKeyConfig
from prompts import train_prompt_en, train_prompt_zh

# from typing_extensions import TypedDict


TORCH_TYPE_MAP: Dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8
}

def load_config(encoding: str = "utf-8") -> Tuple[TrainConfig, APIKeyConfig]:
    with open("./config/train_config.json", "r", encoding=encoding) as f1, open(
        "./config/chat_config.json", "r", encoding=encoding
    ) as f2, open("./config/api_key.json", "r", encoding=encoding) as f3:
        train_config: TrainConfig = TrainConfig(**json.load(f1))
        chat_config: ChatConfig = ChatConfig(**json.load(f2))
        api_key_config: APIKeyConfig = APIKeyConfig(**json.load(f3))
    return train_config, chat_config, api_key_config


def has_chinese_char(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fa5]", text))


def formatting_prompts_func(examples: Dict, eos_token: str) -> Dict:
    inputs: List[str] = examples["Question"]
    cots: List[str] = examples["Complex_CoT"]
    outputs: List[str] = examples["Response"]

    texts: List[str] = []
    text: str = ""
    for input, cot, output in zip(inputs, cots, outputs):
        if has_chinese_char(input):
            text = train_prompt_zh.format(input, cot, output) + eos_token
        else:
            text = train_prompt_en.format(input, cot, output) + eos_token
        texts.append(text)
    return {
        "text": texts,
    }
