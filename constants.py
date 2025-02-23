from typing import Dict

import torch


TORCH_TYPE_MAP: Dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8
}