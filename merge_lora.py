# save_merged_model.py
import argparse
import os

import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from transformers import PreTrainedModel

from constants import TORCH_TYPE_MAP


def main(
    model_dir: str,
    lora_adapter_dir: str,
    max_seq_length: int,
    torch_dtype: str,
    save_model_dir: str,
):
    # 确保目录存在
    os.makedirs(save_model_dir, exist_ok=True)

    print("加载基础模型和LoRA适配器...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=TORCH_TYPE_MAP.get(torch_dtype, None),
        load_in_4bit=True,
    )

    lora_model = PeftModel.from_pretrained(
        model,
        lora_adapter_dir,
        torch_dtype=torch.float16,
    )

    print("合并模型中...")
    merged_model = lora_model.merge_and_unload()
    
    print(f"保存合并模型到 {save_model_dir}...")
    # 使用标准的transformers保存方法而不是unsloth的方法
    if isinstance(merged_model, PreTrainedModel):
        merged_model.save_pretrained(save_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_model_dir)
        print("模型保存成功!")
    else:
        print("错误：模型不是PreTrainedModel类型")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save merged model")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the base model",
    )
    
    parser.add_argument(
        "--lora_adapter_dir",
        type=str,
        required=True,
        help="Directory containing the LoRA adapter",
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="max input token sequence length",
    )

    parser.add_argument(
        "--torch_dtype",
        type=str,
        required=True,
        choices=["bfloat16", "float16", "float32", "int8", "auto"],
        help="torch type for loading model",
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )
    
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        lora_adapter_dir=args.lora_adapter_dir,
        max_seq_length=args.max_seq_length,
        torch_dtype=args.torch_dtype,
        save_model_dir=args.save_model_dir,
    )