import argparse
import os

import torch
from peft import PeftModel
from unsloth import FastLanguageModel

from constants import TORCH_TYPE_MAP


def main(
    model_dir: str,
    lora_adapter_dir: str,
    max_seq_length: int,
    torch_dtype: str,
    save_quant_model_dir: str,
    quantization_method: str,
):
    # 确保目录存在
    os.makedirs(save_quant_model_dir, exist_ok=True)

    print("加载基础模型和LoRA适配器...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=TORCH_TYPE_MAP.get(torch_dtype, None),
        load_in_4bit=False,
    )

    lora_model = PeftModel.from_pretrained(
        model,
        lora_adapter_dir,
        torch_dtype=TORCH_TYPE_MAP.get(torch_dtype, None),
    )

    print("合并模型中...")
    merged_model = lora_model.merge_and_unload()
    print("模型合并结束")
    
    # 然后尝试生成GGUF格式
    try:
        print(f"正在生成GGUF格式模型到 {save_quant_model_dir}...")
        merged_model.save_pretrained_gguf(
            save_quant_model_dir,
            tokenizer,
            quantization_method=quantization_method,
        )
        print("GGUF格式模型生成成功!")
    except Exception as e:
        print(f"生成GGUF格式失败: {e}")
        print("您可以手动使用llama.cpp来量化模型。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")

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
        "--save_quant_model_dir",
        type=str,
        required=True,
        help="Directory to save the quant model",
    )

    parser.add_argument(
        "--quantization_method",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q2_k", "q3_k_l", "q3_k_m"],
        help="Quantization method for GGUF format",
    )

    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        lora_adapter_dir=args.lora_adapter_dir,
        max_seq_length=args.max_seq_length,
        torch_dtype=args.torch_dtype,
        save_quant_model_dir=args.save_quant_model_dir,
        quantization_method=args.quantization_method,
    )