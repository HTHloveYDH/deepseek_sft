import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import TORCH_TYPE_MAP


def main(
    model_dir: str,
    lora_adapter_dir: str,
    torch_dtype: str, 
    save_model_dir: str,
    save_quant_model_dir:str,
    save_lora_adapter: bool,
    save_method: str,
    quantization_method: str,
):
    # load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=TORCH_TYPE_MAP.get(torch_dtype, torch.bfloat16),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #exit()
    # load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_dir, device_map="cuda")

    # merge base model with lora adapter
    merged_model = model.merge_and_unload()

    if save_lora_adapter:
        merged_model.save_pretrained(save_model_dir)
        tokenizer.save_pretrained(save_model_dir)
    else:
        merged_model.save_pretrained_merged(
            save_model_dir,
            tokenizer,
            max_shard_size="8GB",        # 分片存储防止大文件问题
            save_method=save_method,
        )

    # quant model and save to gguf format
    merged_model.save_pretrained_gguf(
        save_quant_model_dir,
        tokenizer,
        quantization_method,
    )


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
        "--torch_dtype",
        type=str,
        required=True,
        help="torch type for loading model",
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )

    parser.add_argument(
        "--save_quant_model_dir",
        type=str,
        required=True,
        help="Directory to save the quant model",
    )

    parser.add_argument(
        "--save_lora_adapter",
        action="store_true",
        help="Whether to save the LoRA adapter separately",
    )

    parser.add_argument(
        "--save_method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit","lora"],
        help="Method to save the model (pytorch or safetensors)",
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
        torch_dtype=args.torch_dtype,
        save_model_dir=args.save_model_dir,
        save_quant_model_dir=args.save_quant_model_dir,
        save_lora_adapter=args.save_lora_adapter,
        save_method=args.save_method,
        quantization_method=args.quantization_method,
    )
