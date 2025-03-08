import os
from typing import Dict, List, Union

import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login

# from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

# Library imports
from unsloth import FastLanguageModel, is_bfloat16_supported

import wandb
from prompts import chat_prompt_en, chat_prompt_zh
from utils import formatting_prompts_func, load_config


def main():
    """-------------------------------------- setup --------------------------------------"""
    # env variables
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # TODO: find out reason
    # hyper-parameter
    train_config, _, api_key_config = load_config()
    dtype = None  # torch.bfloat16
    # wandb setup
    wandb.login(key=api_key_config.wnb_api_key)
    run = wandb.init(
        project="DeepSeek-finetune", job_type="training", anonymous="allow"
    )

    """-------------------------------------- load model --------------------------------------"""
    # fetch model from unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_config.model_dir,
        max_seq_length=train_config.max_seq_length,
        dtype=dtype,
        load_in_8bit=train_config.load_in_8bit,
        load_in_4bit=train_config.load_in_4bit,
        token=None,
    )
    eos_token = tokenizer.eos_token  # must add eos_token

    # load base model with lora for peft
    lora_model = FastLanguageModel.get_peft_model(
        model,
        r=train_config.lora_rank,  # LoRA rank - no of trainable parameters in adapter layers
        target_modules=[  # layers to apply lora to
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=train_config.lora_alpha,  # scaling factor for lora layers (higher values may stabilize training)
        lora_dropout=0,  # dropout rate for lora adapters (0 means no dropout)
        bias="none",  # no additional bias parameters are trained
        use_gradient_checkpointing="unsloth",  # saves GPU memory, reduces training speed
        random_state=3407,  # random seed
        use_rslora=False,  # rank-stabilized LoRA is disabled
        loftq_config=None,
    )

    """-------------------------------------- load data --------------------------------------"""
    # Load dataset from hf
    train_dataset_en = load_dataset(
        train_config.data_dir,
        "en",
        split="train",  # full data
        trust_remote_code=True,
    )
    train_dataset_zh = load_dataset(
        train_config.data_dir,
        "zh",
        split="train",  # full data
        trust_remote_code=True,
    )
    train_dataset = concatenate_datasets([train_dataset_en, train_dataset_zh])
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        fn_kwargs={"eos_token": eos_token},
        batched=True,
    ).shuffle(seed=42)

    """-------------------------------------- test before SFT --------------------------------------"""
    question = """
        A 60 year old woman with a long history of involuntary urine loss during activities like coughing or sneezig but no leakage at night undergoes a gynecological exam and a Q-tip test.
        Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions ?
    """
    # set model to inference mode
    FastLanguageModel.for_inference(model)

    inputs = tokenizer([chat_prompt_en.format(question, "")], return_tensors="pt").to(
        "cuda"
    )

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)

    print(response[0].split("### Response:")[1])

    """-------------------------------------- SFT --------------------------------------"""
    # initialize SFFTrainer from hf
    trainer = SFTTrainer(
        model=lora_model,  # use lora model defined above
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        # training arguments
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=model.max_seq_length,  # from model
            dataset_num_proc=2,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            num_train_epochs=train_config.num_train_epochs,
            warmup_steps=train_config.warmup_steps,
            # max_steps=60,
            learning_rate=train_config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # start model training
    trainer_stats = trainer.train(resume_from_checkpoint=train_config.resume_from_checkpoint)

    # model train stats
    wandb.finish()

    """-------------------------------------- save model after SFT --------------------------------------"""
    if train_config.save_lora_adapter:
        model.save_pretrained(train_config.save_model_dir)
        tokenizer.save_pretrained(train_config.save_model_dir)
    else:
        model.save_pretrained_merged(
            train_config.save_model_dir,
            tokenizer,
            save_method=train_config.save_method,
        )

    """-------------------------------------- quant model after SFT --------------------------------------"""
    model.save_pretrained_gguf(
        train_config.save_quant_model_dir,
        tokenizer,
        train_config.quantization_method,
    )

    """-------------------------------------- test after SFT --------------------------------------"""
    # model inference after finetuning
    question = """
    A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing
    but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
    what would cystometry most likely reveal about her residual volume and detrusor contractions?
    """

    FastLanguageModel.for_inference(model_lora)  # Unsloth has 2x faster inference!

    inputs = tokenizer([chat_prompt_en.format(question, "")], return_tensors="pt").to(
        "cuda"
    )

    outputs = model_lora.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)

    print(response[0].split("### Response:")[1])



if __name__ == "__main__":
    main()
