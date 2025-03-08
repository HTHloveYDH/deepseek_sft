# Library imports
from unsloth import FastLanguageModel, is_bfloat16_supported

from prompts import medical_chat_prompt_en, medical_chat_prompt_zh
from utils import formatting_prompts_func, has_chinese_char, load_config


def main():
    """-------------------------------------- setup --------------------------------------"""
    # hyper-parameter
    _, chat_config, _ = load_config()
    dtype = None  # torch.bfloat16

    """-------------------------------------- load model --------------------------------------"""
    # fetch model from unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=chat_config.model_dir,
        max_seq_length=chat_config.max_seq_length,
        dtype=dtype,
        load_in_8bit=chat_config.load_in_8bit,
        load_in_4bit=chat_config.load_in_4bit,
        token=None,
    )
    FastLanguageModel.for_inference(model)

    """-------------------------------------- generate --------------------------------------"""
    chat_prompt = (
        chat_prompt_zh
        if has_chinese_char(chat_config.user_input)
        else chat_prompt_en
    )
    inputs = tokenizer(
        [chat_prompt_en.format(chat_config.user_input, "")], return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)

    print(response[0].split("### Response:")[1])


if __name__ == "__main__":
    main()
