# DeepSeek R1模型Lora微调

## down load model weights and dataset
```bash
python hf_download.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --save_dir ./models
python hf_download.py --dataset FreedomIntelligence/medical-o1-reasoning-SFT --save_dir ./data
```

## env config
```bash
conda create -n deepseek python=3.12
conda activate deepseek

pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

pip install vllm
pip install wandb
pip install streamlit
pip install isort
pip install black
```
or
```bash
pip install -r requirements.txt
```

## visulize training history
```bash
wandb login
```

## start vllm server
1. start server
```bash
vllm serve ./models/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager
```

2. access remote server by following command
```bash
curl http://localhost:9000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "./models/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

## start llama.cpp server
1. start server
```bash
./llama-server -m path/to/your/gguf_model.gguf --port 8000
```

2. access remote server by following command
```bash
curl http://localhost:9000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "./models/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'


## start web-ui
```bash
streamlit run app.py
```
you can also start simple web-ui by following command:
```bash
streamlit run app_demo.py
```

4. open the webpage and input your question

## merge lora
```bash
python ./merge_lora.py --model_dir path/to/your/base_model_folder --lora_adapter_dir path/to/your/lora_adapter_folder --max_seq_length 32768 --torch_dtype auto --save_model_dir /path/to/your/lora_mergerd_model_folder --save_method merged_16bit
```

## quant model and deploy
### clone and compile llama.cpp

### run following command
```bash
python ./merge_lora_quant_to_gguf.py --model_dir path/to/your/base_model_folder --lora_adapter_dir path/to/your/lora_adapter_folder --max_seq_length 32768 --torch_dtype auto --save_quant_model_dir /path/to/your/lora_mergerd_quant_model_folder --quantization_method q4_k_m
```