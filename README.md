# DeepSeek R1模型Lora微调

## down load model weights and dataset
```bash
python hf_download.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --save_dir ./models
python hf_download.py --dataset FreedomIntelligence/medical-o1-reasoning-SFT --save_dir ./data
```

## training env config
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

## vllm inference env config
```bash
conda create -n vllm python=3.12
conda activate vllm

pip install vllm==0.8.1
pip install streamlit
pip install isort
pip install black
```

## sglang inference env config
```bash
conda create -n sglang python=3.12
conda activate sglang

pip install sglang[all]
pip install streamlit
pip install isort
pip install black
```

## visulize training history
```bash
wandb login
```

## supervised finetune
python train.py

## inference
python chat.py

## start vllm server
1. start server
```bash
vllm serve /path/to/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager
```

2. access remote server by following command
```bash
curl http://localhost:9000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

## start sglang server
1. start server
```bash
python -m sglang.launch_server --model /path/to/DeepSeek-R1-Distill-Qwen-32B --dp 1 --tp 1 --nnodes 1 --trust-remote-code
```

2. access remote server by following command
```bash
curl http://localhost:9000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "path/to/your/gguf_model.gguf",
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
    "model": "path/to/your/gguf_model.gguf",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

## start web-ui
```bash
streamlit run app.py
```
you can also start simple web-ui by following command:
```bash
streamlit run app_demo.py
```

open the webpage and input your question

## merge lora
```bash
python ./merge_lora.py --model_dir path/to/your/base_model_folder --lora_adapter_dir path/to/your/lora_adapter_folder --max_seq_length 32768 --torch_dtype auto --save_model_dir /path/to/your/lora_mergerd_model_folder --save_method merged_16bit
```

## merge lora, quantize and deploy
### clone and compile llama.cpp
```bash
git clone https://github.com/ggml-org/llama.cpp
```
#### CPU Build
Build llama.cpp using CMake:
```bash
cmake -B build
cmake --build build --config Release
```
- For faster compilation, add the -j argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, cmake --build build --config Release -j 8 will run 8 jobs in parallel.

- For faster-repeated compilation, install [ccache](https://ccache.dev/)

#### CUDA Build
Plz install [CUDA](https://developer.nvidia.com/cuda-toolkit) before you begin building llama.cpp
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```
#### Move the compiled bin file
```bash
cd build
mv ./bin/llama-quantize ../
mv ./bin/llama-server ../
```

**If you encounter any error in the build procedure, please reference the official doc [llama.cpp_build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)**

## merge base model with lora adpater
```bash
python merge_lora.py --model_dir /path/to/base_model --lora_adapter_dir /path/to/lora_adapter --max_seq_length 32768 --torch_dtype bfloat16 --save_model_dir /path/to/target_dir --save_method merged_16bit
```

## merge base model with lora adpater and quant to gguf format
```bash
python ./merge_lora_quant_to_gguf.py --model_dir path/to/your/base_model_folder --lora_adapter_dir path/to/your/lora_adapter_folder --max_seq_length 32768 --torch_dtype auto --save_quant_model_dir /path/to/your/lora_mergerd_quant_model_folder --quantization_method q4_k_m
```
