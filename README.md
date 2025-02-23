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

3. start web-ui
```bash
streamlit run app.py
```
you can also start simple web-ui by following command:
```bash
streamlit run app_demo.py
```

4. open the webpage and input your question

## quant model and deploy
