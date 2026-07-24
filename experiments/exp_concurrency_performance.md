# 并发性能压测

## 1 指标说明

| 指标 | 含义 |
| --- | --- |
| 吞吐 | 单位时间内处理的 token 数（tok/s） |
| TTFT | Time To First Token，从发出请求到收到第一个 token 的耗时 |
| TPOT | Time Per Output Token，首 token 之后，平均每生成一个 token 的耗时 |

## 2 数据准备

这里我们以 shareGPT 为例。首先下载数据集：

```bash
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 3 使用 vllm bench serve 进行压测

### 3.1 实验 vllm bench serve 命令

```bash
vllm bench serve \
  --model /root/autodl-tmp/qwen2_5-7b-instruct/ \
  --served_model_name qwen2.5-7b-instruct \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200 \
  --max-concurrency 4
```

输出结果：

再次运行命令，输出结果：

### 3.2 写脚本对不同并发进行压测

```bash
# vLLM server 已在另一个终端跑着
python run_sweep.py \
  --model /root/autodl-tmp/qwen2_5-7b-instruct/ \
  --served-model-name qwen2.5-7b-instruct \
  --dataset-path /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --concurrency 8,16,32,48,64,128 --num-prompts 400
```

