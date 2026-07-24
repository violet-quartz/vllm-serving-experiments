# 量化模型对比

对比模型量化前后的性能、能耗与正确率。

## 1 量化模型

按照 [readme.md](../readme.md) 中的说明准备环境，原始模型：Qwen/Qwen2.5-7B-Instruct，其权重是 bf16 的。
关于量化模型，vllm 支持在线动态量化(online dynamic) 模型为 fp8。该种量化方式，不需要任何校准数据，加载时直接把 BF16/FP16 权重量化下去。具体机制:

所有 Linear 模块(除了最后的 lm_head)的权重被量化到 FP8_E4M3,用 per-tensor scale(整个张量共享一个缩放因子)
激活值则在每次前向传播时动态计算最大最小值,得到动态的 per-tensor scale。

它量化的是权重矩阵(attention 的 qkv/o_proj、MLP 的 gate/up/down),不碰 embedding、LayerNorm、lm_head——这些层要么对精度敏感,要么占比小。

我们先选用在线动态量化这种量化方式。

## 2 压测数据

压测数据集，我们仍选择 ShareGPT：

```bash
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 3 模型精度衡量

模型精度的衡量通常需要在多个数据集上进行测试，这里我们仅选择 GSM8K（Hugging Face: openai/gsm8k）数据集为例。 GSM8K 包含 8,500 道高质量的小学数学应用题，能够评测模型的数学推理能力。




## 4 实验步骤

使用 vllm 将模型在线动态量化为 fp8，只需在启动 vllm 服务时，加上 `--quantization fp8` 参数:

```bash
vllm serve /root/autodl-tmp/qwen2_5-7b-instruct/ \
    --quantization fp8 \
    --served_model_name qwen2.5-7b-instruct-fp8 \
    --port 8000
    --no-enable-prefix-caching
```

### 4.1 性能压测对比

参考[并发性能压测](exp_concurrency_performance.md) 中的性能压测方法，对动态量化的 fp8 模型进行压测



```
pip install lm-eval
```

```

lm_eval --model local-chat-completions \
  --model_args model=qwen2.5-7b-instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,tokenized_requests=False \
  --tasks gsm8k \
  --apply_chat_template \
  --limit 200 \
  --output_path eval_bf16/
```