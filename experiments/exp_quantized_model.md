# 量化模型对比

对比 Qwen2.5-7B-Instruct 在量化前后的性能、精度与显存占用。原始模型权重为 bf16，量化方式为 vLLM 的在线动态量化（online dynamic）fp8。

## 1 实验结论

在 RTX 4090 上，使用 ShareGPT 压测性能、GSM8K 评测精度，fp8 相比 bf16 的表现如下：

- **吞吐显著提升**：各并发下输出吞吐提高约 **34%–47%**，并发 128 时总吞吐从 5274.6 提升到 7068.0 tok/s。
- **延迟全面下降**：mean TPOT 降低约 **25%–32%**，mean TTFT 降低约 **18%–27%**。
- **精度基本无损**：GSM8K（flexible-extract）fp8 70.58% vs bf16 69.60%，差异约 1 个百分点，落在标准误内，统计上不可区分。
- **显存占用更低**：fp8 加载占用的显存明显少于 bf16，为更大 batch / 更长上下文留出余量。

**综合结论**：在本实验负载下，fp8 在线动态量化是一次性能、显存与精度的三方共赢；详细数据见 [5.1 性能压测对比](#51-性能压测对比) 与 [5.2 精度对比](#52-精度对比)。

## 2 量化方式

按照 [readme.md](../readme.md) 中的说明准备环境。原始模型为 Qwen/Qwen2.5-7B-Instruct，权重精度为 bf16。

本实验采用 vLLM 内置的 **fp8 在线动态量化（online dynamic quantization）**，属于 W8A8（权重与激活均为 8-bit）方案，具体特点如下：

- **无需校准数据**：不依赖任何校准集或离线量化流程，服务启动加载权重时即完成量化，是一种 post-training、零校准的量化方式，接入成本最低。
- **权重量化**：所有 Linear 层（`lm_head` 除外）的权重被量化为 `FP8_E4M3`（1 符号位 + 4 指数位 + 3 尾数位），采用 per-tensor 缩放，即整个权重张量共享一个静态 scale，在加载时一次性算好。
- **激活量化**：激活值在每次前向传播中动态统计其数值范围，实时计算 per-tensor 的动态 scale 后再量化；动态 scale 能更好地适配不同输入的分布，减小量化误差。
- **量化范围**：仅量化 attention 的 `qkv_proj`/`o_proj` 与 MLP 的 `gate`/`up`/`down_proj` 等权重矩阵（模型的主要计算量与显存占用所在），而 embedding、LayerNorm、`lm_head` 保持原精度——这些层要么对量化误差敏感，要么参数占比很小，量化收益有限。

fp8 矩阵乘法可直接利用 Ada 架构（RTX 4090）的 FP8 Tensor Core，在降低权重显存占用的同时提升计算吞吐，因此本实验优先选用该方案。

## 3 压测数据集

压测数据集，我们仍选择 ShareGPT：

```bash
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 4 精度评测方法

模型精度的衡量通常需要在多个数据集上综合测试，这里仅以 GSM8K（Hugging Face: openai/gsm8k）数据集为例。GSM8K 包含约 8,500 道高质量的小学数学应用题，常用于评测模型的数学推理能力。

评测框架采用 lm-eval（EleutherAI 的 lm-evaluation-harness 的简称），它是当前大语言模型（LLM）开源社区中较为权威、使用广泛的标准化评测框架。

```
pip install lm-eval
pip install lm-eval[api]
```


## 5 实验步骤

使用 vLLM 将模型在线动态量化为 fp8，只需在启动 vLLM 服务时加上 `--quantization fp8` 参数：

```bash
vllm serve /root/autodl-tmp/qwen2_5-7b-instruct/ \
    --quantization fp8 \
    --served_model_name qwen2.5-7b-instruct-fp8 \
    --port 8000 \
    --no-enable-prefix-caching
```

从 vLLM 的启动日志可见，fp8 加载占用的显存明显少于 bf16，为更大 batch / 更长上下文留出余量。

![启动动态量化模型服务](../assets/vram_dynamic_fp8_model.png)

![启动原始模型服务](../assets/vram_original_model.png)


### 5.1 性能压测对比

参考[并发性能压测](exp_concurrency_performance.md) 中的性能压测方法，对动态量化的 fp8 模型进行压测

```bash
# vLLM server 在线动态量化已在另一个终端跑着
python run_sweep.py \
  --model /root/autodl-tmp/qwen2_5-7b-instruct/ \
  --served-model-name qwen2.5-7b-instruct-fp8 \
  --dataset-path /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --outdir run4 \
  --concurrency 8,16,32,64,128 --num-prompts 400
```

```bash
python plot.py --windows run4/windows.jsonl --outdir run4
```

将 fp8 量化模型（run4）与原始 bf16 模型（[并发性能压测 run1](exp_concurrency_performance.md)）在相同并发下对比：

| 并发 | 输出吞吐 (tok/s) bf16 → fp8 | mean TTFT (ms) bf16 → fp8 | mean TPOT (ms) bf16 → fp8 |
| ---: | --- | --- | --- |
| 8   | 463.1 → 676.5（+46%） | 69.4 → 56.6（-18%） | 16.7 → 11.4（-32%） |
| 16  | 853.6 → 1236.3（+45%） | 80.1 → 64.5（-19%） | 17.7 → 12.2（-31%） |
| 32  | 1367.1 → 2002.4（+47%） | 121.5 → 92.7（-24%） | 21.2 → 14.5（-32%） |
| 64  | 2039.2 → 2859.8（+40%） | 266.6 → 201.3（-24%） | 27.7 → 19.8（-29%） |
| 128 | 2580.0 → 3457.2（+34%） | 766.2 → 557.2（-27%） | 42.9 → 32.0（-25%） |

**结论：**

- **吞吐显著提升**：fp8 在各并发下输出吞吐比 bf16 高约 **34%–47%**；并发 128 时总吞吐从 5274.6 提升到 7068.0 tok/s。
- **延迟全面下降**：mean TPOT 降低约 **25%–32%**，直接得益于 fp8 计算/访存更快；mean TTFT 也下降约 **18%–27%**。
- **收益随并发变化的趋势相反**：吞吐的相对优势随并发升高而收窄（46% → 34%，高并发下逐渐受 KV cache、调度等非量化因素制约），而 TTFT 的相对优势随并发升高而扩大（18% → 27%，量化后单步更快，在高并发排队时更能摊薄首 token 等待）。


### 5.2 精度对比

对量化模型在 GSM8K 上进行精度测试，结果输出到 [eval_fp8](../experiment_results/quantized_model/eval_fp8) 中

```
lm_eval --model local-chat-completions \
  --model_args model=qwen2.5-7b-instruct-fp8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,tokenized_requests=False,add_bos_token=True \
  --tasks gsm8k \
  --apply_chat_template \
  --output_path eval_fp8/
```

对原始模型在 GSM8K 上进行精度测试，结果输出到 [eval_bf16](../experiment_results/quantized_model/eval_bf16) 中

```
lm_eval --model local-chat-completions \
  --model_args model=qwen2.5-7b-instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8,tokenized_requests=False,add_bos_token=True \
  --tasks gsm8k \
  --apply_chat_template \
  --output_path eval_bf16/
```

两个模型在 GSM8K（1319 题）上的 exact-match 准确率如下（括号为标准误）：

| 模型 | flexible-extract | strict-match |
| --- | ---: | ---: |
| bf16（原始） | 69.60% (±1.27%) | 16.83% (±1.03%) |
| fp8（量化） | 70.58% (±1.26%) | 19.79% (±1.10%) |

**结论：**

- **精度基本无损**：以 flexible-extract 为准（该指标从回答中提取最后一个数字，反映真实推理能力），fp8 为 70.58%，bf16 为 69.60%，相差仅约 1 个百分点，且落在两者标准误范围内（差值 ≈1.0pp < 合并标准误 ≈1.8pp），统计上不可区分。fp8 甚至略高，属正常波动。
- **strict-match 两者都很低，不宜作为对比依据**：strict-match 要求回答严格匹配 `#### 数字` 格式，Qwen2.5-Instruct 的自由格式回答大多不满足，故 bf16/fp8 都只有约 17%–20%，这是格式解析问题而非推理能力差异。


