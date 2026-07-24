# 并发性能压测

## 1 指标说明

| 指标 | 含义 |
| --- | --- |
| 吞吐 | 单位时间内处理的 token 数（tok/s） |
| TTFT | Time To First Token，从发出请求到收到第一个 token 的耗时 |
| TPOT | Time Per Output Token，首 token 之后，平均每生成一个 token 的耗时 |

## 2 下载数据

这里我们以 shareGPT 为例。首先下载数据集：

```bash
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 3 使用 vllm bench serve 进行压测

### 3.1 实验 vllm bench serve 命令

vllm bench serve 是 vllm 源码中自带的一个压测工具，下面我们来实验一下它的使用。

按照 [readme.md](../readme.md) 中的命令将 vllm 服务启动，同时在另一个终端窗口输入以下命令：

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
这里注意 backend 和 endpoint 参数是要配套的，对于 shareGPT 数据集，backend 选取 openai-chat。

输出结果：

```
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Maximum request concurrency:             4         
Benchmark duration (s):                  184.01    
Total input tokens:                      49360     
Total generated tokens:                  42866     
Request throughput (req/s):              1.09      
Output token throughput (tok/s):         232.95    
Peak output token throughput (tok/s):    249.00    
Peak concurrent requests:                11.00     
Total token throughput (tok/s):          501.19    
---------------Time to First Token----------------
Mean TTFT (ms):                          65.96     
Median TTFT (ms):                        52.74     
P99 TTFT (ms):                           129.83    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.48     
Median TPOT (ms):                        16.31     
P99 TPOT (ms):                           19.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.34     
Median ITL (ms):                         16.21     
P99 ITL (ms):                            20.40     
==================================================
```

再次运行命令，输出结果：

```
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Maximum request concurrency:             4         
Benchmark duration (s):                  183.15    
Total input tokens:                      49360     
Total generated tokens:                  43293     
Request throughput (req/s):              1.09      
Output token throughput (tok/s):         236.38    
Peak output token throughput (tok/s):    249.00    
Peak concurrent requests:                11.00     
Total token throughput (tok/s):          505.89    
---------------Time to First Token----------------
Mean TTFT (ms):                          50.58     
Median TTFT (ms):                        48.11     
P99 TTFT (ms):                           70.63     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.23     
Median TPOT (ms):                        16.23     
P99 TPOT (ms):                           16.42     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.15     
Median ITL (ms):                         16.20     
P99 ITL (ms):                            18.28     
==================================================
```

对比两次结果，发现第二次运行时 TTFT 的指标要显著优于第一次；同时，从 vllm server 运行日志中可以看到，第一次运行时 Prefix cache hit rate 比较低，最高 6% - 7%， 而第二次运行时 Prefix cache hit rate 显著升高，从 7% 逐渐上涨到 50%；因此猜测是 vllm 引擎默认开启 prefix cache， 两次运行同样的数据集，第二次运行 prefix cache hit rate 会升高，导致 TTFT 指标明显提升。

第一次运行日志：
![第一次运行时 vllm server 日志](../assets/vllm_log_1.png)

第二次运行日志：
![第二次运行时 vllm server 日志](../assets/vllm_log_2.png)

为了排除 prefix cache 的影响，在启动 vllm 服务时，关闭 prefix cache 功能：

```bash
vllm serve /root/autodl-tmp/qwen2_5-7b-instruct/ \
    --served_model_name qwen2.5-7b-instruct \
    --port 8000 \
    --no-enable-prefix-caching
```

再次运行两次上述 vllm bench serve 命令，发现两次运行的指标结果基本一致，所以之后我们用同一数据集连续压测时，都关闭 vllm 引擎的 prefix cache 功能，以保证结果的可比性。


### 3.2 写脚本对不同并发进行压测

脚本在 [/src](../src/) 文件夹中。

```bash
# vLLM server 已在另一个终端跑着
python run_sweep.py \
  --model /root/autodl-tmp/qwen2_5-7b-instruct/ \
  --served-model-name qwen2.5-7b-instruct \
  --dataset-path /root/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --concurrency 8,16,32,48,64,128 --num-prompts 400
```

