# vLLM 大模型服务部署实验

本仓库基于 vLLM 引擎部署大模型服务，并围绕推理优化开展一系列实验，旨在实践常见的推理优化方法并观察其实际效果。

## 1 实验环境

### 1.1 机器与镜像

在 [AutoDL](https://www.autodl.com/) 上租用一台 RTX 4090 实例，基础镜像为 Python 3.12 + PyTorch 2.12.1 + CUDA 13.0。

### 1.2 安装 vLLM

```bash
pip install vllm
```

查看版本：

```bash
vllm --version
# 0.25.1
```

### 1.3 下载初始模型

从 ModelScope 下载模型（国内网络友好）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
    --local_dir /root/autodl-tmp/qwen2_5-7b-instruct
```

### 1.4 启动 vLLM 服务

```bash
vllm serve /root/autodl-tmp/qwen2_5-7b-instruct/ \
    --served_model_name qwen2.5-7b-instruct \
    --port 8000
```

### 1.5 验证服务

另开一个终端发起请求。

查询已加载的模型列表：

```bash
curl http://localhost:8000/v1/models
```

发起一次对话请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [{"role": "user", "content": "你好，用一句话介绍你自己"}]
  }'
```

能够正确返回回复，说明服务已就绪。

## 2 实验内容

| 实验 | 说明 | 文档 |
| --- | --- | --- |
| 并发性能压测 | 压测不同并发下的服务性能数据 | [experiments/exp_concurrency_performance.md](experiments/exp_concurrency_performance.md) |
| 量化模型对比 | 对比模型量化前后的性能、能耗与正确率 | [experiments/exp_quantized_model.md](experiments/exp_quantized_model.md) |
