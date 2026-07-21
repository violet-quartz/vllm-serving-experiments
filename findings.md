- 驱动版本
- vllm 命令，如何获取模型名
- prefix hit rate
    调用两次同样的请求
    同样请求，prefix rate 不断提升
- autodl
    - download from huggingface，`export HF_ENDPOINT=https://hf-mirror.com`
    - clone from github, 在终端中输入 `source /etc/network_turbo` 可加速
- vllm 官方压测脚本
  -  压测测的是服务端的延迟和吞吐。如果客户端每条请求都重新建 TCP 连接、重新做 TLS 握手、重新解析 DNS，那测出来的 TTFT
  里就混进了大量"客户端自己的开销"，数据就不准了。所以这里把连接层调成"尽量复用、尽量不重连"。