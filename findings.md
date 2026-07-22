- 驱动版本
- vllm 命令，如何获取模型名
- prefix hit rate
    调用两次同样的请求
    同样请求，prefix rate 不断提升
- autodl
    - download from huggingface，`export HF_ENDPOINT=https://hf-mirror.com`
    - clone from github, 在终端中输入 `source /etc/network_turbo` 可加速
    - 从远程机器下载数据 scp -rP 35394 root@region-1.autodl.com:<实例中的文件/文件夹> <本地文件/文件夹> （注意需要在您本地的机器上执行）
- vllm 官方压测脚本
  -  压测测的是服务端的延迟和吞吐。如果客户端每条请求都重新建 TCP 连接、重新做 TLS 握手、重新解析 DNS，那测出来的 TTFT
  里就混进了大量"客户端自己的开销"，数据就不准了。所以这里把连接层调成"尽量复用、尽量不重连"。
- vllm bench serve 
    - --backend 和 --endpoint 参数要配套，/v1/completions 和 /v1/chat/completions 不同
- 窗口平均功率会被首尾低载段稀释，不能直接拿来比不同并发的功率
    - 现象：power_timeline 里并发 1 的平均功率(316.9W)反而比并发 4/8(275/264W)还高，看起来"单路更耗电"，反直觉。
    - 原因：avg_power_W = 窗口总能耗 / 窗口时长，窗口两端有客户端爬坡(GPU 没喂满)和收尾 drain 的低功耗段(~50W)。这段绝对时长基本固定，但窗口时长随并发暴跌(728s→46s)，所以短窗口(高并发)被稀释得厉害——窗口内 util<50% 的采样占比从并发 1 的 1.9% 一路涨到并发 32 的 31.6%。
    - 真相：只取 util≥90% 的稳态采样求平均，各档功率其实基本持平在 ~310–340W（并发 1 和 32 都顶在 ~340W）。batch=1 的 decode 是显存带宽瓶颈，每步都要读一遍全部权重，即使单路也把显存打满、SM 频率顶到最高，所以稳态功率一点不低。
    - 解决：analyze.py 新增 steady_power_W 列（util≥阈值再求平均，阈值由 --util-thresh 控制，默认 90），另出 steady_frac 记录窗口内满载采样占比。J/token 的能量仍用整窗口累计计数器差值(end−start)，不动。