# SPDX-License-Identifier: Apache-2.0
"""
mini_bench.py —— vLLM 在线压测的精简学习版

只测三个指标：TTFT / TPOT / throughput。
面向 vLLM(或任何 OpenAI 兼容)的 /v1/completions 流式接口。

它是 vllm/benchmarks/serve.py 的"骨架抽取版"，砍掉了多模态、embedding、
goodput、投机解码、ramp-up、多后端分发等所有分支，只留下最核心的四步：

    采样请求  →  按到达模型发送  →  流式逐 token 打时间戳  →  汇总统计
    (build)      (get_request)      (async_request)           (calc_metrics)

对照 serve.py 的位置我都写在了各段注释里。改这个脚本来做实验最方便。

用法示例(先起一个 vLLM server)：
    vllm serve Qwen/Qwen2.5-0.5B-Instruct
    python mini_bench.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-prompts 100 --request-rate 10 \
        --input-len 256 --output-len 128
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np

MS = 1000.0  # 秒 -> 毫秒


# ---------------------------------------------------------------------------
# 1. 数据结构
#    对应 serve.py 的 SampleRequest / RequestFuncOutput / BenchmarkMetrics
# ---------------------------------------------------------------------------
@dataclass
class SampleRequest:
    """一条待发送的请求。真实 vLLM 里由 datasets/ 采样得到,这里用合成 prompt。"""
    prompt: str
    output_len: int


@dataclass
class RequestOutput:
    """单条请求的原始计时结果 —— 全部在客户端用 perf_counter 打点。
    对应 endpoint_request_func.py 的 RequestFuncOutput。"""
    success: bool = False
    ttft: float = 0.0                       # 首 token 延迟(秒)
    latency: float = 0.0                    # 端到端延迟(秒)
    itl: list[float] = field(default_factory=list)  # 相邻 token 到达间隔
    output_tokens: int = 0
    error: str = ""


@dataclass
class Metrics:
    """全局汇总指标。对应 serve.py 的 BenchmarkMetrics(只留我们关心的字段)。"""
    completed: int
    failed: int
    duration: float
    # throughput
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    # TTFT / TPOT (毫秒)
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float


# ---------------------------------------------------------------------------
# 2. 单请求：流式发送并逐 token 打时间戳
#    对应 endpoint_request_func.py:async_request_openai_completions
# ---------------------------------------------------------------------------
async def async_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    req: SampleRequest,
) -> RequestOutput:
    """发送一条 completions 请求,流式接收,记录 TTFT / ITL / latency。"""
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.output_len,
        "temperature": 0.0,
        "stream": True,
        # ignore_eos: 强制生成满 output_len 个 token,保证各请求输出长度可控,
        #             这样 TPOT/throughput 才有可比性(vLLM 压测默认也这么做)。
        "ignore_eos": True,
        # 让 server 在最后一个 chunk 里回传精确的 token 用量。
        "stream_options": {"include_usage": True},
    }

    out = RequestOutput()
    generated = ""
    st = time.perf_counter()          # <-- 计时起点:请求发出瞬间
    most_recent = st
    first_received = False

    try:
        async with session.post(url=api_url, json=payload) as resp:
            if resp.status != 200:
                out.error = f"HTTP {resp.status}: {await resp.text()}"
                return out

            # 逐行读取 SSE 流。每行形如:  data: {json}\n
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break

                now = time.perf_counter()
                import json
                chunk = json.loads(data)

                # 正文 token
                if chunk.get("choices"):
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        if not first_received:
                            first_received = True
                            out.ttft = now - st           # <-- TTFT
                        else:
                            out.itl.append(now - most_recent)  # <-- token 间隔
                        most_recent = now
                        generated += text

                # 最后一个 chunk 带精确 usage
                if chunk.get("usage"):
                    out.output_tokens = chunk["usage"].get("completion_tokens", 0)

            out.latency = most_recent - st                 # <-- 端到端延迟
            # 兜底:server 没回 usage 时,用收到的 token 段数估算
            if out.output_tokens == 0:
                out.output_tokens = len(out.itl) + 1 if first_received else 0
            out.success = first_received
    except Exception as e:  # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"

    return out


# ---------------------------------------------------------------------------
# 3. 到达模型：按 request_rate 控制"何时"发出每条请求
#    对应 serve.py:get_request  (这里只保留泊松/gamma,去掉 ramp-up 和 trace)
# ---------------------------------------------------------------------------
async def get_request(requests: list[SampleRequest], request_rate: float, burstiness: float):
    """异步生成器:按到达间隔逐条 yield 请求。

    request_rate == inf : 一次性全部发出(纯并发压测)。
    burstiness == 1     : 泊松过程(间隔服从指数分布),最贴近真实流量。
    burstiness  > 1     : 更均匀; < 1 : 更突发。
    """
    for req in requests:
        yield req
        if request_rate == float("inf"):
            continue
        # gamma(shape=burstiness, scale=1/(rate*burstiness)) 的均值 = 1/rate。
        # burstiness=1 时退化为指数分布 = 泊松到达。
        theta = 1.0 / (request_rate * burstiness)
        await asyncio.sleep(np.random.gamma(shape=burstiness, scale=theta))


# ---------------------------------------------------------------------------
# 4. 汇总统计:把每条请求的原始计时聚合成全局指标
#    对应 serve.py:calculate_metrics
# ---------------------------------------------------------------------------
def calculate_metrics(outputs: list[RequestOutput], duration: float) -> Metrics:
    ttfts, tpots = [], []
    total_output = 0
    completed = 0

    for o in outputs:
        if not o.success:
            continue
        completed += 1
        total_output += o.output_tokens
        ttfts.append(o.ttft)
        # TPOT (Time Per Output Token):除首 token 外,平均每个 token 的生成耗时。
        # = (端到端延迟 - 首 token 延迟) / (输出 token 数 - 1)
        if o.output_tokens > 1:
            tpots.append((o.latency - o.ttft) / (o.output_tokens - 1))

    failed = len(outputs) - completed
    ttfts = ttfts or [0.0]
    tpots = tpots or [0.0]

    return Metrics(
        completed=completed,
        failed=failed,
        duration=duration,
        request_throughput=completed / duration,
        output_throughput=total_output / duration,
        total_token_throughput=total_output / duration,  # 简化:只算输出侧
        mean_ttft_ms=float(np.mean(ttfts)) * MS,
        median_ttft_ms=float(np.median(ttfts)) * MS,
        p99_ttft_ms=float(np.percentile(ttfts, 99)) * MS,
        mean_tpot_ms=float(np.mean(tpots)) * MS,
        median_tpot_ms=float(np.median(tpots)) * MS,
        p99_tpot_ms=float(np.percentile(tpots, 99)) * MS,
    )


# ---------------------------------------------------------------------------
# 5. 主流程:建会话 -> warmup -> 并发发压 -> 统计 -> 打印
#    对应 serve.py:benchmark
# ---------------------------------------------------------------------------
def build_requests(num: int, input_len: int, output_len: int) -> list[SampleRequest]:
    """合成 num 条 prompt。用重复词近似 input_len 个 token(学习用,长度非精确)。"""
    prompt = " ".join(["hello"] * input_len)
    return [SampleRequest(prompt=prompt, output_len=output_len) for _ in range(num)]


async def run(args: argparse.Namespace) -> None:
    api_url = f"{args.base_url}/v1/completions"
    requests = build_requests(args.num_prompts, args.input_len, args.output_len)

    # 连接复用,减少 TLS/握手噪声;超时放大避免误杀慢请求。对应 serve.py 的 TCPConnector。
    connector = aiohttp.TCPConnector(limit=args.max_concurrency or 0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # warmup:先打一发,把冷启动开销挡在计时之外。
        print("Warming up...")
        await async_request(session, api_url, args.model, requests[0])

        # 用信号量限制"同时在飞"的请求数(与 request_rate 正交)。
        sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

        async def one(req: SampleRequest) -> RequestOutput:
            if sem is None:
                return await async_request(session, api_url, args.model, req)
            async with sem:
                return await async_request(session, api_url, args.model, req)

        print(
            f"Sending {args.num_prompts} requests "
            f"(rate={args.request_rate}, concurrency={args.max_concurrency})..."
        )
        tasks = []
        benchmark_start = time.perf_counter()
        async for req in get_request(requests, args.request_rate, args.burstiness):
            tasks.append(asyncio.create_task(one(req)))
        outputs = await asyncio.gather(*tasks)
        duration = time.perf_counter() - benchmark_start

    m = calculate_metrics(outputs, duration)
    print_metrics(m)


def print_metrics(m: Metrics) -> None:
    line = "-" * 50
    print(f"\n{line}\n{'Result':^50}\n{line}")
    print(f"{'Successful requests:':<35}{m.completed}")
    print(f"{'Failed requests:':<35}{m.failed}")
    print(f"{'Benchmark duration (s):':<35}{m.duration:.2f}")
    print(f"{'Request throughput (req/s):':<35}{m.request_throughput:.2f}")
    print(f"{'Output token throughput (tok/s):':<35}{m.output_throughput:.2f}")
    print(f"{'--- TTFT (ms) ---':^50}")
    print(f"{'  mean / median / p99:':<35}"
          f"{m.mean_ttft_ms:.2f} / {m.median_ttft_ms:.2f} / {m.p99_ttft_ms:.2f}")
    print(f"{'--- TPOT (ms) ---':^50}")
    print(f"{'  mean / median / p99:':<35}"
          f"{m.mean_tpot_ms:.2f} / {m.median_tpot_ms:.2f} / {m.p99_tpot_ms:.2f}")
    print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini vLLM benchmark (TTFT/TPOT/throughput)")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    # inf = 一次性全发;否则按该 RPS 的到达间隔发送。
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--burstiness", type=float, default=1.0)
    p.add_argument("--max-concurrency", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
