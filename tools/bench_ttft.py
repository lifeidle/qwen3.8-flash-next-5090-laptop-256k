#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测 TTFT（首字节时间）、prefill 速率、decode 速率。
用 stream=True 精确捕捉首个 token 到达的时刻。
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next"

SEG = ("人工智能的发展经历了多次浪潮，从符号主义到连接主义，再到今天的深度学习大规模预训练模型。"
       "每一次范式的转换都伴随着算力、数据与算法的协同进步，也带来了新的工程挑战与伦理讨论。")
TARGET_CHARS = 24000
PROMPT = (SEG * (TARGET_CHARS // len(SEG) + 1))[:TARGET_CHARS] + "\n\n请用一句话总结以上内容。"


def measure(max_tokens=256, warm=True):
    """返回 (ttft, total_time, completion_tokens)"""
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "stream": True, "temperature": 0.7}
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    ct = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            delta = j.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                ct += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft, total, ct


if __name__ == "__main__":
    print("prompt chars: " + str(len(PROMPT)))

    # 预热（消除冷启动）
    _, _, _ = measure(8)
    print("[warmup] done")

    ttft, total, ct = measure(256)
    if ttft is None:
        print("no output received")
        raise SystemExit(1)

    decode_time = total - ttft
    decode_tps = (ct - 1) / decode_time if ct > 1 and decode_time > 0 else 0
    print()
    print(f"TTFT (first token)  = {ttft:.2f} s")
    print(f"total               = {total:.2f} s")
    print(f"output tokens       = {ct}")
    print(f"decode              = {decode_tps:.2f} tok/s")
