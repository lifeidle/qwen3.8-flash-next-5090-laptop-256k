#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测 prefill 与 decode 速度（v3）。
v2 的 bug：第一次请求含冷启动开销（buffer 分配、kernel 编译），
导致 (t2 - t1) 失真。v3 先跑一次预热请求丢弃，再用两次差值。
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next"

SEG = ("人工智能的发展经历了多次浪潮，从符号主义到连接主义，再到今天的深度学习大规模预训练模型。"
       "每一次范式的转换都伴随着算力、数据与算法的协同进步，也带来了新的工程挑战与伦理讨论。")
TARGET_CHARS = 10000
PROMPT = (SEG * (TARGET_CHARS // len(SEG) + 1))[:TARGET_CHARS] + "\n\n请用一句话总结以上内容。"


def call(max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0.7}
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    u = j.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), dt, j["choices"][0].get("finish_reason")


# --- 预热（丢弃结果，消除冷启动） ---
wpt, wct, wt, _ = call(8)
print(f"[warmup] prompt {wpt} tok | {wt:.1f}s（丢弃）")

# --- A: 短输出 ---
pt1, ct1, t1, fr1 = call(8)
print(f"[A] 输出 {ct1} | {t1:.1f}s")

# --- B: 长输出 ---
pt2, ct2, t2, fr2 = call(200)
print(f"[B] 输出 {ct2} | {t2:.1f}s")

dtok = ct2 - ct1
dtime = t2 - t1
if dtime <= 0:
    print(f"结果: 测量无效（dtime={dtime:.2f}s）— 说明仍受启动/缓存影响")
    sys.exit(1)

decode_tps = dtok / dtime
prefill_tps = pt1 / max(t1 - ct1 / decode_tps, 0.001)
print()
print(f"结果: decode ≈ {decode_tps:.2f} tok/s   (差 {dtok} tok / {dtime:.2f}s)")
print(f"      prefill ≈ {prefill_tps:.1f} tok/s")
