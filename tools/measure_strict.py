#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格测速（VBS 关闭后）：
  - temperature=0 → 输出确定性，每次长度一致，确保跨配置可比
  - 固定 max_tokens=512
  - 3 次取中位数
"""
import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next"
PROMPT = "请详细说明人工智能的发展历程，包括主要的技术流派、关键突破和代表性成果。"
REPEAT = 3
MAXTOK = 512


def run():
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK, "stream": False, "temperature": 0,
            "reasoning_effort": "low"}
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    ct = j["usage"]["completion_tokens"]
    pt = j["usage"]["prompt_tokens"]
    fr = j["choices"][0].get("finish_reason")
    return pt, ct, dt, ct / dt, fr


speeds = []
lens = []
for i in range(REPEAT):
    pt, ct, dt, tps, fr = run()
    speeds.append(tps)
    lens.append(ct)
    print(f"  第 {i+1} 次: prompt {pt} tok | 输出 {ct} tok | {dt:.1f}s | {tps:.2f} tok/s | finish={fr}")

print()
print(f"输出长度一致性: {lens}  {'✅ 一致（temperature=0 生效）' if len(set(lens)) == 1 else '⚠️ 不一致'}")
print(f"**中位数 decode = {statistics.median(speeds):.2f} tok/s**")
print()
print("对照参考（VBS 开启时，但输出长度不同，仅供参考）:")
print("  - 462 tok 输出 → 17.93 tok/s")
print("  - 1455~1935 tok 输出 → 16.04 tok/s")
