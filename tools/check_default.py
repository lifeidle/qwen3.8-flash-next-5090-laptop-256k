#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：不传 reasoning_effort 时，服务端默认用哪个档位"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.8-flash-next"
Q = "用一句话解释什么是引力。"


def run(effort=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": Q}],
            "max_tokens": 400, "stream": False, "temperature": 0.7}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    m = j["choices"][0]["message"]
    think = m.get("reasoning_content") or ""
    ans = m.get("content") or ""
    return len(think), len(ans), dt, j["usage"]["completion_tokens"], ans[:60]


for label, eff in [("不传(默认)", None), ("low", "low"), ("xhigh", "xhigh")]:
    try:
        tl, al, dt, ct, ans = run(eff)
        print(f"{label:<12} 思考={tl:>5}字  回答={al:>5}字  总耗时={dt:>6.1f}s  tokens={ct}  | {ans}")
    except Exception as e:
        print(f"{label:<12} 失败: {e}")
