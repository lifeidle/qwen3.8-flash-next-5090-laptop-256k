#!/usr/bin/env python3
"""
多轮取中位数的服务端基准测试 / Multi-round server benchmark (median-of-N)
================================================================================
为什么不用单次测量 / Why median-of-N:
  llama.cpp 服务端的首轮响应普遍偏慢（页缓存/预热），单发测量会得出错误结论。
  本脚本在同一服务器会话内连续测 N 轮，报告中位数与极差。

Why not a single measurement:
  The first responses after startup are consistently slower (page cache / warm-up),
  so single-shot numbers mislead. This script runs N rounds in one server session
  and reports the median plus min/max.

用法 / Usage:
  python ab_bench.py \
      --server  /path/to/llama-server \
      --model   /path/to/model-00001-of-000NN.gguf \
      --ncmoe   38 --ctx 262144 --rounds 6 \
      --extra   "-ctk q8_0 -ctv q8_0"

  # 对比两个配置：分别跑一次，比较输出的「中位数」
  # To A/B two configs: run once per config and compare the reported medians.
"""
import argparse, ctypes, json, os, platform, statistics, subprocess, sys, threading, time
import urllib.request

IS_WIN = platform.system() == "Windows"
STOP, LOCK, RAM_LOG, T0 = [False], threading.Lock(), [], time.time()
MEASURE_PROMPT = "写一段 400 字左右的短文，介绍故宫的建筑特色与历史。"
LONG_PROMPT = ("请先仔细阅读以下资料，然后回答最后的问题。\n\n【资料开始】\n"
               + "".join(f"第{i}段：敦煌莫高窟始建于前秦建元二年，历经十六国、北魏、隋、唐、五代、西夏、元等朝代营建，"
                         f"现存洞窟七百三十五座，壁画四万五千平方米，彩塑两千四百余尊，"
                         f"1900 年藏经洞出土文书五万余件，催生了敦煌学。" for i in range(78))
               + "\n【资料结束】\n\n问题：根据资料，请简要概括莫高窟的历史跨度与艺术价值。")


# ─────────────────────────────────────────── 系统内存采样 / RAM sampling
def ram_sampler():
    if IS_WIN:
        class MC(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MC(); m.dwLength = ctypes.sizeof(MC)
        while not STOP[0]:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            with LOCK:
                RAM_LOG.append((time.time() - T0, m.dwMemoryLoad))
            time.sleep(1)
    else:
        while not STOP[0]:
            try:
                info = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
                with LOCK:
                    RAM_LOG.append((time.time() - T0,
                                    round(100 * (1 - info["MemAvailable"] / info["MemTotal"]))))
            except Exception:
                pass
            time.sleep(1)


def ram_peak_since(t):
    with LOCK:
        v = [x for (tt, x) in RAM_LOG if tt >= t]
    return max(v) if v else 0


# ─────────────────────────────────────────── 进程纪律 / process discipline
def kill_server(name):
    subprocess.run(["taskkill", "/F", "/IM", name] if IS_WIN else ["pkill", "-f", name],
                   capture_output=True)
    time.sleep(5)
    return count_server(name)


def count_server(name):
    if IS_WIN:
        r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}"], capture_output=True)
        return (r.stdout or b"").decode("gbk", errors="replace").lower().count(name.lower())
    r = subprocess.run(["pgrep", "-fc", name], capture_output=True, text=True)
    return int((r.stdout or "0").strip() or 0)


# ─────────────────────────────────────────── HTTP
def req(port, max_tokens, prompt, timeout):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def wait_ready(port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser(description="Multi-round llama.cpp server benchmark (median-of-N)")
    ap.add_argument("--server", required=True, help="llama-server 可执行文件 / path to llama-server")
    ap.add_argument("--model", required=True, help="模型首分片 / first GGUF shard")
    ap.add_argument("--ncmoe", type=int, default=38, help="--n-cpu-moe 值")
    ap.add_argument("--ctx", type=int, default=262144, help="上下文长度")
    ap.add_argument("--rounds", type=int, default=6, help="测量轮数（报告取中位数）")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--extra", default="", help="透传给 llama-server 的额外参数（如 -ctk q8_0 -ctv q8_0）")
    ap.add_argument("--long-test", action="store_true", help="附加 8k 长文预填充测试")
    ap.add_argument("--ready-timeout", type=int, default=240)
    a = ap.parse_args()

    threading.Thread(target=ram_sampler, daemon=True).start()
    time.sleep(2)

    name = os.path.basename(a.server)
    if kill_server(name):
        print("ABORT：有残留进程，先手动清理 / stale processes found")
        return 1

    logf = f"ab_bench_ncmoe{a.ncmoe}_ctx{a.ctx}.log"
    cmd = [a.server, "-m", a.model, "-ngl", "99", "--n-cpu-moe", str(a.ncmoe),
           "-fa", "on", "-fit", "off", "-c", str(a.ctx), "-np", "1",
           "--jinja", "--no-warmup", "--host", "127.0.0.1", "--port", str(a.port)]
    cmd += a.extra.split() if a.extra else []

    label = f"ncmoe={a.ncmoe} ctx={a.ctx} extra=[{a.extra or 'default'}]"
    print(f"启动中 / starting: {label}")
    t_start = time.time() - T0
    with open(logf, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_ready(a.port, a.ready_timeout):
        p.kill(); kill_server(name)
        tail = open(logf, encoding="utf-8", errors="replace").read()[-200:].replace("\n", " / ")
        print(f"FAIL 未就绪（可能显存不足）/ not ready: {tail}")
        return 2

    req(a.port, 16, "hi", 200)                                  # 预热
    req(a.port, 120, "用一句话说明什么是内存带宽。", 200)

    gens = []
    for i in range(a.rounds):
        r, e = req(a.port, 450, MEASURE_PROMPT, 300)
        if r is None:
            print(f"  轮次{i+1}: 失败 {e}"); break
        g = r.get("timings", {}).get("predicted_per_second", 0)
        gens.append(round(g, 2))
        print(f"  轮次{i+1}: {g:.2f} tok/s", flush=True)

    long_note = "skipped"
    if a.long_test and gens:
        r2, e2 = req(a.port, 200, LONG_PROMPT, 900)
        if r2 is not None:
            t2 = r2.get("timings", {})
            long_note = (f"prefill {t2.get('prompt_per_second',0):.1f} tok/s "
                         f"({r2['usage'].get('prompt_tokens',0)} tok)")
            for i in range(2):                                  # 大 KV 建立后再测
                r3, _ = req(a.port, 450, MEASURE_PROMPT, 300)
                if r3 is not None:
                    gens.append(round(r3.get("timings", {}).get("predicted_per_second", 0), 2))
                    print(f"  热态轮次{i+1}: {gens[-1]:.2f} tok/s", flush=True)

    p.kill(); time.sleep(3)
    stale = kill_server(name)
    med = statistics.median(gens) if gens else 0
    line = (f"RESULT | {label} | 轮次={gens} | **中位数={med:.2f} tok/s** | "
            f"min={min(gens) if gens else 0:.2f} max={max(gens) if gens else 0:.2f} | "
            f"RAM峰值={ram_peak_since(t_start)}% | {long_note} | 残留={stale}")
    print(line)
    open("ab_bench_results.txt", "a", encoding="utf-8").write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
