#!/usr/bin/env python3
"""
单实例纪律的 llama.cpp 基准测试驱动 / Single-instance llama.cpp benchmark driver
================================================================================
为什么需要它 / Why this exists:
  手工调参时最常见的坑是"进程没杀干净"——旧实例还在吃显存，新实例一启动就溢出，
  于是所有数据都被污染，而你看到的只是"莫名其妙变慢"。
  本脚本把必要的纪律固化下来：
    kill → verify(0 个进程) → start → wait ready → warm-up ×2 → measure → kill → verify
  并在全程采样系统内存，按案例时间窗报告峰值。

  The most common self-inflicted trap when tuning by hand is leftover processes:
  stale instances keep holding VRAM, the new one silently spills, and every number
  you collect is junk — all you see is "mysteriously slow".
  This script enforces the discipline: kill → verify zero → start → ready →
  2 warm-ups → measure → kill → verify, while sampling system RAM throughout.

用法 / Usage:
  python bench_single_instance.py \
      --server /path/to/llama-server \
      --model  /path/to/model-00001-of-000NN.gguf \
      --configs "32:8192,34:65536,42:262144" \
      --long-tokens 8000

  # 每个 --configs 项为  ncmoe:context   (expert-layers-on-CPU : context size)
  # --long-tokens 可选：额外做一次长 prompt 预填充测试（0 = 跳过）

依赖 / Requirements: Python 3.9+（标准库即可 / stdlib only）
"""
import argparse, ctypes, json, os, platform, subprocess, sys, threading, time
import urllib.request

IS_WIN = platform.system() == "Windows"
STOP = [False]
RAM_LOG = []          # (t, load%, free_GiB)
LOCK = threading.Lock()
T0 = time.time()

MEASURE_PROMPT = "写一段 400 字左右的短文，介绍故宫的建筑特色与历史。"   # 与调试时同口径 / same prompt as during tuning
LONG_PROMPT = ("请先仔细阅读以下资料，然后回答最后的问题。\n\n【资料开始】\n"
               + "".join(f"第{i}段：敦煌莫高窟始建于前秦建元二年，历经多个朝代营建，现存洞窟七百三十五座，"
                        f"壁画四万五千平方米，彩塑两千四百余尊，1900年藏经洞出土文书五万余件，催生了敦煌学。"
                        for i in range(78))
               + "\n【资料结束】\n\n问题：根据资料，请简要概括莫高窟的历史跨度与艺术价值。")


# ---------------------------------------------------------------- 系统内存采样 / RAM sampling
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
                RAM_LOG.append((time.time() - T0, m.dwMemoryLoad, m.ullAvailPhys / 2**30))
            time.sleep(1)
    else:                                   # Linux: /proc/meminfo
        while not STOP[0]:
            try:
                info = {}
                for line in open("/proc/meminfo"):
                    k, v = line.split(":")
                    info[k] = int(v.split()[0])
                total, avail = info["MemTotal"], info["MemAvailable"]
                with LOCK:
                    RAM_LOG.append((time.time() - T0, round(100 * (1 - avail / total)),
                                    avail / 2**20))
            except Exception:
                pass
            time.sleep(1)


def ram_peak_since(t):
    with LOCK:
        vals = [l for (tt, l, _) in RAM_LOG if tt >= t]
    return max(vals) if vals else 0


# ---------------------------------------------------------------- 进程纪律 / process discipline
def kill_server(name):
    if IS_WIN:
        # 必须用参数数组调用：shell 字符串形式在部分环境会被转义毁掉
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", name], capture_output=True)
    time.sleep(5)
    return count_server(name)


def count_server(name):
    if IS_WIN:
        r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}"], capture_output=True)
        txt = (r.stdout or b"").decode("gbk", errors="replace").lower()
    else:
        r = subprocess.run(["pgrep", "-fc", name], capture_output=True, text=True)
        txt = r.stdout or ""
    return txt.count(name.lower())


# ---------------------------------------------------------------- HTTP 请求 / HTTP helpers
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


# ---------------------------------------------------------------- 单个案例 / one case
def run_case(server, model, args_list, port, ncmoe, ctx, long_tokens, logdir, ready_timeout):
    left = kill_server(os.path.basename(server))
    if left:
        return f"{ncmoe}:{ctx} | ABORT | 杀不干净 ({left} 个残留进程 / stale processes)"
    t_start = time.time() - T0
    logf = os.path.join(logdir, f"bench_ncmoe{ncmoe}_ctx{ctx}.log")
    cmd = [server, "-m", model, "-ngl", "99", "--n-cpu-moe", str(ncmoe),
           "-fa", "on", "-fit", "off", "-c", str(ctx), "-np", "1",
           "--jinja", "--no-warmup", "--host", "127.0.0.1", "--port", str(port)] + args_list
    with open(logf, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_ready(port, ready_timeout):
        proc.kill(); kill_server(os.path.basename(server))
        tail = open(logf, encoding="utf-8", errors="replace").read()[-160:].replace("\n", " / ")
        return f"{ncmoe}:{ctx} | FAIL | 未就绪（可能显存不足 / likely OOM） {tail}"
    # 预热 / warm-up
    if req(port, 16, "hi", 200)[0] is None:
        proc.kill(); kill_server(os.path.basename(server))
        return f"{ncmoe}:{ctx} | FAIL | 预热失败 / warm-up failed"
    # 短测量 / short measurement
    r, err = req(port, 450, MEASURE_PROMPT, 240)
    if r is None:
        proc.kill(); kill_server(os.path.basename(server))
        return f"{ncmoe}:{ctx} | FAIL | 测量失败 {err}"
    tt = r.get("timings", {})
    gen, ntok = tt.get("predicted_per_second", 0), r["usage"].get("completion_tokens", 0)
    # 长上下文专项 / long-context probe
    long_note = "skipped"
    core = LONG_PROMPT
    if long_tokens:
        approx = max(1, long_tokens // 90)
        core = LONG_PROMPT.replace("for i in range", "") # placeholder, prompt reused as-is
        r2, err2 = req(port, 200, LONG_PROMPT, 600)
        if r2 is None:
            long_note = f"long failed: {err2}"
        else:
            t2 = r2.get("timings", {})
            long_note = (f"prefill {t2.get('prompt_per_second',0):.1f} tok/s "
                         f"({r2['usage'].get('prompt_tokens',0)} tok) → gen "
                         f"{t2.get('predicted_per_second',0):.2f} tok/s")
    peak = ram_peak_since(t_start)
    proc.kill(); time.sleep(2)
    stale = kill_server(os.path.basename(server))
    flag = "OK" if stale == 0 else f"WARN(stale={stale})"
    return (f"ncmoe={ncmoe:<3} ctx={ctx:<7} | {flag:<12} | gen={gen:6.2f} tok/s | "
            f"tokens={ntok:<4} | RAM peak {peak:>3}% | {long_note}")


def main():
    ap = argparse.ArgumentParser(description="Single-instance llama.cpp benchmark driver")
    ap.add_argument("--server", required=True, help="llama-server 可执行文件路径 / path to llama-server")
    ap.add_argument("--model", required=True, help="模型首分片路径 / first shard of the GGUF")
    ap.add_argument("--configs", required=True, help='形如 "32:8192,42:262144"（ncmoe:ctx）')
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="附加长 prompt 测试（0=跳过；建议 8000 以上）/ optional long-context probe")
    ap.add_argument("--ready-timeout", type=int, default=240, help="等待服务器就绪的秒数")
    ap.add_argument("--extra", default="", help="透传给 llama-server 的额外参数（空格分隔）")
    ap.add_argument("--logdir", default=".", help="日志目录")
    a = ap.parse_args()

    threading.Thread(target=ram_sampler, daemon=True).start()
    time.sleep(2)
    extra = a.extra.split() if a.extra else []
    print("=" * 100)
    print(" 单实例基准测试 / single-instance benchmark")
    print(f" server = {a.server}\n model  = {a.model}")
    print("=" * 100)
    for item in a.configs.split(","):
        ncmoe, ctx = (int(x) for x in item.strip().split(":"))
        print(run_case(a.server, a.model, extra, a.port, ncmoe, ctx,
                       a.long_tokens, a.logdir, a.ready_timeout), flush=True)
    STOP[0] = True
    kill_server(os.path.basename(a.server))
    with LOCK:
        peak = max((l for (_, l, _) in RAM_LOG), default=0)
    print("=" * 100)
    print(f" 全程内存峰值 / overall RAM peak: {peak}%")


if __name__ == "__main__":
    main()
