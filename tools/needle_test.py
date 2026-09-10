#!/usr/bin/env python3
"""
长上下文召回精度测试（多轮随机化「大海捞针」）
Long-context retrieval accuracy test (multi-trial, randomized needle-in-a-haystack)
================================================================================
用途 / Purpose:
  验证上下文相关设置（KV 量化、上下文长度、Flash Attention 等）是否损伤召回能力。
  在长文档的**不同深度**埋入随机事实，让模型全部召回，跨多组统计命中率。

  Validate that context-related settings (KV cache quantization, context length,
  flash attention, ...) do not hurt retrieval. Facts are planted at random depths
  in a long document; the model must recall all of them; hits are aggregated
  across multiple trials.

设计要点 / Design notes:
  ① 多组随机化（避免单次巧合）；② 同随机种子可在不同配置间复现同一批针；
  ③ temperature=0；④ **检查 finish_reason，排除回答被 max_tokens 截断造成的假阴性**
     （这是最容易踩的坑：答案被截断 = 误判为「模型没找到」）

用法 / Usage:
  python needle_test.py --server /path/llama-server --model /path/model.gguf \
      --ncmoe 38 --ctx 262144 --trials 4 --extra "-ctk q8_0 -ctv q8_0"

  # 同一命令换 --extra/--ncmoe 跑另一配置，随机种子相同 → 用到完全相同的针，可直接对比
"""
import argparse, ctypes, json, os, platform, random, subprocess, sys, threading, time
import urllib.request

IS_WIN = platform.system() == "Windows"
STOP = [False]

CODES = ["XK-7412", "QB-3388", "MN-9057", "TR-2264", "WL-6613"]
NAMES = ["李慕白", "周砚秋", "沈砚清", "顾云舟", "苏景明"]
SECRETS = ["蓝鲸-9931", "赤狐-4402", "青隼-7758", "玄龟-3140", "白鹭-5526"]
FILLER = ("敦煌莫高窟始建于前秦建元二年，历经十六国、北魏、隋、唐、五代、西夏、元等朝代营建，"
          "现存洞窟七百三十五座，壁画四万五千平方米，彩塑两千四百余尊，"
          "1900 年藏经洞出土文书五万余件，催生了敦煌学。")


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


def req(port, max_tokens, prompt, timeout=900):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
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


def build_trial(rng, paras, needles_per_trial):
    """构造一组测试：在文档不同深度埋入随机事实"""
    picks = rng.sample(CODES + NAMES + SECRETS, needles_per_trial)
    depths = sorted(rng.sample(range(6, paras - 2), needles_per_trial))
    labels = ["档案编号", "保管人", "保险柜密码", "项目代号", "接头地点"]
    needles = {}
    for dep, val, lab in zip(depths, picks, labels):
        needles[dep] = f"【{lab}】{val}。"
    body = [f"第{i}段：" + (needles[i] + " " if i in needles else "") + FILLER
            for i in range(1, paras + 1)]
    prompt = ("请仔细阅读以下资料，然后回答问题。\n\n【资料开始】\n" + "\n".join(body)
              + f"\n【资料结束】\n\n问题：资料中提到的{'、'.join('①②③④⑤'[:needles_per_trial])} "
                f"{'、'.join(labels[:needles_per_trial])} 分别是什么？请逐项列出，每项都要回答。")
    return prompt, {v: lab for v, lab in zip(picks, labels)}, depths


def main():
    ap = argparse.ArgumentParser(description="Long-context retrieval accuracy test (multi-trial)")
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="模型首分片 / first GGUF shard")
    ap.add_argument("--ncmoe", type=int, default=42)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--trials", type=int, default=4, help="测试组数")
    ap.add_argument("--paras", type=int, default=120, help="文档段数（约 80 token/段）")
    ap.add_argument("--needles", type=int, default=3, help="每组埋入的事实数")
    ap.add_argument("--max-tokens", type=int, default=800, help="留足空间，避免截断造成假阴性")
    ap.add_argument("--seed", type=int, default=20260910, help="随机种子（跨配置保持一致）")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    name = os.path.basename(a.server)
    if kill_server(name):
        print("ABORT：有残留进程 / stale processes"); return 1

    logf = f"needle_ncmoe{a.ncmoe}.log"
    cmd = [a.server, "-m", a.model, "-ngl", "99", "--n-cpu-moe", str(a.ncmoe),
           "-fa", "on", "-fit", "off", "-c", str(a.ctx), "-np", "1", "--jinja", "--no-warmup",
           "--host", "127.0.0.1", "--port", str(a.port)]
    cmd += a.extra.split() if a.extra else []
    with open(logf, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_ready(a.port, 240):
        p.kill(); kill_server(name); print("ABORT 未就绪 / not ready"); return 2
    req(a.port, 16, "hi", 200)

    rng = random.Random(a.seed)
    label = f"ncmoe={a.ncmoe} ctx={a.ctx} extra=[{a.extra or 'default'}]"
    print(f"=== {label} | {a.trials} 组 × {a.needles} 针 ===\n")
    hit_total = total = 0
    for t in range(1, a.trials + 1):
        prompt, exp, depths = build_trial(rng, a.paras, a.needles)
        r, e = req(a.port, a.max_tokens, prompt)
        if r is None:
            print(f"  组{t}: 请求失败 {e}"); continue
        ch = r["choices"][0]
        ans, fin = ch["message"]["content"], ch.get("finish_reason", "?")
        hits = [k for k in exp if k in ans]
        hit_total += len(hits); total += len(exp)
        tt = r.get("timings", {})
        warn = "  ⚠回答被截断(请提高 --max-tokens)" if fin == "length" else ""
        print(f"  组{t}（针深 {depths}）: 命中 {len(hits)}/{len(exp)}{warn}  "
              f"gen={tt.get('predicted_per_second',0):.2f} tok/s  "
              f"prefill={tt.get('prompt_per_second',0):.1f} tok/s  "
              f"漏={[f'{k}({exp[k]})' for k in exp if k not in hits] or '无'}")

    p.kill(); time.sleep(3)
    rate = 100 * hit_total / total if total else 0
    print(f"\n**总命中率 / overall accuracy: {hit_total}/{total} = {rate:.0f}%**")
    print(f"（残留进程 / leftover: {kill_server(name)}）")
    open("needle_accuracy_results.txt", "a", encoding="utf-8").write(
        f"{label} | {hit_total}/{total} = {rate:.0f}%\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
