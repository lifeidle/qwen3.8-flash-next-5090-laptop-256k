"""Verify the speculative window size T: does spec_min_p gate it shut?

Root cause under test (src/program/generate.cpp:4056):
    int T = S_mtp;                                  # = min(mtp_max_t, spec)
    if (req_spec_min_p > 0.0) {
        T = 1;
        while (T < S_mtp && dprob[T-1] >= req_spec_min_p) ++T;
    }
draft_offered += T - 1   ->  "drafts accepted 0 of 0"  means  T == 1 always,
i.e. the drafter is never even given a chance, regardless of draft quality.

If spec_min_p == 0.0 the `if` is skipped and T = S_mtp unconditionally.
"""
import json, time, urllib.request, statistics, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
MODEL = "qwen3.8-flash-next-q2_0"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
METRICS = f"http://127.0.0.1:{PORT}/metrics"

PROMPTS = [
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "List twelve European capitals with one sentence about each.",
]


def run(prompt, tune, max_tokens=256):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if tune:
        body["strata_tune"] = tune
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    el = time.perf_counter() - t0
    n = d.get("usage", {}).get("completion_tokens", 0)
    return el, n, (n / el if el > 0 and n else 0)


def last_reqs(k=3):
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            m = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return f"(metrics unavailable: {e})"
    reqs = (m.get("requests") or [])[-k:]
    out = []
    for q in reqs:
        out.append(f"{q.get('engine_generated')}/{q.get('output_tokens')}"
                   f" hit={q.get('hit_rate')}")
    return "   ".join(out) if out else "(no requests recorded)"


def main():
    print(f"warming up on :{PORT} ...")
    run(PROMPTS[0], None, 96)
    print("warmup ok\n")

    print(f"{'variant':<26}{'median':>9}{'best':>9}{'worst':>9}   engine_gen/out_tok (last 3)")
    print("-" * 92)
    for name, tune in [
        ("spec_min_p 0.00 (forced)", {"spec_min_p": 0.0}),
        ("spec_min_p 0.05", {"spec_min_p": 0.05}),
        ("spec_min_p 0.50 (current)", {"spec_min_p": 0.5}),
        ("spec_min_p 1.00 (MTP off)", {"spec_min_p": 1.0}),
    ]:
        rates = []
        for p in PROMPTS:
            _, n, r = run(p, tune)
            if n:
                rates.append(r)
        if not rates:
            print(f"{name:<26}{'(failed)':>9}")
            continue
        print(f"{name:<26}{statistics.median(rates):>9.2f}{max(rates):>9.2f}"
              f"{min(rates):>9.2f}   {last_reqs(3)}")

    print("\nrestoring spec_min_p to the config default (0.5) ...")


if __name__ == "__main__":
    main()
