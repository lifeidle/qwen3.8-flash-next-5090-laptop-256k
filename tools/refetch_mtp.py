"""Re-fetch the MTP tensors whose saved bytes are actually safetensors shard headers.

The original fetch saved whatever the server returned without checking that a Range
request really came back as 206; when a mirror ignored the header it returned the whole
shard and the tool kept the first n_bytes - the shard's JSON header plus unrelated
tensor data.  This refetches the named tensors from the inventory ranges via curl and
refuses any response that looks like a safetensors file head.
"""
import json, os, subprocess, sys

D = r"D:/Strata/Strata-data/mtp"
REPO = "https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/main/"
inv = json.load(open(os.path.join(D, "mtp-inventory.json"), encoding="utf-8"))
man = json.load(open(os.path.join(D, "mtp-manifest.json"), encoding="utf-8"))
shard_of = {t["name"]: t["shard"] for t in inv["tensors"]}

def looks_like_shard_head(path):
    with open(path, "rb") as f:
        head = f.read(16)
    return len(head) >= 10 and head[8:10] == b'{"'

targets = sys.argv[1:] if len(sys.argv) > 1 else None
ok = fail = 0
for t in man:
    name = t["name"]
    if targets and not any(s in name for s in targets):
        continue
    p = os.path.join(D, t["file"])
    if targets is None and not looks_like_shard_head(p):
        print(f"skip (clean) {name}")
        continue
    shard = shard_of[name]
    rng = None
    for it in inv["tensors"]:
        if it["name"] == name:
            rng = (it["start"], it["end"])
            break
    url = REPO + shard
    tmp = p + ".tmp"
    n_expect = rng[1] - rng[0] + 1
    cmd = ["curl", "-L", "--ssl-no-revoke", "--retry", "3", "--retry-delay", "2",
           "-r", f"{rng[0]}-{rng[1]}", "-o", tmp, url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    size_ok = os.path.exists(tmp) and os.path.getsize(tmp) == n_expect
    head_ok = not looks_like_shard_head(tmp) if os.path.exists(tmp) else False
    if r.returncode == 0 and size_ok and head_ok:
        os.replace(tmp, p)
        ok += 1
        print(f"OK   {name}  ({n_expect} bytes)")
    else:
        fail += 1
        print(f"FAIL {name}  rc={r.returncode} size_ok={size_ok} head_ok={head_ok}")
        if os.path.exists(tmp):
            os.remove(tmp)
print(f"\nrefetched {ok}, failed {fail}")
