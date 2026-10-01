"""Check every tensor in dense.txt for NaN/Inf scale or values, straight from the file."""
import numpy as np

RT = r"D:/Strata/Strata-data/mtp/rt"
data = np.memmap(RT + r"\dense.bin", dtype=np.uint8, mode="r")

def f32_view(off, n):
    return np.frombuffer(data[off:off + n * 4].tobytes(), dtype="<f4")

def check_line(line):
    parts = line.split()
    name, kind, rows, cols, off, length = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    blob = data[off:off + length]
    if kind == "f32":
        v = np.frombuffer(blob.tobytes(), dtype="<f4")
        nan = int(np.isnan(v).sum()); inf = int(np.isinf(v).sum())
        print(f"{name:55s} f32  nan={nan} inf={inf} maxabs={np.abs(v).max():.4g} mean={v.mean():.4g}")
    elif kind == "bf16":
        u16 = np.frombuffer(blob.tobytes(), dtype="<u2").astype(np.uint32) << 16
        v = u16.view("<f4")
        nan = int(np.isnan(v).sum()); inf = int(np.isinf(v).sum())
        print(f"{name:55s} bf16 nan={nan} inf={inf} maxabs={np.abs(v).max():.4g}")
    elif kind == "q8_0":
        b = np.frombuffer(blob.tobytes(), dtype=np.uint8).reshape(-1, 34)
        scales = b[:, :2].copy().view(np.float16).astype(np.float32)
        codes = b[:, 2:].copy().view(np.int8)
        nan = int(np.isnan(scales).sum()); inf = int(np.isinf(scales).sum())
        print(f"{name:55s} q8_0 blocks={len(b)} scale_nan={nan} scale_inf={inf} "
              f"scale_max={np.abs(scales).max():.4g} scale_mean={np.abs(scales).mean():.4g} "
              f"code_min={codes.min()} code_max={codes.max()}")

with open(RT + r"\dense.txt", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            check_line(line.strip())
