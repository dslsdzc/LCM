"""Convert Hugging Face Qwen safetensors to the flat NPZ used by qwen_lm."""
import argparse
import os
import numpy as np
from safetensors import deserialize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    arrays = {}
    with open(args.model, "rb") as f:
        tensors = deserialize(f.read())
    for key, spec in tensors:
        shape, dtype, raw = spec["shape"], spec["dtype"], spec["data"]
        if dtype == "BF16":
            # Expand BF16 bits into the high half of IEEE float32.
            bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = bits.view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f"unsupported dtype {dtype} for {key}")
        arrays[key] = arr.reshape(shape)
    np.savez(args.output, **arrays)
    print(f"saved {len(arrays)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
