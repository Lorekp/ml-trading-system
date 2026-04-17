#!/usr/bin/env python
"""Export a SAC model as a portable, cross-platform zip.

Strips CUDA-only objects (critic_sq) and Windows-pickled LR schedules
that cause segfaults when loaded on Linux or CPU-only PyTorch.

Usage:
    PYTHONPATH=. python scripts/export_model_portable.py [--input path] [--output path]
"""

import argparse
import json
import os
import zipfile

import numpy as np


DEFAULT_INPUT = "vol_arb/drl_hedging/models/production_sac_v4_final.zip"
DEFAULT_OUTPUT = "vol_arb/drl_hedging/models/production_sac_v4_final.zip"

# Objects that contain CUDA tensors and are only needed for training
CUDA_ONLY_KEYS = {"critic_sq", "critic_sq_target", "critic_sq_optimizer"}

# Default LR to replace pickled schedule functions (avoids cross-platform pickle)
DEFAULT_LR = 0.0003


def export_portable(input_path: str, output_path: str) -> None:
    """Strip non-portable objects from a SAC model zip."""
    tmp_path = output_path + ".tmp"

    with zipfile.ZipFile(input_path, "r") as zf_in:
        data = json.loads(zf_in.read("data"))

        # Strip CUDA-only critic objects
        for key in CUDA_ONLY_KEYS:
            if key in data:
                print(f"  Removed {key}")
                del data[key]

        # Replace pickled LR schedule with constant (avoids Windows path in pickle)
        if "learning_rate" in data and isinstance(data["learning_rate"], dict):
            data["learning_rate"] = DEFAULT_LR
            print(f"  Replaced learning_rate with constant {DEFAULT_LR}")
        if "lr_schedule" in data:
            del data["lr_schedule"]
            print("  Removed lr_schedule")

        # Rebuild zip
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            zf_out.writestr("data", json.dumps(data))
            for name in zf_in.namelist():
                if name != "data":
                    zf_out.writestr(name, zf_in.read(name))

    # Atomic replace
    os.replace(tmp_path, output_path)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")

    # Verify round-trip
    from stable_baselines3 import SAC

    model = SAC.load(output_path, device="cpu")
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  Verified: load + inference OK (action={action})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export portable SAC model")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Exporting portable model:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    export_portable(args.input, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
