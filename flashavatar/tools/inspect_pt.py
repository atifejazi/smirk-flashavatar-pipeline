"""
prints key, shape, dtype, and a few stats per-component
so we can tell whether jaw / eyelid motion is actually present.
"""

import argparse
import sys

import numpy as np
import torch


def stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    flat = arr.reshape(-1)
    return (
        f"min={flat.min():+.4f} max={flat.max():+.4f} "
        f"mean={flat.mean():+.4f} std={flat.std():.4f} "
        f"|>1e-3|={int(np.sum(np.abs(flat) > 1e-3))}/{flat.size}"
    )


def per_dim_motion(arr):
    """For a (T, D) tensor, report per-dim std (i.e. how much each dim moves over time)."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return None
    return np.std(arr, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pt_file")
    args = p.parse_args()

    data = torch.load(args.pt_file, map_location="cpu")
    if not isinstance(data, dict):
        print(f"top-level is not a dict, got {type(data)}", file=sys.stderr)
        sys.exit(1)

    print(f"file: {args.pt_file}")
    print(f"keys: {list(data.keys())}")
    print()

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            print(f"  {k:<22} tensor shape={tuple(arr.shape)} dtype={arr.dtype}")
            print(f"    overall: {stats(arr)}")
            pdm = per_dim_motion(arr)
            if pdm is not None:
                pdm_repr = ", ".join(f"{i}:{s:.3f}" for i, s in enumerate(pdm))
                print(f"    per-dim std over T: [{pdm_repr}]")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:<22} list len={len(v)} sample={type(v[0]).__name__ if v else 'empty'}")
        else:
            print(f"  {k:<22} {type(v).__name__} value={v!r}")
        print()


if __name__ == "__main__":
    main()
