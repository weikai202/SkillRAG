import argparse
import os
from typing import List


def expected_post_ckpt_paths(model_id: str, layers: List[int], ds: int, epoch: int, dataset_name: str = "") -> List[str]:
    model_short = model_id.split("/")[-1]
    if ds == 3:
        prefix = "in3_1.0"
        base_dir = "ckpt/_3"
    elif ds == 25:
        prefix = "0.25"
        base_dir = "ckpt/_25"
    elif ds == 50:
        prefix = "0.5"
        base_dir = "ckpt/_5"
    elif ds == 75:
        prefix = "0.75"
        base_dir = "ckpt/_75"
    elif ds == 333:
        prefix = "in3_0.33"
        base_dir = "ckpt/_3_3"
    elif ds == 366:
        prefix = "in3_0.66"
        base_dir = "ckpt/_3_6"
    elif ds == 3000:
        prefix = "in3_1000"
        base_dir = "ckpt/_3_1000"
    elif ds == 1000:
        prefix = "1000"
        base_dir = "ckpt/_1000"
    else:
        # For unsupported ds, still check current training output convention.
        prefix = "in3_1.0"
        base_dir = "ckpt/_3"

    prefix_dataset = f"{dataset_name}/" if dataset_name else ""
    return [
        os.path.join(
            base_dir,
            f"{prefix_dataset}{prefix}_{model_short}_tokens_mean_2_l{layer}_resid_post_ep{epoch}.pt",
        )
        for layer in layers
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--layers", required=True, help="comma-separated layer list, e.g. 12,16,20")
    parser.add_argument("--ds", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    expected = expected_post_ckpt_paths(args.model_id, layers, args.ds, args.epoch, args.dataset_name)

    missing = [p for p in expected if not os.path.exists(p)]
    found = [p for p in expected if os.path.exists(p)]

    print(f"[CKPT CHECK] model={args.model_id}, ds={args.ds}, epoch={args.epoch}")
    print(f"[CKPT CHECK] expected_layers={layers}")
    print(f"[CKPT CHECK] found={len(found)}, missing={len(missing)}")

    if missing:
        for p in missing:
            print(f"[WARNING] Missing checkpoint: {p}")
        print("[WARNING] Some probe layers are missing. Evaluation may fail when loading prober.")
    else:
        print("[OK] All expected probe checkpoints were found.")


if __name__ == "__main__":
    main()
