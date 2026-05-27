import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_SHORT = "Qwen3-8B"
SAVE_DIR = "8b"


def run_cmd(cmd, execute):
    print("$ " + " ".join(str(x) for x in cmd))
    if execute:
        subprocess.run(cmd, check=True)


def require_train_cli_support():
    required_args = [
        "--train_data_path",
        "--dev_data_path",
        "--debug_dump_path",
        "--debug_dump_limit",
    ]
    result = subprocess.run(
        [sys.executable, "train.py", "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout + result.stderr
    missing = [arg for arg in required_args if arg not in help_text]
    if result.returncode != 0 or missing:
        raise SystemExit(
            "Current train.py is not the patched Qwen demo version. "
            f"Missing args: {', '.join(missing) or 'unknown'}. "
            "Please sync train.py, utils.py, exp_rag.py, and quick_train_qwen_probe_demo.py."
        )


def exp_rag_cmd(method, dataset, split, steps, max_new_tokens):
    return [
        sys.executable,
        "exp_rag.py",
        "--retr_method",
        method,
        "--is_sparse",
        "--tr_or_dev",
        split,
        "--extracting_cot_qa",
        "--extract_sep",
        "--steps_limit",
        str(steps),
        "--dataset_name",
        dataset,
        "--is_cot",
        "--sep_number",
        "0",
        "--model_id",
        MODEL_ID,
        "--max_new_tokens",
        str(max_new_tokens),
    ]


def generated_path(dataset, method, split, steps):
    return Path(
        f"dataset/{SAVE_DIR}/retrieval_qa_{MODEL_SHORT}_{dataset}_{method}_{split}_after0_{steps}.csv"
    )


def merged_train_path(dataset):
    return Path(f"dataset/{SAVE_DIR}/retrieval_qa_{MODEL_SHORT}_{dataset}_all_train_in3_.csv")


def merged_dev_path(dataset):
    return Path(f"dataset/{SAVE_DIR}/retrieval_qa_{MODEL_SHORT}_{dataset}_all_zeroshot_test_500.csv")


def has_think(text):
    return "<think" in str(text).lower() or "</think" in str(text).lower()


def has_bad_continuation(text):
    text = str(text)
    return "\nQuestion:" in text or "\nQuery:" in text


def inspect_generation(files, output_csv, limit):
    rows_out = []
    for path in files:
        if not path.exists():
            rows_out.append({"file": str(path), "status": "missing"})
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows[:limit]):
            prompt = row.get("question_with_prompt", "")
            pred = row.get("pred", "")
            pred_with_prompt = row.get("pred_with_prompt", "")
            rows_out.append(
                {
                    "file": str(path),
                    "status": "ok",
                    "index": i,
                    "retr_method": row.get("retr_method", ""),
                    "prompt_has_no_think": "/no_think" in prompt or "<|im_start|>user" in prompt,
                    "pred_has_think": has_think(pred),
                    "pred_with_prompt_has_think": has_think(pred_with_prompt),
                    "pred_has_rationale": "rationale:" in pred.lower(),
                    "pred_has_answer": "answer:" in pred.lower() or bool(pred.strip()),
                    "pred_bad_continuation": has_bad_continuation(pred),
                    "question_with_prompt": prompt,
                    "pred": pred,
                    "pred_with_prompt": pred_with_prompt,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows_out for key in row.keys()})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return rows_out


def write_summary(path, generation_rows, token_dump_path, train_csv, dev_csv, ckpt_path):
    checked = [r for r in generation_rows if r.get("status") == "ok"]
    summary = {
        "num_generation_rows_checked": len(checked),
        "num_generation_rows_with_think": sum(
            1 for r in checked if r.get("pred_has_think") or r.get("pred_with_prompt_has_think")
        ),
        "num_generation_rows_with_bad_continuation": sum(
            1 for r in checked if r.get("pred_bad_continuation")
        ),
        "token_dump_path": str(token_dump_path),
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "ckpt_path": str(ckpt_path),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Minimal Qwen probe-training demo with output dumps.")
    parser.add_argument("--run", action="store_true", help="Execute commands. Default only prints them.")
    parser.add_argument("--dataset_name", default="nq")
    parser.add_argument("--steps", type=int, default=9, help="exp_rag uses a post-step break, so 9 gives about 10 rows.")
    parser.add_argument("--dev_steps", type=int, default=9)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--inspect_limit", type=int, default=10)
    args = parser.parse_args()

    if args.run:
        require_train_cli_support()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports")
    generation_dump = report_dir / f"qwen_probe_demo_generation_{stamp}.csv"
    token_dump = report_dir / f"qwen_probe_demo_token_spans_{stamp}.csv"
    summary_path = report_dir / f"qwen_probe_demo_summary_{stamp}.json"

    for split, steps in [("train", args.steps), ("dev", args.dev_steps)]:
        for method in ["none", "simple"]:
            run_cmd(exp_rag_cmd(method, args.dataset_name, split, steps, args.max_new_tokens), args.run)

    train_csv = merged_train_path(args.dataset_name)
    dev_csv = merged_dev_path(args.dataset_name)
    ckpt_path = (
        Path("ckpt")
        / "_3"
        / args.dataset_name
        / f"in3_1.0_{MODEL_SHORT}_tokens_mean_2_l{args.layer}_resid_post_ep{max(0, args.epochs - 1)}.pt"
    )

    train_cmd = [
        sys.executable,
        "train.py",
        "--method",
        "tokens_mean",
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "0.001",
        "--layer",
        str(args.layer),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--model_id",
        MODEL_ID,
        "--dataset_name",
        args.dataset_name,
        "--train_ds_ratio",
        "1.0",
        "--train_data_path",
        str(train_csv),
        "--dev_data_path",
        str(dev_csv),
        "--max_length",
        str(args.max_length),
        "--debug_dump_path",
        str(token_dump),
        "--debug_dump_limit",
        str(args.inspect_limit),
        "--disable_wandb",
    ]
    run_cmd(train_cmd, args.run)

    if not args.run:
        print(f"[DEMO] generation_dump={generation_dump}")
        print(f"[DEMO] token_dump={token_dump}")
        print(f"[DEMO] summary={summary_path}")
        print("[DEMO] dry-run only. Add --run on the GPU machine to execute and write reports.")
        return

    generation_files = [
        generated_path(args.dataset_name, "none", "train", args.steps),
        generated_path(args.dataset_name, "simple", "train", args.steps),
        generated_path(args.dataset_name, "none", "dev", args.dev_steps),
        generated_path(args.dataset_name, "simple", "dev", args.dev_steps),
        train_csv,
        dev_csv,
    ]
    generation_rows = inspect_generation(generation_files, generation_dump, args.inspect_limit)
    summary = write_summary(summary_path, generation_rows, token_dump, train_csv, dev_csv, ckpt_path)

    print(f"[DEMO] generation_dump={generation_dump}")
    print(f"[DEMO] token_dump={token_dump}")
    print(f"[DEMO] summary={summary_path}")
    print(f"[DEMO] summary_values={json.dumps(summary, ensure_ascii=False)}")
    if not args.run:
        print("[DEMO] dry-run only. Add --run on the GPU machine to execute.")


if __name__ == "__main__":
    main()
