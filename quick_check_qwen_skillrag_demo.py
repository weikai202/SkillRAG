import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_SHORT = "Qwen3-8B"
SAVE_DIR = "8b"
LAYERS = [12, 16, 20, 24, 28, 32]


def run_cmd(cmd, execute):
    print("$ " + " ".join(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def exp_rag_cmd(method, dataset, split, steps, max_new_tokens, prober_train_dataset=None):
    cmd = [
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
    if method in {"probing", "skillrag"}:
        cmd.extend(
            [
                "--ds",
                "3",
                "--position",
                "resid_post",
                "--threshold",
                "0.0",
                "--max_retrieval_rounds",
                "3",
            ]
        )
        if prober_train_dataset:
            cmd.extend(["--prober_train_dataset", prober_train_dataset])
    return cmd


def missing_ckpts(dataset, epoch=None):
    missing = []
    for layer in LAYERS:
        if epoch is None:
            dataset_pattern = (
                Path("ckpt")
                / "_3"
                / dataset
                / f"in3_1.0_{MODEL_SHORT}_tokens_mean_2_l{layer}_resid_post_ep*.pt"
            )
            legacy_pattern = (
                Path("ckpt")
                / "_3"
                / f"in3_1.0_{MODEL_SHORT}_tokens_mean_2_l{layer}_resid_post_ep*.pt"
            )
            if glob.glob(str(dataset_pattern)) or glob.glob(str(legacy_pattern)):
                continue
            missing.append(f"{dataset_pattern} or {legacy_pattern}")
        else:
            dataset_path = (
                Path("ckpt")
                / "_3"
                / dataset
                / f"in3_1.0_{MODEL_SHORT}_tokens_mean_2_l{layer}_resid_post_ep{epoch}.pt"
            )
            legacy_path = (
                Path("ckpt")
                / "_3"
                / f"in3_1.0_{MODEL_SHORT}_tokens_mean_2_l{layer}_resid_post_ep{epoch}.pt"
            )
            if not dataset_path.exists() and not legacy_path.exists():
                missing.append(f"{dataset_path} or {legacy_path}")
    return missing


def latest_output(dataset, method, split, steps):
    pattern = (
        f"dataset/{SAVE_DIR}/retrieval_qa_{MODEL_SHORT}_{dataset}_{method}_"
        f"{split}_after0_{steps}.csv"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def has_bad_continuation(text):
    return "\nQuestion:" in text or "\nQuery:" in text


def inspect_output(path, limit):
    if not path or not path.exists():
        print(f"[CHECK] output not found: {path}")
        return

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"[CHECK] file={path}")
    print(f"[CHECK] rows={len(rows)}")
    for i, row in enumerate(rows[:limit]):
        prompt = row.get("question_with_prompt", "")
        pred = row.get("pred", "")
        pred_with_prompt = row.get("pred_with_prompt", "")
        round_logs_raw = row.get("round_logs", "")
        has_think = "<think" in pred.lower() or "<think" in pred_with_prompt.lower()
        has_answer = "answer:" in pred.lower() or len(pred.strip()) > 0
        has_rationale = "rationale:" in pred.lower()
        prompt_no_think = (
            "/no_think" in prompt
            or "enable_thinking" not in prompt
            and "<|im_start|>user" in prompt
        )
        try:
            round_logs = json.loads(round_logs_raw) if round_logs_raw else []
        except Exception:
            round_logs = []
        skills = [r.get("selected_skill") for r in round_logs if r.get("selected_skill")]

        print(f"[SAMPLE {i}] prompt_no_think={prompt_no_think}")
        print(f"[SAMPLE {i}] has_think={has_think}")
        print(f"[SAMPLE {i}] has_rationale={has_rationale}")
        print(f"[SAMPLE {i}] has_answer={has_answer}")
        print(f"[SAMPLE {i}] bad_continuation={has_bad_continuation(pred)}")
        print(f"[SAMPLE {i}] rounds={len(round_logs)} skills={skills}")
        print(f"[SAMPLE {i}] pred={pred[:500].replace(os.linesep, ' ')}")


def main():
    parser = argparse.ArgumentParser(description="Minimal Qwen SkillRAG smoke demo.")
    parser.add_argument("--run", action="store_true", help="Execute commands. Default only prints them.")
    parser.add_argument("--dataset_name", default="nq")
    parser.add_argument("--prober_train_dataset", default="nq")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=64)
    parser.add_argument("--dev_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--rebuild_probe", action="store_true")
    parser.add_argument("--pretend_ckpt", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args()

    if not args.inspect_only:
        if args.rebuild_probe:
            for split, steps in [("train", args.train_steps), ("dev", args.dev_steps)]:
                for method in ["simple", "none"]:
                    run_cmd(
                        exp_rag_cmd(method, args.prober_train_dataset, split, steps, args.max_new_tokens),
                        args.run,
                    )

            run_cmd(
                [
                    sys.executable,
                    "balance_train_dataset.py",
                    "--model_id",
                    MODEL_ID,
                    "--dataset_name",
                    args.prober_train_dataset,
                ],
                args.run,
            )

            for layer in LAYERS:
                run_cmd(
                    [
                        sys.executable,
                        "train.py",
                        "--method",
                        "tokens_mean",
                        "--batch_size",
                        str(args.batch_size),
                        "--lr",
                        "0.001",
                        "--layer",
                        str(layer),
                        "--device",
                        args.device,
                        "--epochs",
                        str(args.epochs),
                        "--model_id",
                        MODEL_ID,
                        "--dataset_name",
                        args.prober_train_dataset,
                        "--train_ds_ratio",
                        "1.0",
                        "--max_length",
                        str(args.max_length),
                        "--disable_wandb",
                    ],
                    args.run,
                )

        ckpt_epoch = max(0, args.epochs - 1) if args.rebuild_probe else None
        missing = missing_ckpts(args.prober_train_dataset, epoch=ckpt_epoch)
        if missing:
            print("[CHECK] missing prober ckpts:")
            for path in missing:
                print(f"[CHECK]   {path}")
            if args.pretend_ckpt:
                print("[CHECK] pretend_ckpt enabled: continuing command preview/checks without real Qwen ckpts.")
            else:
                print("[CHECK] use --rebuild_probe, or run after full Qwen prober training.")
            if args.run and not args.rebuild_probe and not args.pretend_ckpt:
                raise SystemExit("[CHECK] stop: SkillRAG needs Qwen prober checkpoints.")
            if args.run and args.pretend_ckpt:
                print("[CHECK] not executing SkillRAG because pretend_ckpt cannot satisfy torch.load().")
                args.run = False

        run_cmd(
            exp_rag_cmd(
                "skillrag",
                args.dataset_name,
                "dev",
                args.steps,
                args.max_new_tokens,
                args.prober_train_dataset,
            ),
            args.run,
        )

    inspect_output(latest_output(args.dataset_name, "skillrag", "dev", args.steps), limit=3)


if __name__ == "__main__":
    main()
