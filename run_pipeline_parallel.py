"""Two-GPU parallel pipeline runner.

Splits work across cuda:0 and cuda:1 while respecting all data dependencies:
 - Indexing:   all 5 datasets in parallel (CPU-bound, no GPU needed)
 - Build:      datasets split across GPUs; simple→none order preserved per dataset
 - Balance:    CPU-only, runs after each dataset's build completes
 - Training:   sequential per-dataset on ONE GPU (checkpoints overwrite by design —
               the last dataset must be trivia to match original behaviour)
 - Evaluation: datasets split across GPUs (after training completes)

Usage:
    python run_pipeline_parallel.py --config configs/llama3_8b.yaml
    python run_pipeline_parallel.py --config configs/llama3_8b.yaml --dry_run
"""

import argparse
import ast
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from collect_build_metrics import collect_metrics

try:
    import yaml
except Exception:
    yaml = None


MODEL_TO_DIR = {
    "google/gemma-2b": "2b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "8b",
    "Qwen/Qwen3-8B": "8b",
    "google/gemma-2-9b-it": "9b",
}


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── subprocess helpers ───────────────────────────────────────────────

def run_cmd(cmd: List[str], dry_run: bool, gpu: Optional[int] = None) -> Dict[str, Any]:
    """Run a command, optionally pinned to a specific GPU."""
    cmd = [c for c in cmd if c]
    cmd_str = " ".join(cmd)
    gpu_label = f"[GPU:{gpu}]" if gpu is not None else ""
    if dry_run:
        print(f"[DRY-RUN]{gpu_label}", cmd_str)
        return {"command": cmd_str, "returncode": 0, "gpu": gpu}
    print(f"[RUN]{gpu_label}", cmd_str)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # wandb may not be authenticated; don't let it block training
    env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "online"))
    proc = subprocess.run(cmd, check=False, env=env)
    return {"command": cmd_str, "returncode": proc.returncode, "gpu": gpu}


def _run_index(ds_name: str) -> Dict[str, Any]:
    """Build a sparse index for one dataset (top-level for pickling)."""
    cmd = ["python", "make_indexer.py", "--dataset_name", ds_name, "--is_sparse"]
    print(f"[RUN-PARALLEL] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return {"command": " ".join(cmd), "returncode": proc.returncode}


def run_cmd_async(cmd: List[str], gpu: Optional[int] = None) -> Tuple[subprocess.Popen, str, Optional[int]]:
    """Launch a command asynchronously, returning (Popen, cmd_str, gpu)."""
    cmd = [c for c in cmd if c]
    cmd_str = " ".join(cmd)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "online"))
    gpu_label = f"[GPU:{gpu}]" if gpu is not None else ""
    print(f"[RUN-ASYNC]{gpu_label}", cmd_str)
    proc = subprocess.Popen(cmd, env=env)
    return proc, cmd_str, gpu


# ── metrics (same as run_pipeline.py) ────────────────────────────────

def collect_method_metrics_from_result(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_id = cfg["model"]["id"]
    datasets = cfg["evaluate"]["datasets"]
    methods = cfg["evaluate"]["methods"]
    threshold = cfg["evaluate"]["threshold"]
    steps = cfg["evaluate"]["steps_limit"]
    ds = cfg["prober"]["ds"]
    retr_type = "sparse" if cfg["retrieval"]["is_sparse"] else "dense"
    cot_name = "cot" if cfg["retrieval"]["is_cot"] else "nocot"
    ablation = cfg["prober"]["ablation"]
    rows: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        for method in methods:
            path = f"result/{ablation}_{ds}_{retr_type}_{dataset_name}_{threshold}_{method}_{cot_name}_dev_{steps}.csv"
            if not os.path.exists(path):
                rows.append({"model_id": model_id, "dataset_name": dataset_name, "retr_method": method, "acc": None, "em": None, "file": path})
                continue
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict() if len(df) else {}
            rows.append({
                "model_id": model_id, "dataset_name": dataset_name, "retr_method": method,
                "acc": float(row.get("acc")) if "acc" in row else None,
                "em": float(row.get("em")) if "em" in row else None,
                "file": path,
            })
    return rows


def collect_skillrag_trace(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_id = cfg["model"]["id"]
    model_short = model_id.split("/")[-1]
    save_dir = MODEL_TO_DIR[model_id]
    datasets = cfg["evaluate"]["datasets"]
    out: List[Dict[str, Any]] = []
    for dataset_name in datasets:
        pattern = f"dataset/{save_dir}/retrieval_qa_{model_short}_{dataset_name}_skillrag_dev_*.csv"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not matches:
            continue
        path = matches[0]
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            trace = []
            if "round_logs" in row and isinstance(row["round_logs"], str):
                try:
                    trace = json.loads(row["round_logs"])
                except Exception:
                    try:
                        trace = ast.literal_eval(row["round_logs"])
                    except Exception:
                        pass
            out.append({
                "dataset_name": dataset_name,
                "question_with_prompt": row.get("question_with_prompt", ""),
                "pred": row.get("pred", ""),
                "answer": row.get("answer", ""),
                "acc": row.get("acc", None),
                "em": row.get("em", None),
                "round_logs": trace,
                "file": path,
            })
    return out


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs to use (default: 0,1)")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) >= 2, "Need at least 2 GPUs"
    gpu_a, gpu_b = gpus[0], gpus[1]

    cfg = load_yaml(args.config)
    model_id = cfg["model"]["id"]
    model_short = model_id.split("/")[-1]
    datasets_build = cfg["build_dataset"]["datasets"]
    index_datasets = cfg.get("index_datasets", datasets_build)
    steps_train = cfg["build_dataset"]["steps_limit_train"]
    steps_dev = cfg["build_dataset"]["steps_limit_dev"]
    sep_number = cfg["build_dataset"].get("sep_number", 0)
    is_sparse = cfg["retrieval"]["is_sparse"]
    is_cot = cfg["retrieval"]["is_cot"]
    build_index = cfg.get("build_index", True)
    ds = cfg["prober"]["ds"]
    threshold = cfg["evaluate"]["threshold"]
    position = cfg["evaluate"]["position"]
    eval_extracting_cot_qa = cfg["evaluate"].get("extracting_cot_qa", True)
    eval_extract_sep = cfg["evaluate"].get("extract_sep", True)
    eval_sep_number = cfg["evaluate"].get("sep_number", 0)
    eval_tr_or_dev = cfg["evaluate"].get("tr_or_dev", "dev")
    eval_max_retrieval_rounds = cfg["evaluate"].get("max_retrieval_rounds", 3)
    train_ratio = cfg["train"]["train_ds_ratio"]

    run_logs: List[Dict[str, Any]] = []

    # ── Pre-create shared directories and files to avoid TOCTOU races ─
    save_dir = MODEL_TO_DIR[model_id]
    for d in ["result", f"dataset/{save_dir}", "ckpt/_3", "pckpt/_3",
              "raw_data/sparse_index", "raw_data/dense_index"]:
        os.makedirs(d, exist_ok=True)
    # Seed metrics_log.yaml header so parallel appenders don't race on it.
    metrics_yaml = os.path.join("result", "metrics_log.yaml")
    if not os.path.exists(metrics_yaml):
        with open(metrics_yaml, "w", encoding="utf-8") as f:
            f.write("runs:\n")

    # ── Phase 1: Parallel index building (CPU-bound) ─────────────────
    if build_index:
        if args.dry_run:
            for ds_name in index_datasets:
                run_logs.append(run_cmd(
                    ["python", "make_indexer.py", "--dataset_name", ds_name, "--is_sparse"],
                    dry_run=True,
                ))
        else:
            print(f"\n{'=' * 60}")
            print(f"PHASE 1: Parallel index building ({len(index_datasets)} datasets)")
            print(f"{'=' * 60}")
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=min(len(index_datasets), os.cpu_count() or 4)) as pool:
                futures = {pool.submit(_run_index, ds_name): ds_name for ds_name in index_datasets}
                for fut in as_completed(futures):
                    run_logs.append(fut.result())
            print(f"[DONE] All indices built in {time.time() - t0:.1f}s\n")

    # ── Phase 2: Parallel build (exp_rag) across GPUs ────────────────
    # Split datasets across GPUs. Each dataset's simple→none order is
    # preserved since they run sequentially within their GPU's thread.
    # The merged CSV is produced when 'none' finds 'simple's output.
    mid = (len(datasets_build) + 1) // 2  # e.g. [nq, hotpotqa] on gpu_a, [trivia] on gpu_b
    gpu_a_datasets = datasets_build[:mid]
    gpu_b_datasets = datasets_build[mid:]

    def _build_exp_rag_cmd(dataset_name, retr_method, tr_or_dev, steps_limit):
        return [
            "python", "exp_rag.py",
            "--retr_method", retr_method,
            "--is_sparse" if is_sparse else "",
            "--tr_or_dev", tr_or_dev,
            "--extracting_cot_qa", "--extract_sep",
            "--steps_limit", str(steps_limit),
            "--dataset_name", dataset_name,
            "--is_cot" if is_cot else "",
            "--sep_number", str(sep_number),
            "--model_id", model_id,
        ]

    # Threading event to stagger model loads — the first GPU signals
    # after its model is loaded (i.e. after the first subprocess has
    # been running for a short while), so the second GPU doesn't
    # compete for CPU RAM during the load phase.
    import threading
    _model_load_gate = threading.Event()

    def _run_build_for_datasets(ds_list, gpu, label, wait_for_gate=False):
        """Run build phase for a list of datasets on a specific GPU, sequentially."""
        if wait_for_gate:
            print(f"  [{label}] Waiting for first GPU to finish model loading...")
            _model_load_gate.wait()
            print(f"  [{label}] Gate open — starting model load on GPU:{gpu}")
        results = []
        first_job = True
        for dataset_name in ds_list:
            for retr_method in ["simple", "none"]:
                for tr_or_dev, sl in [("train", steps_train), ("dev", steps_dev)]:
                    r = run_cmd(
                        _build_exp_rag_cmd(dataset_name, retr_method, tr_or_dev, sl),
                        dry_run=False, gpu=gpu,
                    )
                    results.append(r)
                    if first_job and not wait_for_gate:
                        # First subprocess on the leading GPU has finished,
                        # meaning its model is loaded and on GPU.  Signal
                        # the second GPU that it is safe to start loading.
                        _model_load_gate.set()
                        first_job = False
            # Balance immediately after this dataset's build completes
            r = run_cmd(
                ["python", "balance_train_dataset.py", "--model_id", model_id, "--dataset_name", dataset_name],
                dry_run=False,
            )
            results.append(r)
        # Ensure the gate is set even if ds_list was empty
        _model_load_gate.set()
        return results

    if args.dry_run:
        for dataset_name in datasets_build:
            for retr_method in ["simple", "none"]:
                for tr_or_dev, sl in [("train", steps_train), ("dev", steps_dev)]:
                    run_logs.append(run_cmd(
                        _build_exp_rag_cmd(dataset_name, retr_method, tr_or_dev, sl),
                        dry_run=True,
                    ))
            run_logs.append(run_cmd(
                ["python", "balance_train_dataset.py", "--model_id", model_id, "--dataset_name", dataset_name],
                dry_run=True,
            ))
    else:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: Parallel build — GPU:{gpu_a} {gpu_a_datasets} | GPU:{gpu_b} {gpu_b_datasets}")
        print(f"{'=' * 60}")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_run_build_for_datasets, gpu_a_datasets, gpu_a, "GPU-A", wait_for_gate=False)
            fut_b = pool.submit(_run_build_for_datasets, gpu_b_datasets, gpu_b, "GPU-B", wait_for_gate=True)
            run_logs.extend(fut_a.result())
            run_logs.extend(fut_b.result())
        print(f"[DONE] Build phase completed in {time.time() - t0:.1f}s\n")

    # ── Phase 3: Training (sequential, last dataset wins checkpoint) ──
    # Checkpoints don't include dataset_name, so the last dataset's
    # training overwrites prior ones. Must match original order.
    print(f"\n{'=' * 60}") if not args.dry_run else None
    print(f"PHASE 3: Training probers (sequential, last-dataset-wins)") if not args.dry_run else None
    print(f"{'=' * 60}") if not args.dry_run else None
    if not args.dry_run:
        t0 = time.time()

    for dataset_name in datasets_build:
        # Split layers across 2 GPUs for the SAME dataset.
        layers = cfg["train"]["layers"]
        mid_l = (len(layers) + 1) // 2
        layers_a = layers[:mid_l]
        layers_b = layers[mid_l:]

        def _train_cmd(layer, gpu_device):
            return [
                "python", "train.py",
                "--method", cfg["train"]["method"],
                "--batch_size", str(cfg["train"]["batch_size"]),
                "--lr", str(cfg["train"]["lr"]),
                "--layer", str(layer),
                "--device", "cuda:0",  # always cuda:0 since CUDA_VISIBLE_DEVICES remaps
                "--epochs", str(cfg["train"]["epochs"]),
                "--model_id", model_id,
                "--dataset_name", dataset_name,
                "--train_ds_ratio", str(train_ratio),
            ]

        if args.dry_run:
            for layer in layers:
                run_logs.append(run_cmd(_train_cmd(layer, gpu_a), dry_run=True))
        else:
            # Launch layers in parallel across 2 GPUs.
            # Different layers write to different checkpoint files (l{layer} in path),
            # so no conflict between layers.
            _train_gate = threading.Event()

            def _run_train_layers(layer_list, gpu, wait_for_gate=False):
                if wait_for_gate:
                    _train_gate.wait()
                results = []
                first_job = True
                for layer in layer_list:
                    results.append(run_cmd(_train_cmd(layer, gpu), dry_run=False, gpu=gpu))
                    if first_job and not wait_for_gate:
                        _train_gate.set()
                        first_job = False
                _train_gate.set()  # ensure gate is set even if list was empty
                return results

            print(f"\n  Training {dataset_name}: layers {layers_a} on GPU:{gpu_a}, {layers_b} on GPU:{gpu_b}")
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(_run_train_layers, layers_a, gpu_a, wait_for_gate=False)
                fut_b = pool.submit(_run_train_layers, layers_b, gpu_b, wait_for_gate=True)
                run_logs.extend(fut_a.result())
                run_logs.extend(fut_b.result())

    if not args.dry_run:
        print(f"[DONE] Training completed in {time.time() - t0:.1f}s\n")

    # ── Phase 4: Parallel evaluation across GPUs ─────────────────────
    eval_datasets = cfg["evaluate"]["datasets"]
    eval_methods = cfg["evaluate"]["methods"]

    def _eval_cmd(dataset_name, retr_method):
        return [
            "python", "exp_rag.py",
            "--retr_method", retr_method,
            "--steps_limit", str(cfg["evaluate"]["steps_limit"]),
            "--dataset_name", dataset_name,
            "--tr_or_dev", eval_tr_or_dev,
            "--is_cot" if is_cot else "",
            "--is_sparse" if is_sparse else "",
            "--model_id", model_id,
            "--ds", str(ds),
            "--position", position,
            "--threshold", str(threshold),
            "--max_retrieval_rounds", str(eval_max_retrieval_rounds),
            "--extracting_cot_qa" if eval_extracting_cot_qa else "",
            "--extract_sep" if eval_extract_sep else "",
            "--sep_number", str(eval_sep_number),
        ]

    # Build list of (dataset, method) eval jobs
    eval_jobs = [(d, m) for d in eval_datasets for m in eval_methods]
    mid_e = (len(eval_jobs) + 1) // 2
    eval_jobs_a = eval_jobs[:mid_e]
    eval_jobs_b = eval_jobs[mid_e:]

    if args.dry_run:
        for d, m in eval_jobs:
            run_logs.append(run_cmd(_eval_cmd(d, m), dry_run=True))
    else:
        print(f"\n{'=' * 60}")
        print(f"PHASE 4: Parallel evaluation — {len(eval_jobs_a)} jobs on GPU:{gpu_a}, {len(eval_jobs_b)} jobs on GPU:{gpu_b}")
        print(f"{'=' * 60}")
        t0 = time.time()

        _eval_gate = threading.Event()

        def _run_eval_jobs(jobs, gpu, wait_for_gate=False):
            if wait_for_gate:
                _eval_gate.wait()
            results = []
            first_job = True
            for d, m in jobs:
                results.append(run_cmd(_eval_cmd(d, m), dry_run=False, gpu=gpu))
                if first_job and not wait_for_gate:
                    _eval_gate.set()
                    first_job = False
            _eval_gate.set()
            return results

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_run_eval_jobs, eval_jobs_a, gpu_a, wait_for_gate=False)
            fut_b = pool.submit(_run_eval_jobs, eval_jobs_b, gpu_b, wait_for_gate=True)
            run_logs.extend(fut_a.result())
            run_logs.extend(fut_b.result())
        print(f"[DONE] Evaluation completed in {time.time() - t0:.1f}s\n")

    # ── Report ───────────────────────────────────────────────────────
    for row in run_logs:
        if "command" in row and isinstance(row["command"], str):
            row["command"] = " ".join([p for p in row["command"].split(" ") if p])

    build_metrics = collect_metrics(model_id, datasets_build, ["train", "dev"], ["simple", "none"])
    eval_metrics = collect_method_metrics_from_result(cfg)
    skillrag_trace = collect_skillrag_trace(cfg)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "model_id": model_id,
        "model_short": model_short,
        "gpus": gpus,
        "build_simple_none_metrics": build_metrics,
        "evaluation_metrics": eval_metrics,
        "skillrag_traces": skillrag_trace,
        "commands": run_logs,
    }

    os.makedirs("reports", exist_ok=True)
    report_name = f"reports/{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_parallel.yaml"
    if yaml is None:
        with open(report_name, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    else:
        with open(report_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(f"Saved report: {report_name}")


if __name__ == "__main__":
    main()
