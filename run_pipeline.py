import argparse
import ast
import glob
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

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
        raise RuntimeError("PyYAML is required. Please install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: List[str], dry_run: bool) -> Dict[str, Any]:
    cmd = [c for c in cmd if c]
    cmd_str = " ".join(cmd)
    if dry_run:
        print("[DRY-RUN]", cmd_str)
        return {"command": cmd_str, "returncode": 0}
    print("[RUN]", cmd_str)
    proc = subprocess.run(cmd, check=False)
    return {"command": cmd_str, "returncode": proc.returncode}


def latest_file(pattern: str) -> str:
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else ""


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
                rows.append(
                    {
                        "model_id": model_id,
                        "dataset_name": dataset_name,
                        "retr_method": method,
                        "acc": None,
                        "em": None,
                        "file": path,
                    }
                )
                continue
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict() if len(df) else {}
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_name": dataset_name,
                    "retr_method": method,
                    "acc": float(row.get("acc")) if "acc" in row else None,
                    "em": float(row.get("em")) if "em" in row else None,
                    "file": path,
                }
            )
    return rows


def collect_skillrag_trace(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_id = cfg["model"]["id"]
    model_short = model_id.split("/")[-1]
    save_dir = MODEL_TO_DIR[model_id]
    datasets = cfg["evaluate"]["datasets"]
    out: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        pattern = f"dataset/{save_dir}/retrieval_qa_{model_short}_{dataset_name}_skillrag_dev_*.csv"
        path = latest_file(pattern)
        if not path:
            continue
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
                        trace = []
            out.append(
                {
                    "dataset_name": dataset_name,
                    "question_with_prompt": row.get("question_with_prompt", ""),
                    "pred": row.get("pred", ""),
                    "answer": row.get("answer", ""),
                    "acc": row.get("acc", None),
                    "em": row.get("em", None),
                    "round_logs": trace,  # includes each-step prober_scores/logits
                    "file": path,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

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
    device = cfg["train"]["device"]
    train_ratio = cfg["train"]["train_ds_ratio"]

    run_logs: List[Dict[str, Any]] = []

    if build_index:
        for dataset_name in index_datasets:
            run_logs.append(
                run_cmd(
                    ["python", "make_indexer.py", "--dataset_name", dataset_name, "--is_sparse"],
                    args.dry_run,
                )
            )

    for dataset_name in datasets_build:
        for retr_method in ["simple", "none"]:
            run_logs.append(
                run_cmd(
                    [
                        "python",
                        "exp_rag.py",
                        "--retr_method",
                        retr_method,
                        "--is_sparse" if is_sparse else "",
                        "--tr_or_dev",
                        "train",
                        "--extracting_cot_qa",
                        "--extract_sep",
                        "--steps_limit",
                        str(steps_train),
                        "--dataset_name",
                        dataset_name,
                        "--is_cot" if is_cot else "",
                        "--sep_number",
                        str(sep_number),
                        "--model_id",
                        model_id,
                    ],
                    args.dry_run,
                )
            )
            run_logs.append(
                run_cmd(
                    [
                        "python",
                        "exp_rag.py",
                        "--retr_method",
                        retr_method,
                        "--is_sparse" if is_sparse else "",
                        "--tr_or_dev",
                        "dev",
                        "--extracting_cot_qa",
                        "--extract_sep",
                        "--steps_limit",
                        str(steps_dev),
                        "--dataset_name",
                        dataset_name,
                        "--is_cot" if is_cot else "",
                        "--sep_number",
                        str(sep_number),
                        "--model_id",
                        model_id,
                    ],
                    args.dry_run,
                )
            )

        run_logs.append(
            run_cmd(
                [
                    "python",
                    "balance_train_dataset.py",
                    "--model_id",
                    model_id,
                    "--dataset_name",
                    dataset_name,
                ],
                args.dry_run,
            )
        )

        for layer in cfg["train"]["layers"]:
            run_logs.append(
                run_cmd(
                    [
                        "python",
                        "train.py",
                        "--method",
                        cfg["train"]["method"],
                        "--batch_size",
                        str(cfg["train"]["batch_size"]),
                        "--lr",
                        str(cfg["train"]["lr"]),
                        "--layer",
                        str(layer),
                        "--device",
                        device,
                        "--epochs",
                        str(cfg["train"]["epochs"]),
                        "--model_id",
                        model_id,
                        "--dataset_name",
                        dataset_name,
                        "--train_ds_ratio",
                        str(train_ratio),
                    ],
                    args.dry_run,
                )
            )

    for dataset_name in cfg["evaluate"]["datasets"]:
        for retr_method in cfg["evaluate"]["methods"]:
            run_logs.append(
                run_cmd(
                    [
                        "python",
                        "exp_rag.py",
                        "--retr_method",
                        retr_method,
                        "--steps_limit",
                        str(cfg["evaluate"]["steps_limit"]),
                        "--dataset_name",
                        dataset_name,
                        "--tr_or_dev",
                        eval_tr_or_dev,
                        "--is_cot" if is_cot else "",
                        "--is_sparse" if is_sparse else "",
                        "--model_id",
                        model_id,
                        "--ds",
                        str(ds),
                        "--position",
                        position,
                        "--threshold",
                        str(threshold),
                        "--max_retrieval_rounds",
                        str(eval_max_retrieval_rounds),
                        "--extracting_cot_qa" if eval_extracting_cot_qa else "",
                        "--extract_sep" if eval_extract_sep else "",
                        "--sep_number",
                        str(eval_sep_number),
                    ],
                    args.dry_run,
                )
            )

    # drop empty args that come from conditional flags
    for row in run_logs:
        if "command" in row:
            row["command"] = " ".join([p for p in row["command"].split(" ") if p])

    build_metrics = collect_metrics(model_id, datasets_build, ["train", "dev"], ["simple", "none"])
    eval_metrics = collect_method_metrics_from_result(cfg)
    skillrag_trace = collect_skillrag_trace(cfg)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
        "model_id": model_id,
        "model_short": model_short,
        "build_simple_none_metrics": build_metrics,  # list as requested
        "evaluation_metrics": eval_metrics,  # acc/em by method
        "skillrag_traces": skillrag_trace,  # query/reasoning/answer + step prober_scores
        "commands": run_logs,
    }

    if not os.path.exists("reports"):
        os.makedirs("reports")
    report_name = f"reports/{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    if yaml is None:
        # Fallback to json text with yaml extension when PyYAML is unavailable.
        with open(report_name, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    else:
        with open(report_name, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    print(f"Saved report: {report_name}")


if __name__ == "__main__":
    main()
