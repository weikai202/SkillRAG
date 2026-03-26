from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from baseline.common import RESULTS_ROOT, ensure_dir, format_model_name


DATASETS = ["hotpotqa", "nq", "trivia", "musique", "2wikimultihopqa"]
METHODS = ["flare", "dragin", "adaptive"]
MODELS = [
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-8B",
    "unsloth/gemma-2-9b-it",
]


def load_summary(model_id: str, method: str, dataset_name: str, split: str = "dev", limit: int = 500) -> dict | None:
    model_dir = RESULTS_ROOT / format_model_name(model_id)
    path = model_dir / f"{method}_{dataset_name}_{split}_{limit}.summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows: list[dict] = []
    for model_id in MODELS:
        for method in METHODS:
            dataset_summaries = []
            missing_datasets = []
            for dataset_name in DATASETS:
                summary = load_summary(model_id, method, dataset_name)
                if summary is None:
                    missing_datasets.append(dataset_name)
                    rows.append(
                        {
                            "model_id": model_id,
                            "method": method,
                            "dataset_name": dataset_name,
                            "em": None,
                            "acc": None,
                            "count": None,
                            "avg_retrieval_count": None,
                        }
                    )
                    continue
                rows.append(
                    {
                        "model_id": model_id,
                        "method": method,
                        "dataset_name": dataset_name,
                        "em": summary["em"],
                        "acc": summary["acc"],
                        "count": summary["count"],
                        "avg_retrieval_count": summary["avg_retrieval_count"],
                    }
                )
                dataset_summaries.append(summary)

            if len(dataset_summaries) == len(DATASETS):
                rows.append(
                    {
                        "model_id": model_id,
                        "method": method,
                        "dataset_name": "average",
                        "em": round(sum(item["em"] for item in dataset_summaries) / len(dataset_summaries), 4),
                        "acc": round(sum(item["acc"] for item in dataset_summaries) / len(dataset_summaries), 4),
                        "count": sum(item["count"] for item in dataset_summaries),
                        "avg_retrieval_count": round(
                            sum(item["avg_retrieval_count"] for item in dataset_summaries) / len(dataset_summaries),
                            4,
                        ),
                        "complete": True,
                    }
                )
            elif dataset_summaries:
                rows.append(
                    {
                        "model_id": model_id,
                        "method": method,
                        "dataset_name": "average_incomplete",
                        "em": None,
                        "acc": None,
                        "count": sum(item["count"] for item in dataset_summaries),
                        "avg_retrieval_count": None,
                        "complete": False,
                        "missing_datasets": ",".join(missing_datasets),
                    }
                )

    ensure_dir(RESULTS_ROOT)
    output_path = RESULTS_ROOT / "paper_baseline_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
