from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from baseline.baselines import BaselineRunner
from baseline.common import (
    EvalRow,
    RESULTS_ROOT,
    RAW_DATA_ROOT,
    aggregate_rows,
    compute_acc,
    ensure_dir,
    format_model_name,
    max_metric_over_ground_truths,
    write_json,
)
from baseline.data import DATASETS, load_examples
from baseline.generation import HFGenerator
from baseline.indexing import load_bm25_retriever


def compute_f1(gold: str, pred: str) -> float:
    from baseline.common import token_f1

    return token_f1(gold, pred)


def compute_em(gold: str, pred: str) -> float:
    from baseline.common import compute_exact

    return float(compute_exact(gold, pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated BM25 baselines on dev/train splits.")
    parser.add_argument("--method", required=True, choices=["adaptive", "dragin", "flare"])
    parser.add_argument("--dataset_name", required=True, choices=DATASETS)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flare_confidence_threshold", type=float, default=-2.5)
    parser.add_argument("--dragin_entropy_threshold", type=float, default=2.5)
    args = parser.parse_args()

    examples = load_examples(args.dataset_name, split=args.split, limit=args.limit, root=RAW_DATA_ROOT)
    retriever = load_bm25_retriever(args.dataset_name, root=RAW_DATA_ROOT, top_k=args.top_k)
    generator = HFGenerator(args.model_id, device=args.device)
    runner = BaselineRunner(
        generator=generator,
        retriever=retriever,
        top_k=args.top_k,
        max_rounds=args.max_rounds,
        flare_confidence_threshold=args.flare_confidence_threshold,
        dragin_entropy_threshold=args.dragin_entropy_threshold,
    )

    rows: list[EvalRow] = []
    for example in tqdm(examples, desc=f"{args.method}-{args.dataset_name}"):
        trace = runner.run(args.method, example.question)
        em = max_metric_over_ground_truths(compute_em, trace.answer, example.answers)
        f1 = max_metric_over_ground_truths(compute_f1, trace.answer, example.answers)
        acc = compute_acc(trace.answer, example.answers)
        rows.append(
            EvalRow(
                dataset_name=args.dataset_name,
                method=args.method,
                question=example.question,
                answers=example.answers,
                prediction=trace.answer,
                raw_output=trace.raw_output,
                acc=acc,
                em=em,
                f1=f1,
                retrieval_count=trace.retrieval_count,
                trace={
                    "queries": trace.queries,
                    "passages": trace.passages,
                    "steps": trace.steps,
                },
            )
        )

    summary = aggregate_rows(rows)
    summary.update(
        {
            "dataset_name": args.dataset_name,
            "method": args.method,
            "model_id": args.model_id,
            "split": args.split,
            "limit": args.limit,
            "top_k": args.top_k,
            "max_rounds": args.max_rounds,
        }
    )

    model_dir = ensure_dir(RESULTS_ROOT / format_model_name(args.model_id))
    stem = f"{args.method}_{args.dataset_name}_{args.split}_{args.limit}"
    rows_path = model_dir / f"{stem}.csv"
    summary_path = model_dir / f"{stem}.summary.json"
    pd.DataFrame([row.to_record() for row in rows]).to_csv(rows_path, index=False)
    write_json(summary, summary_path)

    print(summary_path)
    print(rows_path)


if __name__ == "__main__":
    main()
