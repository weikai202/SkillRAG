from __future__ import annotations

import json
import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = BASELINE_ROOT / "raw_data"
RESULTS_ROOT = BASELINE_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def format_model_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def detect_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def best_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(text: str) -> str:
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def token_f1(gold: str, pred: str) -> float:
    from collections import Counter

    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)
    gold_counter = Counter(gold_toks)
    pred_counter = Counter(pred_toks)
    overlap = sum((gold_counter & pred_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_metric_over_ground_truths(metric_fn, prediction: str, answers: Sequence[str]) -> float:
    if not answers:
        return 0.0
    return max(metric_fn(answer, prediction) for answer in answers)


def compute_acc(prediction: str, answers: Sequence[str]) -> int:
    pred_norm = normalize_answer(prediction)
    return int(any(normalize_answer(answer) in pred_norm for answer in answers))


def _strip_special_tags(text: str) -> str:
    text = text.replace("</s>", "").replace("<eos>", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_answer_text(text: str) -> str:
    cleaned = _strip_special_tags(text)
    if "Answer:" in cleaned:
        after = cleaned.rsplit("Answer:", 1)[-1].strip()
        lines = after.splitlines()
        return lines[0].strip() if lines else after
    if "\n" in cleaned:
        lines = cleaned.strip().splitlines()
        return lines[-1].strip() if lines else cleaned.strip()
    return cleaned


def clean_search_query(text: str, fallback: str) -> str:
    cleaned = _strip_special_tags(text)
    lines = [line.strip(" -") for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return fallback
    for prefix in ("Search Query:", "Query:", "Question:"):
        for line in lines:
            if line.startswith(prefix):
                candidate = line.split(":", 1)[-1].strip()
                return candidate or fallback
    return lines[0]


def split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", stripped)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


@dataclass
class QAExample:
    question: str
    answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineTrace:
    method: str
    raw_output: str
    answer: str
    retrieval_count: int
    queries: List[str] = field(default_factory=list)
    passages: List[List[str]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalRow:
    dataset_name: str
    method: str
    question: str
    answers: List[str]
    prediction: str
    raw_output: str
    acc: int
    em: float
    f1: float
    retrieval_count: int
    trace: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "method": self.method,
            "question": self.question,
            "answers": json_dumps(self.answers),
            "prediction": self.prediction,
            "raw_output": self.raw_output,
            "acc": self.acc,
            "em": self.em,
            "f1": self.f1,
            "retrieval_count": self.retrieval_count,
            "trace": json_dumps(self.trace),
        }


def aggregate_rows(rows: Sequence[EvalRow]) -> Dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "acc": round(sum(row.acc for row in rows) / count, 4) if count else 0.0,
        "em": round(sum(row.em for row in rows) / count, 4) if count else 0.0,
        "f1": round(sum(row.f1 for row in rows) / count, 4) if count else 0.0,
        "avg_retrieval_count": round(sum(row.retrieval_count for row in rows) / count, 4) if count else 0.0,
    }


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(entropy.item())

