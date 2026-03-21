import json
from pathlib import Path


def check_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    assert isinstance(data, list), f"{path} must be a JSON list"
    assert len(data) == 5, f"{path} should contain 5 queries, got {len(data)}"
    required = {"question", "answers", "positive_ctxs", "negative_ctxs", "hard_negative_ctxs"}
    for i, item in enumerate(data):
        miss = required - set(item.keys())
        assert not miss, f"{path} item#{i} missing keys: {sorted(miss)}"
        assert isinstance(item["answers"], list) and len(item["answers"]) > 0
    print(f"[OK] {path} -> {len(data)} samples")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    check_file(root / "raw_data" / "nq" / "biencoder-nq-dev.json")
    check_file(root / "raw_data" / "nq" / "biencoder-nq-train.json")
    print("[DONE] tiny nq dataset format is valid.")
