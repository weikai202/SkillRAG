from quick_check_dataset_format import check_file
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    check_file(root / "raw_data" / "nq" / "biencoder-nq-dev.json")
    check_file(root / "raw_data" / "nq" / "biencoder-nq-train.json")
    print("[DONE] dataset format is valid. (legacy entry: quick_check_tiny_dataset.py)")
