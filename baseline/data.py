from __future__ import annotations

import bz2
import csv
import gzip
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm import tqdm

from baseline.common import QAExample, RAW_DATA_ROOT, ensure_dir


DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia"]


def _stream_download(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    if output_path.exists():
        return
    ensure_dir(output_path.parent)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def _download_google_drive(file_id: str, output_path: Path) -> None:
    if output_path.exists():
        return
    ensure_dir(output_path.parent)
    import gdown

    gdown.download(id=file_id, output=str(output_path), quiet=False)


def _gunzip(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    ensure_dir(dest.parent)
    with gzip.open(src, "rb") as gz, dest.open("wb") as out:
        shutil.copyfileobj(gz, out)


def _unzip(src: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with zipfile.ZipFile(src, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            if not filename or filename == ".DS_Store":
                continue
            target = dest_dir / filename
            if target.exists():
                continue
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def _extract_tar_bz2(src: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with tarfile.open(src, "r:bz2") as archive:
        archive.extractall(dest_dir, filter="data")


def download_nq(root: Path = RAW_DATA_ROOT) -> None:
    dataset_dir = ensure_dir(root / "nq")
    dev_gz = dataset_dir / "biencoder-nq-dev.json.gz"
    train_gz = dataset_dir / "biencoder-nq-train.json.gz"
    _stream_download(
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz",
        dev_gz,
    )
    _stream_download(
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        train_gz,
    )
    _gunzip(dev_gz, dataset_dir / "biencoder-nq-dev.json")
    _gunzip(train_gz, dataset_dir / "biencoder-nq-train.json")


def download_trivia(root: Path = RAW_DATA_ROOT) -> None:
    dataset_dir = ensure_dir(root / "trivia")
    dev_gz = dataset_dir / "biencoder-trivia-dev.json.gz"
    train_gz = dataset_dir / "biencoder-trivia-train.json.gz"
    _stream_download(
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz",
        dev_gz,
    )
    _stream_download(
        "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz",
        train_gz,
    )
    _gunzip(dev_gz, dataset_dir / "biencoder-trivia-dev.json")
    _gunzip(train_gz, dataset_dir / "biencoder-trivia-train.json")


def download_hotpotqa(root: Path = RAW_DATA_ROOT) -> None:
    dataset_dir = ensure_dir(root / "hotpotqa")
    _stream_download(
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        dataset_dir / "hotpot_train_v1.1.json",
    )
    _stream_download(
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        dataset_dir / "hotpot_dev_distractor_v1.json",
    )
    tar_path = dataset_dir / "wikipedia-paragraphs.tar.bz2"
    _stream_download(
        "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        tar_path,
    )
    extracted = dataset_dir / "enwiki-20171001-pages-meta-current-withlinks-abstracts"
    wiki_root = dataset_dir / "wikpedia-paragraphs"
    if extracted.exists() and not any(extracted.iterdir()) and not wiki_root.exists():
        extracted.rmdir()
    if not extracted.exists() and not wiki_root.exists():
        _extract_tar_bz2(tar_path, dataset_dir)
    if extracted.exists() and not wiki_root.exists():
        extracted.rename(wiki_root)


def download_2wikimultihopqa(root: Path = RAW_DATA_ROOT) -> None:
    dataset_dir = ensure_dir(root / "2wikimultihopqa")
    archive_path = dataset_dir / "2wikimultihopqa.zip"
    _stream_download(
        "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1",
        archive_path,
    )
    _unzip(archive_path, dataset_dir)


def download_musique(root: Path = RAW_DATA_ROOT) -> None:
    dataset_dir = ensure_dir(root / "musique")
    archive_path = dataset_dir / "musique_v1.0.zip"
    _download_google_drive("1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h", archive_path)
    _unzip(archive_path, dataset_dir)


def download_all(root: Path = RAW_DATA_ROOT) -> None:
    download_nq(root)
    download_trivia(root)
    download_hotpotqa(root)
    download_2wikimultihopqa(root)
    download_musique(root)


def load_examples(dataset_name: str, split: str = "dev", limit: int | None = None, root: Path = RAW_DATA_ROOT) -> List[QAExample]:
    if dataset_name == "hotpotqa":
        filename = "hotpot_dev_distractor_v1.json" if split == "dev" else "hotpot_train_v1.1.json"
        data = json.loads((root / "hotpotqa" / filename).read_text(encoding="utf-8"))
        examples = [QAExample(question=item["question"], answers=[str(item["answer"])], metadata={"id": item.get("_id")}) for item in data]
    elif dataset_name == "nq":
        data = json.loads((root / "nq" / f"biencoder-nq-{split}.json").read_text(encoding="utf-8"))
        examples = [QAExample(question=item["question"], answers=[str(ans) for ans in item["answers"]], metadata={"id": idx}) for idx, item in enumerate(data)]
    elif dataset_name == "trivia":
        data = json.loads((root / "trivia" / f"biencoder-trivia-{split}.json").read_text(encoding="utf-8"))
        examples = [QAExample(question=item["question"], answers=[str(ans) for ans in item["answers"]], metadata={"id": idx}) for idx, item in enumerate(data)]
    elif dataset_name == "2wikimultihopqa":
        data = json.loads((root / "2wikimultihopqa" / f"{split}.json").read_text(encoding="utf-8"))
        examples = [QAExample(question=item["question"], answers=[str(item["answer"])], metadata={"id": item.get("_id")}) for item in data]
    elif dataset_name == "musique":
        path = root / "musique" / f"musique_full_v1.0_{split}.jsonl"
        examples = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                examples.append(QAExample(question=item["question"], answers=[str(item["answer"])], metadata={"id": item.get("id", idx)}))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return examples[:limit] if limit is not None else examples


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _extract_ctx_texts(data: list, keys: tuple = ("positive_ctxs", "negative_ctxs", "hard_negative_ctxs")) -> List[str]:
    texts: List[str] = []
    for item in data:
        for key in keys:
            texts.extend(ctx["text"] for ctx in item.get(key, []) if ctx.get("text"))
    return texts


def corpus_texts_for_dataset(dataset_name: str, root: Path = RAW_DATA_ROOT) -> List[str]:
    if dataset_name == "hotpotqa":
        wiki_root = root / "hotpotqa" / "wikpedia-paragraphs"
        texts: List[str] = []
        for filepath in sorted(wiki_root.glob("*/*.bz2")):
            with bz2.open(filepath, "rb") as handle:
                for raw in handle:
                    item = json.loads(raw.decode("utf-8").strip())
                    paragraph = " ".join(sentence.strip() for sentence in item["text"]).strip()
                    if paragraph:
                        texts.append(paragraph)
        return _unique_preserve_order(texts)

    if dataset_name == "2wikimultihopqa":
        texts = []
        for split in ("train", "dev", "test"):
            path = root / "2wikimultihopqa" / f"{split}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                for title, sentences in item["context"]:
                    paragraph = " ".join(sentences).strip()
                    if paragraph:
                        texts.append(paragraph)
        return _unique_preserve_order(texts)

    if dataset_name == "musique":
        texts = []
        files = [
            "musique_ans_v1.0_dev.jsonl",
            "musique_ans_v1.0_test.jsonl",
            "musique_ans_v1.0_train.jsonl",
            "musique_full_v1.0_dev.jsonl",
            "musique_full_v1.0_test.jsonl",
            "musique_full_v1.0_train.jsonl",
        ]
        for filename in files:
            filepath = root / "musique" / filename
            if not filepath.exists():
                continue
            with filepath.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    for paragraph in item.get("paragraphs", []):
                        text = paragraph["paragraph_text"].strip()
                        if text:
                            texts.append(text)
        return _unique_preserve_order(texts)

    if dataset_name == "nq":
        texts = []
        for split in ("dev", "train"):
            path = root / "nq" / f"biencoder-nq-{split}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            texts.extend(_extract_ctx_texts(data))
        return _unique_preserve_order(texts)

    if dataset_name == "trivia":
        texts = []
        for split in ("train", "dev"):
            path = root / "trivia" / f"biencoder-trivia-{split}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            texts.extend(_extract_ctx_texts(data))
        return _unique_preserve_order(texts)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def write_corpus_csv(dataset_name: str, texts: List[str], root: Path = RAW_DATA_ROOT) -> Path:
    output_path = root / f"{dataset_name}_index_2.csv"
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc", "doc_id"])
        for idx, text in enumerate(texts):
            writer.writerow([text, idx])
    return output_path
