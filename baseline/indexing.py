from __future__ import annotations

from pathlib import Path
from typing import List

from baseline.common import RAW_DATA_ROOT, ensure_dir
from baseline.data import corpus_texts_for_dataset, write_corpus_csv


def sparse_index_path(dataset_name: str, root: Path = RAW_DATA_ROOT) -> Path:
    return root / "sparse_index" / f"llama_index_bm25_model_{dataset_name}_2.json"


def build_sparse_index(dataset_name: str, root: Path = RAW_DATA_ROOT) -> Path:
    from llama_index.core import Document
    from llama_index.core.storage.docstore import SimpleDocumentStore

    texts = corpus_texts_for_dataset(dataset_name, root=root)
    write_corpus_csv(dataset_name, texts, root=root)

    documents = [Document(text=text, doc_id=str(idx)) for idx, text in enumerate(texts)]
    docstore = SimpleDocumentStore()
    docstore.add_documents(documents)

    output_path = sparse_index_path(dataset_name, root=root)
    ensure_dir(output_path.parent)
    docstore.persist(output_path)
    return output_path


def load_bm25_retriever(dataset_name: str, root: Path = RAW_DATA_ROOT, top_k: int = 5) -> BM25Retriever:
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.retrievers.bm25 import BM25Retriever

    docstore = SimpleDocumentStore.from_persist_path(str(sparse_index_path(dataset_name, root=root)))
    return BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=top_k)


def retrieve_texts(retriever: BM25Retriever, query: str) -> List[str]:
    return [node.text for node in retriever.retrieve(query)]
