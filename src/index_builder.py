#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, cfg, keep_tables=True, do_visualize=False)
"""

import os
import pickle
import pathlib
import re
from typing import List, Dict, Any

import faiss
from rank_bm25 import BM25Okapi

from src.embedder import SentenceTransformer
from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.config import QueryPlanConfig  # kept for compatibility / future use

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ["questions", "exercises", "summary", "references"]


# ------------------------ Main index builder -----------------------------


def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    do_visualize: bool = False,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
    """
    all_chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict[str, Any]] = []

    # Extract sections from markdown. Exclude some with certain
    # keywords if required.
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS,
    )

    # Step 1: Chunk using DocumentChunker
    for section_idx, sec in enumerate(sections):
        content = sec["content"]
        heading = sec.get("heading", None)

        # Try to capture page-ish info if extraction provides it
        page = sec.get("page") or sec.get("page_num")
        has_table = bool(TABLE_RE.search(content))

        base_meta: Dict[str, Any] = {
            "filename": markdown_file,
            "section_index": section_idx,
            "section": heading,
            "mode": chunk_config.to_string(),
            "keep_tables": chunker.keep_tables,
            "char_len": len(content),
            "word_len": len(content.split()),
            "has_table": has_table,
            "text_preview": content[:100],
        }
        if page is not None:
            base_meta["page"] = page

        # Use DocumentChunker to split this section into smaller chunks
        sub_chunks = chunker.chunk(content)

        for local_idx, sub_c in enumerate(sub_chunks):
            # store the text chunk
            all_chunks.append(sub_c)
            sources.append(markdown_file)

            # clone metadata and add per-subchunk info
            m = dict(base_meta)
            m["subchunk_index"] = local_idx
            m["global_chunk_index"] = len(all_chunks) - 1
            metadata.append(m)

    # Step 2: Create embeddings for FAISS index
    print(
        f"Embedding {len(all_chunks):,} chunks with "
        f"{pathlib.Path(embedding_model_path).stem} ..."
    )
    embedder = SentenceTransformer(embedding_model_path)
    embeddings = embedder.encode(
        all_chunks,
        batch_size=4,
        show_progress_bar=True,
    )

    # Step 3: Build FAISS index
    print(f"Building FAISS index for {len(all_chunks):,} chunks...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(pathlib.Path(artifacts_dir) / f"{index_prefix}.faiss"))
    print(f"FAISS index built successfully: {index_prefix}.faiss")

    # Step 4: Build BM25 index
    print(f"Building BM25 index for {len(all_chunks):,} chunks...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"BM25 index built successfully: {index_prefix}_bm25.pkl")

    # Step 5: Dump index artifacts
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(pathlib.Path(artifacts_dir) / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

    # Step 6: Optional visualization
    if do_visualize:
        visualize(embeddings, sources)


# ------------------------ Helper functions ------------------------------


def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    # Convert to lowercase
    text = text.lower()

    # Keep only allowed characters
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)

    # Split by whitespace
    tokens = text.split()

    return tokens


def visualize(embeddings, sources):
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        red = PCA(n_components=2).fit_transform(embeddings)
        uniq = sorted(set(sources))
        cmap = {s: i for i, s in enumerate(uniq)}
        colors = [cmap[s] for s in sources]

        plt.figure(figsize=(10, 7))
        sc = plt.scatter(red[:, 0], red[:, 1], c=colors, cmap="tab10", alpha=0.55)
        plt.title("Vector index (PCA)")
        plt.legend(
            handles=sc.legend_elements()[0],
            labels=uniq,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[visualize] skipped ({e})")
