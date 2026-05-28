from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "pdf"

for module_dir in (
    PROJECT_ROOT / "src" / "chatbot",
    PROJECT_ROOT / "src" / "indexing" / "ingestion",
    PROJECT_ROOT / "src" / "indexing" / "chunking",
    PROJECT_ROOT / "src" / "indexing" / "embedding",
):
    sys.path.insert(0, str(module_dir))

import chat as chat_service  # noqa: E402
import chunker as chunker_service  # noqa: E402
import embed_and_index as indexing_service  # noqa: E402
import parse_pdf as pdf_parser  # noqa: E402
import retrieval as retrieval_service  # noqa: E402


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename(name: str) -> str:
    safe_name = SAFE_FILENAME_PATTERN.sub("_", Path(name).name)
    return safe_name.strip("._") or "document.pdf"


def save_uploaded_files(uploaded_files: list[Any]) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        file_path = RAW_DIR / safe_filename(uploaded_file.name)
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def runtime_args(
    env_file: Path,
    region: str | None,
    embedding_model_id: str | None,
    dimensions: int | None,
    qdrant_url: str | None,
    collection_name: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        env_file=env_file,
        region=region,
        embedding_model_id=embedding_model_id,
        dimensions=dimensions,
        qdrant_url=qdrant_url,
        collection=collection_name,
    )


def process_documents(
    env_file: Path,
    region: str | None,
    embedding_model_id: str | None,
    dimensions: int | None,
    qdrant_url: str | None,
    collection_name: str | None,
    strategy: str,
    extract_images: bool,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    recreate_collection: bool,
    limit: int | None,
) -> dict[str, Any]:
    indexing_service.load_env_file(env_file)
    args = runtime_args(
        env_file=env_file,
        region=region,
        embedding_model_id=embedding_model_id,
        dimensions=dimensions,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
    )
    config = retrieval_service.load_runtime_config(args)

    written_files = pdf_parser.parse_raw_pdfs(
        input_dir=RAW_DIR,
        output_dir=PARSED_DIR,
        strategy=strategy,
        extract_images=extract_images,
    )
    input_files = chunker_service.find_document_inputs(PARSED_DIR)
    chunk_paths = [
        chunker_service.chunk_file(
            input_path=input_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for input_path in input_files
    ]

    bedrock_client = indexing_service.build_bedrock_client(config["region"])
    qdrant_client = indexing_service.build_qdrant_client(
        config["qdrant_url"],
        api_key=config["qdrant_api_key"],
    )

    index_results = []
    for index, chunks_path in enumerate(chunk_paths):
        index_results.append(
            indexing_service.upsert_chunks(
                chunks_path=chunks_path,
                qdrant_client=qdrant_client,
                bedrock_client=bedrock_client,
                collection_name=config["collection_name"],
                embedding_model_id=config["embedding_model_id"],
                dimensions=config["dimensions"],
                batch_size=batch_size,
                limit=limit,
                recreate_collection=recreate_collection and index == 0,
            ),
        )

    return {
        "parsed_files": [str(path) for path in written_files],
        "chunk_files": [str(path) for path in chunk_paths],
        "index_results": index_results,
        "collection_name": config["collection_name"],
    }


def answer_question(
    question: str,
    env_file: Path,
    region: str | None,
    embedding_model_id: str | None,
    dimensions: int | None,
    qdrant_url: str | None,
    collection_name: str | None,
    chat_model_id: str | None,
    top_k: int,
    score_threshold: float | None,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    retrieval_service.load_env_file(env_file)
    args = runtime_args(
        env_file=env_file,
        region=region,
        embedding_model_id=embedding_model_id,
        dimensions=dimensions,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
    )
    config = retrieval_service.load_runtime_config(args)
    resolved_chat_model_id = (
        chat_model_id
        or os.environ.get("CHAT_MODEL_ID")
        or os.environ.get("MODEL_ID")
        or chat_service.DEFAULT_CHAT_MODEL_ID
    )

    bedrock_client = retrieval_service.build_bedrock_client(config["region"])
    qdrant_client = retrieval_service.build_qdrant_client(
        config["qdrant_url"],
        config["qdrant_api_key"],
    )
    results = retrieval_service.search_chunks(
        query=question,
        bedrock_client=bedrock_client,
        qdrant_client=qdrant_client,
        collection_name=config["collection_name"],
        embedding_model_id=config["embedding_model_id"],
        dimensions=config["dimensions"],
        top_k=top_k,
        score_threshold=score_threshold,
    )
    context = retrieval_service.format_results_for_prompt(results)
    answer = chat_service.generate_answer(
        question=question,
        context=context,
        bedrock_client=bedrock_client,
        model_id=resolved_chat_model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return {
        "answer": answer,
        "sources": results,
        "source_text": chat_service.format_sources(results),
        "chat_model_id": resolved_chat_model_id,
        "embedding_model_id": config["embedding_model_id"],
        "collection_name": config["collection_name"],
    }


def optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


st.set_page_config(page_title="Intelldoc", layout="wide")
st.title("Intelldoc")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Runtime")
    env_file = Path(st.text_input("Env file", str(retrieval_service.DEFAULT_ENV_FILE)))
    region = optional_text(st.text_input("AWS region", ""))
    qdrant_url = optional_text(
        st.text_input("Qdrant URL", retrieval_service.DEFAULT_QDRANT_URL),
    )
    collection_name = optional_text(
        st.text_input("Collection", retrieval_service.DEFAULT_COLLECTION_NAME),
    )
    embedding_model_id = optional_text(
        st.text_input("Embedding model", retrieval_service.DEFAULT_EMBEDDING_MODEL_ID),
    )
    dimensions = st.number_input(
        "Embedding dimensions",
        min_value=1,
        value=retrieval_service.DEFAULT_EMBEDDING_DIMENSIONS,
        step=1,
    )

    st.header("Chat")
    chat_model_id = optional_text(st.text_input("Chat model", chat_service.DEFAULT_CHAT_MODEL_ID))
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=retrieval_service.DEFAULT_TOP_K)
    use_score_threshold = st.checkbox("Use score threshold", value=False)
    score_threshold = (
        st.number_input("Score threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        if use_score_threshold
        else None
    )
    max_tokens = st.number_input(
        "Max tokens",
        min_value=1,
        value=chat_service.DEFAULT_MAX_TOKENS,
        step=50,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

tab_chat, tab_documents = st.tabs(["Chat", "Documents"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    st.text(message["sources"])

    question = st.chat_input("Ask a question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    response = answer_question(
                        question=question,
                        env_file=env_file,
                        region=region,
                        embedding_model_id=embedding_model_id,
                        dimensions=int(dimensions),
                        qdrant_url=qdrant_url,
                        collection_name=collection_name,
                        chat_model_id=chat_model_id,
                        top_k=int(top_k),
                        score_threshold=score_threshold,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.markdown(response["answer"])
                    with st.expander("Sources"):
                        st.text(response["source_text"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["source_text"],
                        },
                    )

with tab_documents:
    uploaded_files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        strategy = st.selectbox("Parse strategy", ["hi_res", "auto", "fast", "ocr_only"])
        extract_images = st.checkbox("Extract images", value=True)
        recreate_collection = st.checkbox("Reset collection before indexing", value=False)
    with col_right:
        chunk_size = st.number_input("Chunk size", min_value=100, value=1000, step=100)
        overlap = st.number_input("Chunk overlap", min_value=0, value=200, step=50)
        batch_size = st.number_input("Index batch size", min_value=1, value=32, step=1)
        embedding_limit_value = st.number_input(
            "Embedding limit",
            min_value=0,
            value=0,
            step=1,
        )

    if st.button("Process documents", type="primary"):
        try:
            saved_paths = save_uploaded_files(uploaded_files) if uploaded_files else []
            if saved_paths:
                st.write("Saved files")
                st.json([str(path) for path in saved_paths])

            with st.spinner("Parsing, chunking, embedding, and indexing..."):
                result = process_documents(
                    env_file=env_file,
                    region=region,
                    embedding_model_id=embedding_model_id,
                    dimensions=int(dimensions),
                    qdrant_url=qdrant_url,
                    collection_name=collection_name,
                    strategy=strategy,
                    extract_images=extract_images,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                    batch_size=int(batch_size),
                    recreate_collection=recreate_collection,
                    limit=int(embedding_limit_value) or None,
                )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Indexed documents into {result['collection_name']}")
            with st.expander("Pipeline result", expanded=True):
                st.json(result)
