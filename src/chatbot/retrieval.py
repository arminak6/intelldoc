from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.bedrock"
DEFAULT_COLLECTION_NAME = "intelldoc_chunks"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_TOP_K = 6


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_env_file(env_path: Path = DEFAULT_ENV_FILE) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_bedrock_client(region: str | None = None) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Install it with: pip install boto3") from exc

    return boto3.client("bedrock-runtime", region_name=region)


def build_qdrant_client(url: str, api_key: str | None = None) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install it with: pip install qdrant-client"
        ) from exc

    return QdrantClient(url=url, api_key=api_key, check_compatibility=False)


def embed_query(
    query: str,
    bedrock_client: Any,
    model_id: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    normalize: bool = True,
) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "inputText": query,
                "dimensions": dimensions,
                "normalize": normalize,
            },
        ),
        accept="application/json",
        contentType="application/json",
    )
    body = json.loads(response["body"].read())
    embedding = body.get("embedding")
    if not embedding:
        raise RuntimeError(f"Bedrock returned no query embedding for model {model_id}")
    return embedding


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_result(point: Any, rank: int) -> dict[str, Any]:
    payload = dict(getattr(point, "payload", None) or {})
    return {
        "rank": rank,
        "score": getattr(point, "score", None),
        "point_id": str(getattr(point, "id", "")),
        "chunk_id": payload.get("chunk_id"),
        "document_id": payload.get("document_id"),
        "content_type": payload.get("content_type"),
        "content_types": _as_list(payload.get("content_types")),
        "text": payload.get("text", ""),
        "page_numbers": _as_list(payload.get("page_numbers")),
        "source_element_ids": _as_list(payload.get("source_element_ids")),
        "image_paths": _as_list(payload.get("image_paths")),
        "table_paths": _as_list(payload.get("table_paths")),
        "payload": payload,
    }


def search_chunks(
    query: str,
    bedrock_client: Any,
    qdrant_client: Any,
    collection_name: str,
    embedding_model_id: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    query_vector = embed_query(
        query=query,
        bedrock_client=bedrock_client,
        model_id=embedding_model_id,
        dimensions=dimensions,
    )
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=score_threshold,
    )
    points = getattr(response, "points", [])
    return [normalize_result(point, rank=index + 1) for index, point in enumerate(points)]


def format_results_for_prompt(
    results: list[dict[str, Any]],
    max_chars_per_result: int = 1600,
) -> str:
    blocks: list[str] = []
    for result in results:
        text = result.get("text", "").strip()
        if len(text) > max_chars_per_result:
            text = f"{text[:max_chars_per_result].rstrip()}..."

        pages = ", ".join(str(page) for page in result.get("page_numbers", [])) or "unknown"
        source = result.get("chunk_id") or result.get("point_id")
        content_type = result.get("content_type") or "unknown"
        score = result.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "unknown"

        blocks.append(
            "\n".join(
                [
                    f"[{result['rank']}] source={source}",
                    f"document={result.get('document_id')} pages={pages} type={content_type} score={score_text}",
                    text,
                ],
            ),
        )

    return "\n\n".join(blocks)


def load_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    return {
        "region": args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        "embedding_model_id": (
            args.embedding_model_id
            or os.environ.get("EMBEDDING_MODEL_ID")
            or os.environ.get("AWS_BEDROCK_EMBEDDING_MODEL_ID")
            or DEFAULT_EMBEDDING_MODEL_ID
        ),
        "dimensions": (
            args.dimensions
            or int(os.environ.get("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))
        ),
        "qdrant_url": args.qdrant_url or os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL,
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY"),
        "collection_name": (
            args.collection
            or os.environ.get("QDRANT_COLLECTION")
            or DEFAULT_COLLECTION_NAME
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from Qdrant.")
    parser.add_argument("query", nargs="*", help="Question/search query.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Local env file containing AWS/Qdrant settings.",
    )
    parser.add_argument("--region", help="AWS region.")
    parser.add_argument("--embedding-model-id", help="Bedrock embedding model ID.")
    parser.add_argument("--dimensions", type=int, help="Embedding dimensions.")
    parser.add_argument("--qdrant-url", help="Qdrant URL.")
    parser.add_argument("--collection", help="Qdrant collection name.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument("--score-threshold", type=float, help="Optional Qdrant score threshold.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    return parser


def main() -> None:
    configure_stdout()
    args = build_parser().parse_args()
    query = " ".join(args.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        raise ValueError("A query is required.")

    config = load_runtime_config(args)
    bedrock_client = build_bedrock_client(config["region"])
    qdrant_client = build_qdrant_client(config["qdrant_url"], config["qdrant_api_key"])
    results = search_chunks(
        query=query,
        bedrock_client=bedrock_client,
        qdrant_client=qdrant_client,
        collection_name=config["collection_name"],
        embedding_model_id=config["embedding_model_id"],
        dimensions=config["dimensions"],
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(format_results_for_prompt(results))


if __name__ == "__main__":
    main()
