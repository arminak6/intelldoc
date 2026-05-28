from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "pdf"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.bedrock"
DEFAULT_COLLECTION_NAME = "intelldoc_chunks"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_QDRANT_URL = "http://localhost:6333"


def load_env_file(env_path: Path) -> None:
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

    return QdrantClient(url=url, api_key=api_key)


def embed_text(
    text: str,
    bedrock_client: Any,
    model_id: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    normalize: bool = True,
) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(
            {
                "inputText": text,
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
        raise RuntimeError(f"Bedrock returned no embedding for model {model_id}")
    return embedding


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_for_chunk(chunk: dict[str, Any], embedding_model_id: str) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in chunk.items()
        if key not in {"vector", "embedding", "metadata"}
    }
    payload["embedding_model_id"] = embedding_model_id
    payload["text_hash"] = text_hash(chunk.get("text", ""))
    return payload


def ensure_collection(
    qdrant_client: Any,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing_collections = {
        collection.name for collection in qdrant_client.get_collections().collections
    }

    if collection_name in existing_collections and recreate:
        qdrant_client.delete_collection(collection_name=collection_name)
        existing_collections.remove(collection_name)

    if collection_name not in existing_collections:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    chunks_path: Path,
    qdrant_client: Any,
    bedrock_client: Any,
    collection_name: str,
    embedding_model_id: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    batch_size: int = 32,
    limit: int | None = None,
    recreate_collection: bool = False,
) -> dict[str, Any]:
    from qdrant_client.models import PointStruct

    chunk_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [
        chunk
        for chunk in chunk_payload.get("chunks", [])
        if chunk.get("text", "").strip()
    ]
    if limit is not None:
        chunks = chunks[:limit]

    if not chunks:
        return {
            "chunks_path": str(chunks_path),
            "collection_name": collection_name,
            "embedded_chunks": 0,
            "upserted_points": 0,
        }

    first_vector = embed_text(
        chunks[0]["text"],
        bedrock_client=bedrock_client,
        model_id=embedding_model_id,
        dimensions=dimensions,
    )
    ensure_collection(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        vector_size=len(first_vector),
        recreate=recreate_collection,
    )

    pending_points: list[PointStruct] = [
        PointStruct(
            id=point_id_for_chunk(chunks[0]["chunk_id"]),
            vector=first_vector,
            payload=payload_for_chunk(chunks[0], embedding_model_id),
        ),
    ]
    upserted_points = 0

    for chunk in chunks[1:]:
        vector = embed_text(
            chunk["text"],
            bedrock_client=bedrock_client,
            model_id=embedding_model_id,
            dimensions=dimensions,
        )
        pending_points.append(
            PointStruct(
                id=point_id_for_chunk(chunk["chunk_id"]),
                vector=vector,
                payload=payload_for_chunk(chunk, embedding_model_id),
            ),
        )

        if len(pending_points) >= batch_size:
            qdrant_client.upsert(collection_name=collection_name, points=pending_points)
            upserted_points += len(pending_points)
            pending_points = []

    if pending_points:
        qdrant_client.upsert(collection_name=collection_name, points=pending_points)
        upserted_points += len(pending_points)

    return {
        "chunks_path": str(chunks_path),
        "collection_name": collection_name,
        "embedding_model_id": embedding_model_id,
        "embedding_dimensions": len(first_vector),
        "embedded_chunks": len(chunks),
        "upserted_points": upserted_points,
    }


def find_chunks_files(parsed_dir: Path) -> list[Path]:
    return sorted(parsed_dir.glob("*/chunks.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed chunks with AWS Bedrock and index them into Qdrant.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to one chunks.json file. Defaults to all parsed document chunks.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="Directory containing per-document parsed PDF folders.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Local env file containing AWS/Qdrant settings.",
    )
    parser.add_argument(
        "--region",
        help="AWS region. Defaults to AWS_REGION or AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=None,
        help="Bedrock embedding model ID. Defaults to EMBEDDING_MODEL_ID.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Titan embedding output dimensions.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL. Defaults to QDRANT_URL or http://localhost:6333.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of points to upsert per Qdrant request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum chunks to embed per chunks.json file.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be indexed without calling Bedrock or Qdrant.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_env_file(args.env_file)

    region = args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    embedding_model_id = (
        args.embedding_model_id
        or os.environ.get("EMBEDDING_MODEL_ID")
        or os.environ.get("AWS_BEDROCK_EMBEDDING_MODEL_ID")
        or DEFAULT_EMBEDDING_MODEL_ID
    )
    dimensions = (
        args.dimensions
        or int(os.environ.get("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))
    )
    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = (
        args.collection
        or os.environ.get("QDRANT_COLLECTION")
        or DEFAULT_COLLECTION_NAME
    )

    chunks_files = [args.input] if args.input else find_chunks_files(args.parsed_dir)
    if not chunks_files:
        raise FileNotFoundError(f"No chunks.json files found in {args.parsed_dir}")

    if args.dry_run:
        for chunks_path in chunks_files:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = payload.get("chunks", [])
            count = min(len(chunks), args.limit) if args.limit is not None else len(chunks)
            print(
                json.dumps(
                    {
                        "chunks_path": str(chunks_path),
                        "collection_name": collection_name,
                        "embedding_model_id": embedding_model_id,
                        "embedding_dimensions": dimensions,
                        "qdrant_url": qdrant_url,
                        "chunks_to_index": count,
                        "dry_run": True,
                    },
                    ensure_ascii=False,
                ),
            )
        return

    bedrock_client = build_bedrock_client(region)
    qdrant_client = build_qdrant_client(qdrant_url, api_key=qdrant_api_key)

    for chunks_path in chunks_files:
        result = upsert_chunks(
            chunks_path=chunks_path,
            qdrant_client=qdrant_client,
            bedrock_client=bedrock_client,
            collection_name=collection_name,
            embedding_model_id=embedding_model_id,
            dimensions=dimensions,
            batch_size=args.batch_size,
            limit=args.limit,
            recreate_collection=args.recreate_collection,
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
