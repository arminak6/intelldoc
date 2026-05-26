from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "pdf"
DEFAULT_INPUT_NAMES = ("enriched_elements.json", "elements.json")
DEFAULT_OUTPUT_NAME = "chunks.json"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
STANDALONE_CONTENT_TYPES = {"table", "image_description"}


def _unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []

    for value in values:
        if value is None:
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)

    return result


def _compact_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    text = _compact_text(text)
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to zero and smaller than chunk_size")

    parts: list[dict[str, Any]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        part_text = text[start:end].strip()
        if part_text:
            parts.append(
                {
                    "text": part_text,
                    "char_start": start,
                    "char_end": end,
                },
            )

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return parts


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:chunk:{chunk_index:06d}"


def _element_page_number(element: dict[str, Any]) -> Any:
    return element.get("page_number") or element.get("metadata", {}).get("page_number")


def _element_parser(element: dict[str, Any]) -> Any:
    return element.get("parser") or element.get("metadata", {}).get("parser")


def _element_extraction_method(element: dict[str, Any]) -> Any:
    return element.get("extraction_method") or element.get("metadata", {}).get(
        "extraction_method",
    )


def _element_path(element: dict[str, Any], key: str) -> Any:
    return element.get(key) or element.get("metadata", {}).get(key)


def _base_chunk(
    document_id: str,
    chunk_index: int,
    text: str,
    elements: list[dict[str, Any]],
    content_type: str,
    source_char_start: int | None = None,
    source_char_end: int | None = None,
) -> dict[str, Any]:
    source_element_ids = [element.get("element_id") for element in elements]
    source_element_indexes = [element.get("element_index") for element in elements]
    page_numbers = [_element_page_number(element) for element in elements]
    image_paths = [_element_path(element, "image_path") for element in elements]
    table_paths = [_element_path(element, "table_path") for element in elements]
    parsers = [_element_parser(element) for element in elements]
    extraction_methods = [_element_extraction_method(element) for element in elements]

    chunk = {
        "chunk_id": _chunk_id(document_id, chunk_index),
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content_type": content_type,
        "content_types": _unique_preserve_order(
            [element.get("content_type") for element in elements],
        ),
        "text": text,
        "source_element_ids": _unique_preserve_order(source_element_ids),
        "source_element_indexes": _unique_preserve_order(source_element_indexes),
        "page_numbers": _unique_preserve_order(page_numbers),
        "image_paths": _unique_preserve_order(image_paths),
        "table_paths": _unique_preserve_order(table_paths),
        "parsers": _unique_preserve_order(parsers),
        "extraction_methods": _unique_preserve_order(extraction_methods),
        "previous_chunk_id": None,
        "next_chunk_id": None,
        "metadata": {
            "source_element_ids": _unique_preserve_order(source_element_ids),
            "source_element_indexes": _unique_preserve_order(source_element_indexes),
            "page_numbers": _unique_preserve_order(page_numbers),
            "image_paths": _unique_preserve_order(image_paths),
            "table_paths": _unique_preserve_order(table_paths),
            "parsers": _unique_preserve_order(parsers),
            "extraction_methods": _unique_preserve_order(extraction_methods),
        },
    }

    if source_char_start is not None:
        chunk["source_char_start"] = source_char_start
        chunk["metadata"]["source_char_start"] = source_char_start

    if source_char_end is not None:
        chunk["source_char_end"] = source_char_end
        chunk["metadata"]["source_char_end"] = source_char_end

    return chunk


def _link_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, chunk in enumerate(chunks):
        previous_chunk_id = chunks[index - 1]["chunk_id"] if index else None
        next_chunk_id = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
        chunk["previous_chunk_id"] = previous_chunk_id
        chunk["next_chunk_id"] = next_chunk_id
        chunk["metadata"]["previous_chunk_id"] = previous_chunk_id
        chunk["metadata"]["next_chunk_id"] = next_chunk_id

    return chunks


def create_chunks(
    payload: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    document_id = payload["document_id"]
    elements = payload.get("elements", [])
    chunks: list[dict[str, Any]] = []

    for element in elements:
        content_type = element.get("content_type", "text")
        text = _compact_text(element.get("text", ""))
        if not text:
            continue

        if content_type in STANDALONE_CONTENT_TYPES:
            chunks.append(
                _base_chunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=text,
                    elements=[element],
                    content_type=content_type,
                ),
            )
            continue

        if content_type != "text":
            continue

        for part in _split_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                _base_chunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=part["text"],
                    elements=[element],
                    content_type="text",
                    source_char_start=part["char_start"],
                    source_char_end=part["char_end"],
                ),
            )

    return _link_chunks(chunks)


def choose_input_file(document_dir: Path) -> Path | None:
    for input_name in DEFAULT_INPUT_NAMES:
        input_path = document_dir / input_name
        if input_path.exists():
            return input_path
    return None


def find_document_inputs(parsed_dir: Path) -> list[Path]:
    input_files: list[Path] = []
    for document_dir in sorted(path for path in parsed_dir.iterdir() if path.is_dir()):
        input_path = choose_input_file(document_dir)
        if input_path is not None:
            input_files.append(input_path)
    return input_files


def chunk_file(
    input_path: Path,
    output_path: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = create_chunks(payload, chunk_size=chunk_size, overlap=overlap)
    output_path = output_path or input_path.with_name(DEFAULT_OUTPUT_NAME)

    chunk_payload = {
        "document_id": payload["document_id"],
        "source": payload.get("source"),
        "input_file": str(input_path),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": chunks,
    }
    output_path.write_text(
        json.dumps(chunk_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create RAG chunks from parsed/enriched document elements.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to one enriched_elements.json or elements.json file.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="Directory containing per-document parsed PDF folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output chunks.json path. Only valid with --input.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum characters per text chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Character overlap between consecutive chunks from the same text element.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.output and not args.input:
        raise ValueError("--output can only be used together with --input")

    input_files = [args.input] if args.input else find_document_inputs(args.parsed_dir)
    if not input_files:
        raise FileNotFoundError(f"No parsed document JSON files found in {args.parsed_dir}")

    for input_path in input_files:
        output_path = chunk_file(
            input_path=input_path,
            output_path=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        print(output_path)


if __name__ == "__main__":
    main()
