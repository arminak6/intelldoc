from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARSED_DIR = PROJECT_ROOT / "data" / "parsed" / "pdf"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.bedrock"
DEFAULT_OUTPUT_NAME = "enriched_elements.json"
PROMPT_VERSION = "image-description-v1"
SUPPORTED_IMAGE_FORMATS = {
    ".gif": "gif",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
}


BASE_IMAGE_PROMPT = """Describe this image for a RAG system that answers questions about internal company documents.

Be detailed and factual. Include:
- visible text, labels, numbers, and legends
- entities, systems, people, forms, UI screens, diagrams, arrows, and relationships
- what the image contributes to the surrounding document

Do not guess beyond what is visible. If the image is decorative, blank, tiny, or not useful, say that clearly.
"""


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


def image_format_for_path(image_path: Path) -> str:
    image_format = SUPPORTED_IMAGE_FORMATS.get(image_path.suffix.lower())
    if image_format is None:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ValueError(f"Unsupported image format for {image_path}. Supported: {supported}")
    return image_format


def nearby_context(
    elements: list[dict[str, Any]],
    element_index: int,
    max_chars: int = 1600,
    max_elements_each_side: int = 3,
) -> str:
    before: list[str] = []
    after: list[str] = []

    for element in reversed(elements[max(0, element_index - max_elements_each_side) : element_index]):
        text = element.get("text", "").strip()
        if text:
            before.append(text)

    for element in elements[element_index + 1 : element_index + 1 + max_elements_each_side]:
        text = element.get("text", "").strip()
        if text:
            after.append(text)

    context = "\n\n".join(
        [
            "Previous document text:",
            "\n\n".join(reversed(before)) or "(none)",
            "Following document text:",
            "\n\n".join(after) or "(none)",
        ],
    )
    return context[:max_chars]


def build_image_prompt(element: dict[str, Any], context: str) -> str:
    page_number = element.get("page_number") or element.get("metadata", {}).get("page_number")
    page_text = f"Page number: {page_number}" if page_number is not None else "Page number: unknown"
    return f"{BASE_IMAGE_PROMPT}\n\n{page_text}\n\nSurrounding context:\n{context}"


def describe_image(
    image_path: Path,
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int = 900,
    temperature: float = 0.0,
) -> str:
    image_bytes = image_path.read_bytes()
    image_format = image_format_for_path(image_path)

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        },
                    },
                    {"text": prompt},
                ],
            },
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    content = response["output"]["message"]["content"]
    description = "\n".join(block["text"] for block in content if "text" in block).strip()
    if not description:
        raise RuntimeError(f"Bedrock returned an empty description for {image_path}")
    return description


def enrich_image_element(
    element: dict[str, Any],
    description: str,
    model_id: str,
) -> dict[str, Any]:
    metadata = element.setdefault("metadata", {})
    metadata.update(
        {
            "description_model_id": model_id,
            "description_prompt_version": PROMPT_VERSION,
            "description_status": "described",
            "original_content_type": element.get("content_type"),
        },
    )

    element["type"] = "ImageDescription"
    element["content_type"] = "image_description"
    element["original_content_type"] = "image"
    element["text"] = description
    element["description"] = description
    element["description_model_id"] = model_id
    element["description_prompt_version"] = PROMPT_VERSION
    element["description_status"] = "described"
    return element


def mark_image_skipped(element: dict[str, Any], reason: str) -> dict[str, Any]:
    metadata = element.setdefault("metadata", {})
    metadata["description_status"] = "skipped"
    metadata["description_skip_reason"] = reason
    element["description_status"] = "skipped"
    element["description_skip_reason"] = reason
    return element


def enrich_document(
    input_path: Path,
    output_path: Path,
    client: Any | None,
    model_id: str | None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    enriched_payload = copy.deepcopy(payload)
    elements = enriched_payload.get("elements", [])

    image_elements = [
        element for element in elements if element.get("content_type") == "image"
    ]
    described_count = 0
    skipped_count = 0

    for element in image_elements:
        if limit is not None and described_count >= limit:
            skipped_count += 1
            mark_image_skipped(element, "limit_reached")
            continue

        element_index = element.get("element_index")
        if not isinstance(element_index, int):
            element_index = elements.index(element)

        image_path_text = element.get("image_path") or element.get("metadata", {}).get("image_path")
        if not image_path_text:
            skipped_count += 1
            mark_image_skipped(element, "missing_image_path")
            continue

        image_path = Path(image_path_text)
        if not image_path.exists():
            skipped_count += 1
            mark_image_skipped(element, "image_file_not_found")
            continue

        if dry_run:
            described_count += 1
            continue

        if client is None or model_id is None:
            raise RuntimeError("Bedrock client and model_id are required unless --dry-run is used.")

        context = nearby_context(elements, element_index)
        prompt = build_image_prompt(element, context)
        description = describe_image(image_path, client, model_id, prompt)
        enrich_image_element(element, description, model_id)
        described_count += 1

    enriched_payload["enrichment"] = {
        "description_model_id": model_id,
        "description_prompt_version": PROMPT_VERSION,
        "image_elements": len(image_elements),
        "described_images": described_count,
        "skipped_images": skipped_count,
        "dry_run": dry_run,
    }

    if not dry_run:
        output_path.write_text(
            json.dumps(enriched_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return enriched_payload["enrichment"]


def find_input_files(parsed_dir: Path) -> list[Path]:
    return sorted(parsed_dir.glob("*/elements.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Bedrock image descriptions for parsed document elements.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to one elements.json file. Defaults to all parsed PDF documents.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=DEFAULT_PARSED_DIR,
        help="Directory containing per-document parsed PDF folders.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output JSON filename written beside each elements.json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Local env file containing AWS Bedrock credentials/settings.",
    )
    parser.add_argument(
        "--region",
        help="AWS region. Defaults to AWS_REGION or AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--model-id",
        help="Bedrock model ID. Defaults to AWS_BEDROCK_MODEL_ID or MODEL_ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images to describe per document.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count processable images without calling Bedrock or writing output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_env_file(args.env_file)

    region = args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    model_id = args.model_id or os.environ.get("AWS_BEDROCK_MODEL_ID") or os.environ.get("MODEL_ID")

    input_files = [args.input] if args.input else find_input_files(args.parsed_dir)
    if not input_files:
        raise FileNotFoundError(f"No elements.json files found in {args.parsed_dir}")

    client = None if args.dry_run else build_bedrock_client(region)

    for input_path in input_files:
        output_path = input_path.with_name(args.output_name)
        enrichment = enrich_document(
            input_path=input_path,
            output_path=output_path,
            client=client,
            model_id=model_id,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        print(f"{input_path} -> {output_path}")
        print(json.dumps(enrichment, ensure_ascii=False))


if __name__ == "__main__":
    main()
