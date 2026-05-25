from __future__ import annotations

import argparse
import json
import re
import shutil
import warnings
from collections import Counter
from mimetypes import guess_type
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed" / "pdf"
DEFAULT_STRATEGY = "hi_res"
IMAGE_BLOCK_TYPES = ("Image", "Table")
TABLE_CAPTION_PATTERN = re.compile(r"^Table\s+(\d+):\s*(.*)", re.IGNORECASE)
SECTION_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*\.\s+\S+")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
    except TypeError:
        return str(value)

    return value


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}

    if hasattr(metadata, "to_dict"):
        return _json_safe(metadata.to_dict())

    if isinstance(metadata, dict):
        return _json_safe(metadata)

    return _json_safe(dict(getattr(metadata, "__dict__", {})))


def _content_type(element_type: str) -> str:
    if element_type == "Table":
        return "table"

    if element_type == "Image":
        return "image"

    return "text"


def _element_to_dict(element: Any) -> dict[str, Any]:
    element_type = element.__class__.__name__
    metadata = _metadata_to_dict(getattr(element, "metadata", None))
    payload = {
        "type": element_type,
        "content_type": _content_type(element_type),
        "text": str(element),
        "metadata": metadata,
    }

    table_html = metadata.get("text_as_html")
    if table_html:
        payload["table_html"] = table_html

    image_path = metadata.get("image_path")
    if image_path:
        payload["image_path"] = image_path

    image_mime_type = metadata.get("image_mime_type")
    if image_mime_type:
        payload["image_mime_type"] = image_mime_type

    return payload


def _safe_filename(name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in {".", "-", "_"} else "_"
        for char in name
    )
    return safe_name.strip("._") or "image"


def _looks_like_table_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    tokens = compact.split()
    has_digit = any(char.isdigit() for char in compact)
    has_many_tokens = len(tokens) >= 3
    has_table_symbol = any(symbol in compact for symbol in ("±", "%", ".", "-", "–"))
    return has_digit and has_many_tokens and (has_table_symbol or len(tokens) >= 5)


def _table_like_score(lines: list[str]) -> int:
    return sum(1 for line in lines if _looks_like_table_line(line))


def _extract_table_like_elements(
    page_text: str,
    pdf_path: Path,
    page_number: int,
) -> list[dict[str, Any]]:
    lines = page_text.splitlines()
    table_elements: list[dict[str, Any]] = []
    seen_table_numbers: set[int] = set()

    for caption_index, line in enumerate(lines):
        caption_match = TABLE_CAPTION_PATTERN.match(line.strip())
        if not caption_match:
            continue

        table_number = int(caption_match.group(1))
        if table_number in seen_table_numbers:
            continue

        window_start = max(0, caption_index - 45)
        table_lines = lines[window_start:caption_index]
        if _table_like_score(table_lines) < 2:
            continue

        caption_lines = [line.strip()]
        for next_line in lines[caption_index + 1 : caption_index + 5]:
            stripped = next_line.strip()
            if not stripped:
                break
            if SECTION_HEADING_PATTERN.match(stripped):
                break
            if TABLE_CAPTION_PATTERN.match(stripped):
                break
            caption_lines.append(stripped)
            if stripped.endswith("."):
                break

        seen_table_numbers.add(table_number)
        table_elements.append(
            {
                "type": "Table",
                "content_type": "table",
                "text": "\n".join([*table_lines, *caption_lines]).strip(),
                "metadata": {
                    "page_number": page_number,
                    "filename": pdf_path.name,
                    "parser": "pypdf",
                    "extraction_method": "caption_window",
                    "table_number": table_number,
                    "is_structured": False,
                },
            },
        )

    return table_elements


def _parse_pdf_with_pypdf(
    pdf_path: Path,
    image_output_dir: Path | None = None,
    extract_images: bool = True,
) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    elements: list[dict[str, Any]] = []

    if extract_images and image_output_dir is not None:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            elements.append(
                {
                    "type": "Text",
                    "content_type": "text",
                    "text": text,
                    "metadata": {
                        "page_number": page_index,
                        "filename": pdf_path.name,
                        "parser": "pypdf",
                    },
                },
            )
            elements.extend(_extract_table_like_elements(text, pdf_path, page_index))

        if not extract_images or image_output_dir is None:
            continue

        for image_index, image in enumerate(page.images, start=1):
            image_name = _safe_filename(getattr(image, "name", "image"))
            image_path = image_output_dir / f"page_{page_index:03d}_{image_index:03d}_{image_name}"
            image_path.write_bytes(image.data)
            image_mime_type = guess_type(image_path.name)[0]
            elements.append(
                {
                    "type": "Image",
                    "content_type": "image",
                    "text": "",
                    "image_path": str(image_path),
                    "image_mime_type": image_mime_type,
                    "metadata": {
                        "page_number": page_index,
                        "filename": pdf_path.name,
                        "image_path": str(image_path),
                        "image_mime_type": image_mime_type,
                        "parser": "pypdf",
                    },
                },
            )

    return elements


def _format_text_element(element: dict[str, Any]) -> str:
    text = element.get("text", "").strip()
    metadata = element.get("metadata", {})
    page_number = metadata.get("page_number")
    page = f" page {page_number}" if page_number else ""

    if element["content_type"] == "table":
        table_path = element.get("table_path")
        suffix = f": {table_path}" if table_path else ""
        return f"[TABLE{page}{suffix}]\n{text}" if text else f"[TABLE{page}{suffix}]"

    if element["content_type"] == "image":
        image_path = element.get("image_path") or metadata.get("image_path")
        suffix = f": {image_path}" if image_path else ""
        return f"[IMAGE{page}{suffix}]"

    return text


def _write_table_files(
    elements: list[dict[str, Any]],
    table_output_dir: Path,
) -> list[Path]:
    table_elements = [
        element for element in elements if element.get("content_type") == "table"
    ]
    if not table_elements:
        return []

    table_output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for table_index, element in enumerate(table_elements, start=1):
        page_number = element.get("metadata", {}).get("page_number")
        page_suffix = f"_page_{page_number}" if page_number else ""
        table_html = element.get("table_html")
        extension = "html" if table_html else "txt"
        table_path = table_output_dir / (
            f"table_{table_index:03d}{page_suffix}.{extension}"
        )
        table_path.write_text(table_html or element.get("text", ""), encoding="utf-8")
        element["table_path"] = str(table_path)
        written_files.append(table_path)

    return written_files


def parse_pdf(
    pdf_path: Path,
    image_output_dir: Path | None = None,
    strategy: str = DEFAULT_STRATEGY,
    extract_images: bool = True,
) -> list[dict[str, Any]]:
    if strategy in {"hi_res", "ocr_only"} and shutil.which("tesseract") is None:
        warnings.warn(
            "Tesseract OCR was not found. Falling back to pypdf, which extracts "
            "text and embedded images but does not detect tables as structured tables.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _parse_pdf_with_pypdf(
            pdf_path,
            image_output_dir=image_output_dir,
            extract_images=extract_images,
        )

    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError(
            "Unstructured is not installed. Install it with the PDF extras, "
            "for example: pip install 'unstructured[pdf]'"
        ) from exc

    partition_options: dict[str, Any] = {
        "filename": str(pdf_path),
        "strategy": strategy,
        "infer_table_structure": strategy in {"auto", "hi_res"},
    }

    if extract_images:
        if image_output_dir is None:
            image_output_dir = DEFAULT_OUTPUT_DIR / "images" / pdf_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        partition_options.update(
            {
                "extract_image_block_types": list(IMAGE_BLOCK_TYPES),
                "extract_image_block_output_dir": str(image_output_dir),
            },
        )

    try:
        elements = partition_pdf(**partition_options)
    except Exception as exc:
        if strategy == "hi_res":
            raise RuntimeError(
                "Unable to parse PDF with hi_res strategy. This strategy is required "
                "for table structure and image extraction. Make sure the PDF extras "
                "and OCR/layout dependencies are installed."
            ) from exc
        raise

    return [_element_to_dict(element) for element in elements]


def parse_raw_pdfs(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strategy: str = DEFAULT_STRATEGY,
    extract_images: bool = True,
) -> list[Path]:
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for pdf_path in pdf_paths:
        document_output_dir = output_dir / pdf_path.stem
        image_output_dir = document_output_dir / "images"
        table_output_dir = document_output_dir / "tables"
        document_output_dir.mkdir(parents=True, exist_ok=True)
        table_output_dir.mkdir(parents=True, exist_ok=True)
        for old_table_file in table_output_dir.glob("table_*"):
            if old_table_file.is_file():
                old_table_file.unlink()

        elements = parse_pdf(
            pdf_path,
            image_output_dir=image_output_dir,
            strategy=strategy,
            extract_images=extract_images,
        )
        table_files = _write_table_files(elements, table_output_dir)
        content_counts = Counter(element["content_type"] for element in elements)
        payload = {
            "source": str(pdf_path),
            "element_count": len(elements),
            "content_counts": dict(content_counts),
            "strategy": strategy,
            "output_dir": str(document_output_dir),
            "image_output_dir": str(image_output_dir) if extract_images else None,
            "table_output_dir": str(table_output_dir),
            "elements": elements,
        }

        json_path = document_output_dir / "elements.json"
        text_path = document_output_dir / "text.txt"

        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        text_path.write_text(
            "\n\n".join(
                formatted
                for element in elements
                if (formatted := _format_text_element(element))
            ),
            encoding="utf-8",
        )
        written_files.extend([json_path, text_path, *table_files])

    return written_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse raw PDFs with text, table, and image extraction.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where parsed JSON and text files will be written.",
    )
    parser.add_argument(
        "--strategy",
        choices=("hi_res", "auto", "fast", "ocr_only"),
        default=DEFAULT_STRATEGY,
        help="Unstructured PDF parsing strategy. hi_res is best for tables/images.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image/table block image extraction.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written_files = parse_raw_pdfs(
        args.input_dir,
        args.output_dir,
        strategy=args.strategy,
        extract_images=not args.no_images,
    )

    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
