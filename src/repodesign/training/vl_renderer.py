"""Qwen3-VL Renderer: builds multimodal ModelInput with interleaved text and images.

Handles Qwen3-VL's chat template with vision special tokens:
  <|im_start|>system\n...<|im_end|>
  <|im_start|>user\n<|vision_start|>[IMAGE]<|vision_end|>text...<|im_end|>
  <|im_start|>assistant\n<think>\n

For the GRPO training loop, this renderer:
  - Builds ModelInput with EncodedTextChunk and ImageChunk interleaved
  - Handles multiple images per message
  - Falls back to text-only when no images are present
  - Parses responses (text-only, same as Qwen3Renderer)
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Any

from PIL import Image
import tinker
from tinker.types import EncodedTextChunk, ImageChunk, ModelInput

logger = logging.getLogger(__name__)

# Qwen3-VL special token IDs
VISION_START_TOKEN = 151652  # <|vision_start|>
VISION_END_TOKEN = 151653    # <|vision_end|>
IMAGE_PAD_TOKEN = 151655     # <|image_pad|>
IM_START_TOKEN = 151644      # <|im_start|>
IM_END_TOKEN = 151645        # <|im_end|>


# Qwen3-VL image processing constants (from preprocessor_config.json)
QWEN3_VL_PATCH_SIZE = 16
QWEN3_VL_MERGE_SIZE = 2
QWEN3_VL_MIN_PIXELS = 65536     # shortest_edge in preprocessor_config
QWEN3_VL_MAX_PIXELS = 16777216  # longest_edge in preprocessor_config


def compute_image_tokens(image_data: bytes) -> int:
    """Compute the expected number of vision tokens for a Qwen3-VL image.

    Qwen3-VL processes images by:
      1. Resizing to fit within min/max pixel constraints
      2. Rounding dimensions to multiples of (patch_size * merge_size)
      3. Dividing into patches and spatially merging

    tokens = (H / patch_size) * (W / patch_size) / (merge_size^2)
    """
    img = Image.open(io.BytesIO(image_data))
    w, h = img.size

    # Smart resize to fit within pixel budget
    total_pixels = w * h
    factor = (patch_size := QWEN3_VL_PATCH_SIZE) * (merge_size := QWEN3_VL_MERGE_SIZE)  # 32

    if total_pixels < QWEN3_VL_MIN_PIXELS:
        scale = math.sqrt(QWEN3_VL_MIN_PIXELS / total_pixels)
        w = int(w * scale)
        h = int(h * scale)
    elif total_pixels > QWEN3_VL_MAX_PIXELS:
        scale = math.sqrt(QWEN3_VL_MAX_PIXELS / total_pixels)
        w = int(w * scale)
        h = int(h * scale)

    # Round to nearest multiple of factor (patch_size * merge_size = 32)
    w = max(factor, round(w / factor) * factor)
    h = max(factor, round(h / factor) * factor)

    # Compute token count
    tokens = (h // patch_size) * (w // patch_size) // (merge_size * merge_size)
    return tokens


class Qwen3VLRenderer:
    """Renderer for Qwen3-VL models with multimodal (text + image) support.

    Builds ModelInput using chunks so that ImageChunk objects are placed
    between <|vision_start|> and <|vision_end|> tokens, matching the
    Qwen3-VL chat template.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def get_stop_sequences(self) -> list[int]:
        """Return stop token IDs for generation."""
        return [IM_END_TOKEN]

    def build_generation_prompt(
        self,
        messages: list[dict],
        diagram_images: list[bytes] | None = None,
    ) -> ModelInput:
        """Build a multimodal ModelInput from chat messages + optional images.

        Args:
            messages: List of chat messages. Each has 'role' and 'content'.
            diagram_images: Optional list of PNG image bytes to include in
                           the user message as architecture diagrams.

        Returns:
            ModelInput with interleaved EncodedTextChunk and ImageChunk.
        """
        chunks: list[EncodedTextChunk | ImageChunk] = []

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            # Build the role header: \n<|im_start|>role\n
            maybe_newline = "\n" if idx > 0 else ""
            header = f"{maybe_newline}<|im_start|>{role}\n"
            header_tokens = self.tokenizer.encode(header, add_special_tokens=False)

            # For the user message, insert images before text
            if role == "user" and diagram_images:
                # Header tokens
                chunks.append(EncodedTextChunk(tokens=header_tokens))

                # Insert each diagram image wrapped in vision tokens
                for img_bytes in diagram_images:
                    fmt = _detect_image_format(img_bytes)
                    expected_tokens = compute_image_tokens(img_bytes)
                    # <|vision_start|>
                    chunks.append(EncodedTextChunk(tokens=[VISION_START_TOKEN]))
                    # The actual image with expected token count
                    chunks.append(ImageChunk(
                        data=img_bytes,
                        format=fmt,
                        expected_tokens=expected_tokens,
                    ))
                    # <|vision_end|>
                    chunks.append(EncodedTextChunk(tokens=[VISION_END_TOKEN]))

                # Text content + <|im_end|>
                footer = f"{content}<|im_end|>"
                footer_tokens = self.tokenizer.encode(footer, add_special_tokens=False)
                chunks.append(EncodedTextChunk(tokens=footer_tokens))
            else:
                # Text-only message (system, assistant, or user without images)
                full_text = f"{header}{content}<|im_end|>"
                tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
                chunks.append(EncodedTextChunk(tokens=tokens))

        # Add the assistant generation prompt with <think> tag
        gen_header = "\n<|im_start|>assistant\n<think>\n"
        gen_tokens = self.tokenizer.encode(gen_header, add_special_tokens=False)
        chunks.append(EncodedTextChunk(tokens=gen_tokens))

        return ModelInput(chunks=chunks)

    def build_text_only_prompt(self, messages: list[dict]) -> ModelInput:
        """Build a text-only prompt (no images). Convenience wrapper."""
        return self.build_generation_prompt(messages, diagram_images=None)

    def parse_response(self, response_tokens: list[int]) -> tuple[dict, bool]:
        """Parse sampled tokens into a message dict.

        Returns:
            (message_dict, parse_success)
        """
        # Find <|im_end|> token to know where the response ends
        end_idx = None
        for i, tok in enumerate(response_tokens):
            if tok == IM_END_TOKEN:
                end_idx = i
                break

        if end_idx is not None:
            content_tokens = response_tokens[:end_idx]
        else:
            content_tokens = response_tokens

        content = self.tokenizer.decode(content_tokens, skip_special_tokens=False)

        # Strip thinking tags if present
        if "<think>" in content:
            # Extract content after </think> if it exists
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            else:
                # Model is still thinking (incomplete), return as-is
                pass

        message = {"role": "assistant", "content": content}
        return message, end_idx is not None


def _detect_image_format(data: bytes) -> str:
    """Detect PNG vs JPEG from magic bytes."""
    if data[:4] == b"\x89PNG":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpeg"
    # Default to png
    return "png"


def load_diagram_images(
    repo_ir: dict,
    repo_dir: str,
    max_images: int = 5,
    max_size_bytes: int = 2_000_000,
) -> list[bytes]:
    """Load diagram images from a repo's extracted data.

    Args:
        repo_ir: The RepoIR dict (has 'diagram_paths' field).
        repo_dir: Path to the repo's data directory.
        max_images: Max number of images to include (for token budget).
        max_size_bytes: Skip images larger than this.

    Returns:
        List of image bytes, ready for ImageChunk.
    """
    images = []
    base = Path(repo_dir)

    for rel_path in repo_ir.get("diagram_paths", []):
        if len(images) >= max_images:
            break

        img_path = base / rel_path
        if not img_path.exists():
            continue
        if img_path.stat().st_size > max_size_bytes:
            continue

        # Only include actual image files
        suffix = img_path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
            continue

        try:
            images.append(img_path.read_bytes())
        except Exception as e:
            logger.warning(f"Failed to read {img_path}: {e}")

    return images
