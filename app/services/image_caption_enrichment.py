"""
Optional OpenAI vision captions for extracted PDF figures — improves vector/keyword retrieval
when the default text is only page/section context (Issue: diagrams not findable by topic).
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_BYTES = 4_000_000


def enrich_image_blocks_for_search(blocks: List[Dict[str, Any]]) -> None:
    """
    Mutates image blocks in place: appends a short vision-model description to ``content``
    for embedding and FTS. No-op if ENABLE_VISION_IMAGE_CAPTIONS is false or OPENAI_API_KEY missing.
    """
    if not getattr(settings, "ENABLE_VISION_IMAGE_CAPTIONS", False):
        return
    key = (getattr(settings, "OPENAI_API_KEY", None) or "").strip()
    if not key:
        logger.warning("ENABLE_VISION_IMAGE_CAPTIONS is on but OPENAI_API_KEY is empty; skipping vision captions")
        return

    max_n = max(0, int(getattr(settings, "MAX_VISION_CAPTIONS_PER_DOCUMENT", 30) or 0))
    if max_n == 0:
        return

    model = (getattr(settings, "VISION_CAPTION_MODEL", None) or "gpt-4o-mini").strip()
    used = 0

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
    except Exception as e:
        logger.warning("Vision caption client init failed: %s", e)
        return

    for block in blocks:
        if block.get("block_type") != "image":
            continue
        if used >= max_n:
            logger.info("Vision captions: hit MAX_VISION_CAPTIONS_PER_DOCUMENT=%s for this PDF", max_n)
            break
        path_str = (block.get("image_meta") or {}).get("name")
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            if len(data) > _MAX_BYTES:
                logger.warning("Vision caption skipped (file > %s bytes): %s", _MAX_BYTES, path)
                continue
            b64 = base64.b64encode(data).decode("ascii")
            low = path.suffix.lower()
            mime = "image/png" if low == ".png" else "image/jpeg" if low in (".jpg", ".jpeg") else "image/png"
            url = f"data:{mime};base64,{b64}"
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this technical figure for search indexing. "
                                    "In 2–4 short sentences: diagram or image type, main components, "
                                    "readable labels or titles, and the topic it illustrates. "
                                    "If unreadable or blank, say so briefly."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
                        ],
                    }
                ],
                max_tokens=280,
                temperature=0.2,
            )
            desc = (r.choices[0].message.content or "").strip()
            if desc:
                base = (block.get("content") or "").strip()
                block["content"] = (base + "\n\n[Visual description]: " + desc).strip()
                block.setdefault("image_meta", {})["vision_caption"] = desc
                used += 1
        except Exception as e:
            logger.warning("Vision caption failed for %s: %s", path, e)
