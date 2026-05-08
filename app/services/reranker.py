from openai import OpenAI
from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

# Chars of each chunk shown to the reranker LLM (code/tables often need more than 500).
RERANK_SNIPPET_MAX_CHARS = 1600


class RerankerService:
    """
    Reranks retrieved chunks using an LLM (same provider as chat — OpenAI or Groq).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**kwargs)
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Rerank chunks based on relevance to the query.
        """

        if not chunks:
            logger.info(">>> RERANK: no chunks, returning []")
            return []

        logger.info(">>> RERANK: start query=%s input_chunks=%s top_n=%s", repr((query or "")[:60]), len(chunks), top_n)

        # Build reranking prompt
        numbered_chunks = "\n\n".join(
            f"[{i}] {chunk.get('text', '')[:RERANK_SNIPPET_MAX_CHARS]}"
            for i, chunk in enumerate(chunks)
        )

        prompt = f"""
You are a search relevance evaluator.

Given a user question and a list of document chunks,
rank the chunks from MOST relevant to LEAST relevant.

Return ONLY a comma-separated list of chunk indices.

Question:
{query}

Chunks:
{numbered_chunks}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        ranking_text = response.choices[0].message.content.strip()
        logger.info(">>> RERANK: LLM raw_ranking=%s", ranking_text[:120] + ("..." if len(ranking_text) > 120 else ""))

        # Parse indices: allow "0, 2, 1", "0,2,1", or "0 2 1"
        try:
            parts = re.split(r"[\s,]+", ranking_text)
            seen = set()
            ranked_indices = []
            for p in parts:
                p = p.strip()
                if not p or not p.isdigit():
                    continue
                i = int(p)
                if 0 <= i < len(chunks) and i not in seen:
                    seen.add(i)
                    ranked_indices.append(i)
            # If LLM returned too few, append remaining chunk indices in order
            for i in range(len(chunks)):
                if i not in seen:
                    ranked_indices.append(i)
            reranked = [chunks[i] for i in ranked_indices[:top_n]]
        except Exception as e:
            logger.warning(">>> RERANK: parse failed, using original order: %s", e)
            return chunks[:top_n]

        logger.info(">>> RERANK: done output_chunks=%s ranked_indices=%s", len(reranked), ranked_indices[:top_n])
        return reranked
