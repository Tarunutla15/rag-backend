"""Chat API route - RAG Q&A with session and context."""
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
from app.config import settings, get_completion_client_config
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import get_chat_service
from app.services.query_classifier import QueryClassifier, classify_query_with_context, classify_query
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.document_store import get_document_store as _get_document_store
from app.services.reranker import RerankerService
from app.services.keyword_search import get_keyword_search_service
from app.services.chunk_store import get_chunk_store
from app.services.raw_block_store import get_raw_block_store
from app.services.usage_store import record_chat_completion
import json

# Reuse vector store getter from upload route to avoid duplicate init logic
from app.api.routes.upload import get_vector_store

router = APIRouter(prefix="/chat", tags=["chat"])

# Lazy-initialized services
_embedding_service = None
_llm_service = None
_document_service = None
_reranker_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
    return _embedding_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        key, model, base_url = get_completion_client_config()
        _llm_service = LLMService(api_key=key, model=model, base_url=base_url)
    return _llm_service


def get_document_store():
    """Get DB-backed document store singleton."""
    return _get_document_store()


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        key, model, base_url = get_completion_client_config()
        _reranker_service = RerankerService(api_key=key, model=model, base_url=base_url)
    return _reranker_service


def _get_previous_user_message(recent_messages: list) -> str:
    """Return the most recent user message before the current one, if any."""
    if not recent_messages or len(recent_messages) < 2:
        return ""
    # recent_messages are chronological; last item is current user query
    for msg in reversed(recent_messages[:-1]):
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            return (msg.get("content") or "").strip()
    return ""


def _get_last_user_message_with_tech(recent_messages: list) -> tuple:
    """
    Return (text, technologies, domain) for the most recent user message that yields a tech.
    Uses classify_query on user messages only.
    """
    if not recent_messages or len(recent_messages) < 2:
        return "", [], None
    for msg in reversed(recent_messages[:-1]):
        if msg.get("role") != "user":
            continue
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        techs, dom = classify_query(text)
        if techs:
            return text, techs, dom
    return "", [], None


def _build_search_query(
    query: str,
    technologies: list,
    from_context: bool,
    recent_messages: list,
    topic_context_text: str = "",
) -> str:
    """Build search query, optionally using previous user message and tech bias."""
    base = query.strip()
    # Only stitch the previous user turn into the embedding for short follow-ups
    # ("what about X?", "and Y?"). Long questions stay standalone so unrelated prior
    # chats do not dilute retrieval (e.g. "web frameworks" before "Module basics").
    if from_context and len(base) <= 36:
        prev_user = topic_context_text or _get_previous_user_message(recent_messages)
        if prev_user:
            base = f"{prev_user} {base}".strip()
    if from_context and technologies:
        return (" ".join(technologies) + " " + base).strip()
    return base


def _retrieve_context_chunks(
    *,
    search_query: str,
    rerank_query: str,
    technologies: list,
    domain: str,
    request: ChatRequest,
    vector_store,
    embedding_service: EmbeddingService,
    keyword_service,
    enable_keyword: bool,
    keyword_top_k: int,
    vector_weight: float,
    keyword_weight: float,
    enable_reranker: bool,
    initial_top_k: int,
    rerank_top_n: int,
    max_chunks_no_rerank: int,
) -> list:
    """Retrieve and (optionally) rerank context chunks."""
    query_embedding = embedding_service.generate_embedding(search_query)

    # Balanced retrieval when scope has 2+ technologies
    if len(technologies) >= 2:
        logger.info(">>> RETRIEVAL: multi-scope balanced retrieval for technologies=%s", technologies)
        context_chunks = []
        reranker = get_reranker_service() if enable_reranker else None
        per_tech_top_n = max(1, rerank_top_n)
        scoped_doc = request.file_id

        def _hybrid_merged(tech: str, dom: Optional[str]) -> tuple[list, list, list]:
            vr = vector_store.search(
                query_embedding=query_embedding,
                top_k=initial_top_k,
                technology=tech,
                domain=dom,
                document_id=request.file_id,
                file_id=request.file_id,
                query_text=search_query,
            )
            kr = []
            if enable_keyword and keyword_service:
                kr = keyword_service.search(
                    query=search_query,
                    top_k=keyword_top_k,
                    technology=tech,
                    domain=dom,
                    document_id=request.file_id,
                    file_id=request.file_id,
                )
            merged = _merge_hybrid_results(vr, kr, vector_weight, keyword_weight)
            return vr, kr, merged

        for tech in technologies:
            tech_domain = QueryClassifier.TECHNOLOGY_TO_DOMAIN.get(tech, "general")
            vector_results, keyword_results, merged = _hybrid_merged(tech, tech_domain)
            if not merged and scoped_doc:
                vector_results, keyword_results, merged = _hybrid_merged(tech, None)
                if merged:
                    logger.info(
                        ">>> RETRIEVAL: technology=%s using domain-agnostic filter (ingest domain != query routing)",
                        tech,
                    )
            logger.info(
                ">>> RETRIEVAL: technology=%s vector=%s keyword=%s merged=%s",
                tech, len(vector_results), len(keyword_results), len(merged)
            )
            if not merged:
                continue
            if enable_reranker and len(merged) > per_tech_top_n and reranker:
                kept = reranker.rerank(query=rerank_query, chunks=merged, top_n=per_tech_top_n)
            else:
                kept = merged[:per_tech_top_n]
            context_chunks.extend(kept)
        # Chunks are stored with document-level tech/domain; query routing may use java+backend while ingest used general
        if not context_chunks and scoped_doc:
            wide_top = max(initial_top_k * 2, rerank_top_n * 4)
            logger.info(
                ">>> RETRIEVAL: multi-tech still empty; document-wide search document_id=%s (no tech/domain filter)",
                scoped_doc,
            )
            vr_w = vector_store.search(
                query_embedding=query_embedding,
                top_k=wide_top,
                technology=None,
                domain=None,
                document_id=scoped_doc,
                file_id=scoped_doc,
                query_text=search_query,
            )
            kr_w = []
            if enable_keyword and keyword_service:
                kr_w = keyword_service.search(
                    query=search_query,
                    top_k=max(keyword_top_k, wide_top),
                    technology=None,
                    domain=None,
                    document_id=scoped_doc,
                    file_id=scoped_doc,
                )
            merged_wide = _merge_hybrid_results(vr_w, kr_w, vector_weight, keyword_weight)
            logger.info(
                ">>> RETRIEVAL: document-wide vector=%s keyword=%s merged=%s",
                len(vr_w), len(kr_w), len(merged_wide),
            )
            if merged_wide:
                if enable_reranker and len(merged_wide) > rerank_top_n and reranker:
                    context_chunks = reranker.rerank(query=rerank_query, chunks=merged_wide, top_n=rerank_top_n)
                elif len(merged_wide) > rerank_top_n:
                    merged_wide.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    context_chunks = merged_wide[:rerank_top_n]
                else:
                    context_chunks = merged_wide
        # One global rerank across technologies so ordering is not biased by tech loop order
        if enable_reranker and reranker and len(context_chunks) > rerank_top_n:
            context_chunks = reranker.rerank(
                query=rerank_query, chunks=context_chunks, top_n=rerank_top_n
            )
        elif len(context_chunks) > rerank_top_n:
            context_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            context_chunks = context_chunks[:rerank_top_n]
        logger.info(">>> RETRIEVAL: balanced total context_chunks=%s (sent to LLM)", len(context_chunks))
        return context_chunks

    # Single scope or no scope: one search, then rerank
    technology = technologies[0] if technologies else None
    logger.info(
        ">>> RETRIEVAL: vector_search top_k=%s (reranker_enabled=%s, rerank_keep=%s)",
        initial_top_k if enable_reranker else max_chunks_no_rerank,
        enable_reranker,
        rerank_top_n if enable_reranker else "n/a",
    )
    scoped_doc = request.file_id
    vector_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=initial_top_k if enable_reranker else max_chunks_no_rerank,
        technology=technology,
        domain=domain,
        document_id=request.file_id,
        file_id=request.file_id,
        query_text=search_query,
    )
    keyword_results = []
    if enable_keyword and keyword_service:
        keyword_results = keyword_service.search(
            query=search_query,
            top_k=keyword_top_k,
            technology=technology,
            domain=domain,
            document_id=request.file_id,
            file_id=request.file_id,
        )
    merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
    if not merged and scoped_doc and domain:
        vector_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=initial_top_k if enable_reranker else max_chunks_no_rerank,
            technology=technology,
            domain=None,
            document_id=request.file_id,
            file_id=request.file_id,
            query_text=search_query,
        )
        keyword_results = []
        if enable_keyword and keyword_service:
            keyword_results = keyword_service.search(
                query=search_query,
                top_k=keyword_top_k,
                technology=technology,
                domain=None,
                document_id=request.file_id,
                file_id=request.file_id,
            )
        merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
        if merged:
            logger.info(">>> RETRIEVAL: single-scope fallback omit domain filter (scoped doc)")
    if not merged and scoped_doc and technology:
        wide_top = max(
            (initial_top_k if enable_reranker else max_chunks_no_rerank) * 2,
            rerank_top_n * 4,
        )
        vector_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=wide_top,
            technology=None,
            domain=None,
            document_id=scoped_doc,
            file_id=scoped_doc,
            query_text=search_query,
        )
        keyword_results = []
        if enable_keyword and keyword_service:
            keyword_results = keyword_service.search(
                query=search_query,
                top_k=max(keyword_top_k, wide_top),
                technology=None,
                domain=None,
                document_id=scoped_doc,
                file_id=scoped_doc,
            )
        merged = _merge_hybrid_results(vector_results, keyword_results, vector_weight, keyword_weight)
        if merged:
            logger.info(">>> RETRIEVAL: single-scope document-wide search (technology labels mismatch ingest)")
    logger.info(
        ">>> RETRIEVAL: vector=%s keyword=%s merged=%s",
        len(vector_results), len(keyword_results), len(merged)
    )

    if enable_reranker and len(merged) > rerank_top_n:
        logger.info(">>> RERANK: input_chunks=%s -> keeping top_n=%s", len(merged), rerank_top_n)
        reranker = get_reranker_service()
        context_chunks = reranker.rerank(query=rerank_query, chunks=merged, top_n=rerank_top_n)
        logger.info(">>> RERANK: output_chunks=%s (sent to LLM)", len(context_chunks))
    else:
        context_chunks = merged[:rerank_top_n] if enable_reranker else merged[:max_chunks_no_rerank]
        logger.info(
            ">>> RETRIEVAL: rerank skipped (chunks=%s), using %s chunks for LLM",
            len(merged), len(context_chunks),
        )
    return context_chunks


def _normalize_similarity(score: float) -> float:
    """
    Normalize a vector search score into an approximate similarity in [0, 1].
    If score is already in [0,1], return it.
    If score > 1, treat as distance-like and map to (0,1] via 1/(1+score).
    """
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    if s <= 1.0:
        return s
    return 1.0 / (1.0 + s)


def _compute_retrieval_confidence(context_chunks: list) -> tuple:
    """
    Compute average similarity and spread from context chunks.
    Returns (avg_similarity, spread, count).
    """
    if not context_chunks:
        return 0.0, 0.0, 0
    sims = []
    for c in context_chunks:
        score = c.get("score", 0.0)
        sims.append(_normalize_similarity(score))
    if not sims:
        return 0.0, 0.0, 0
    avg_sim = sum(sims) / len(sims)
    spread = max(sims) - min(sims) if len(sims) > 1 else 0.0
    return avg_sim, spread, len(sims)


def _insufficient_context_message(available_technologies: Optional[list] = None) -> str:
    base = "I couldn't find enough information in the uploaded documents to answer that."
    if available_technologies:
        topics = ", ".join(available_technologies)
        return f"{base} Available topics: {topics}."
    return base


def _normalize_bm25(score: float) -> float:
    """
    Normalize BM25 score (lower is better) into similarity in (0,1].
    """
    try:
        s = float(score)
    except Exception:
        return 0.0
    if s < 0:
        return 0.0
    return 1.0 / (1.0 + s)


def _chunk_key(item: dict) -> str:
    chunk_id = item.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    doc_id = item.get("document_id", "")
    chunk_index = item.get("chunk_index", "")
    file_name = item.get("file_name", "")
    return f"{doc_id}:{chunk_index}:{file_name}"


def _merge_hybrid_results(
    vector_results: list,
    keyword_results: list,
    vector_weight: float,
    keyword_weight: float,
) -> list:
    merged = {}

    # Add vector results
    for v in vector_results:
        key = _chunk_key(v)
        vec_sim = _normalize_similarity(v.get("score", 0.0))
        item = merged.get(key, dict(v))
        item["vector_score"] = vec_sim
        item["keyword_score"] = item.get("keyword_score", 0.0)
        item["hybrid_score"] = item.get("hybrid_score", 0.0) + (vector_weight * vec_sim)
        # Use hybrid score as primary score for downstream gating/sorting
        item["score"] = item["hybrid_score"]
        merged[key] = item

    # Add keyword results
    for k in keyword_results:
        key = _chunk_key(k)
        kw_sim = _normalize_bm25(k.get("score", 0.0))
        item = merged.get(key, dict(k))
        item["vector_score"] = item.get("vector_score", 0.0)
        item["keyword_score"] = kw_sim
        item["hybrid_score"] = item.get("hybrid_score", 0.0) + (keyword_weight * kw_sim)
        item["score"] = item["hybrid_score"]
        merged[key] = item

    merged_list = list(merged.values())
    merged_list.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return merged_list


def _hydrate_chunks(context_chunks: list) -> list:
    """
    Enrich chunks with metadata, chunk_type, and raw table/code/image content.
    Prefer chunk_store for metadata when chunk_id is present so raw_code_id/raw_table_id
    are never lost to vector-store truncation.
    """
    if not context_chunks:
        return context_chunks

    chunk_store = get_chunk_store()
    raw_store = get_raw_block_store()

    chunk_ids = [c.get("chunk_id") for c in context_chunks if c.get("chunk_id")]
    chunk_map = chunk_store.get_chunks(chunk_ids) if chunk_ids else {}

    for c in context_chunks:
        cid = c.get("chunk_id")
        if cid and cid in chunk_map:
            row = chunk_map[cid]
            c["metadata_json"] = row.get("metadata_json", "")
            c["chunk_type"] = row.get("chunk_type", "paragraph")
            c["page_number"] = row.get("page_number", -1)
            c["section_title"] = row.get("section_title", "")
            c["text"] = row.get("retrieval_text", "") or c.get("text", "")

        meta = {}
        if c.get("metadata_json"):
            try:
                meta = json.loads(c.get("metadata_json"))
            except Exception:
                meta = {}
        c["metadata"] = meta

        chunk_type = c.get("chunk_type")
        if chunk_type == "table_summary":
            raw_id = meta.get("raw_table_id")
            if raw_id:
                c["raw_table"] = raw_store.get_table(raw_id)
        elif chunk_type == "code_summary":
            raw_id = meta.get("raw_code_id")
            if raw_id:
                c["raw_code"] = raw_store.get_code(raw_id)
        elif chunk_type == "image_caption":
            raw_id = meta.get("raw_image_id")
            if raw_id:
                c["raw_image"] = raw_store.get_image(raw_id)

    return context_chunks


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG chat: classify query, search vectors, build context, call LLM.
    Creates a new session if session_id is not provided.
    """
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    logger.info(">>> CHAT: query=%s", repr(query[:80] + ("..." if len(query) > 80 else "")))
    chat_service = get_chat_service()

    # Resolve or create session
    session_id = request.session_id
    if not session_id:
        session = chat_service.create_session(title="New Chat")
        session_id = session["id"]
        logger.info(">>> CHAT: created new session_id=%s", session_id[:8] + "...")
    else:
        logger.info(">>> CHAT: using session_id=%s", session_id[:8] + "...")

    # Ensure session exists if session_id was provided
    if request.session_id and not chat_service.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # ==================== DOCUMENT SCOPE (SEGREGATION) ====================
    # Prefer explicit scope from request. If absent, fall back to session-attached docs.
    requested_scope = []
    if getattr(request, "file_ids", None):
        requested_scope = [fid for fid in (request.file_ids or []) if fid]
    elif request.file_id:
        requested_scope = [request.file_id]

    # Persist scope on the server whenever the client sends it (so message 2+ works
    # even if the UI stops re-sending file_ids or session_documents was never PUT).
    if requested_scope:
        try:
            chat_service.set_session_documents(session_id, requested_scope)
        except Exception as e:
            logger.warning(">>> CHAT: could not persist session documents: %s", e)

    session_scope = []
    if not requested_scope:
        try:
            session_scope = chat_service.get_session_documents(session_id)
        except Exception:
            session_scope = []

    scope_file_ids = requested_scope or session_scope

    if not scope_file_ids and settings.CHAT_AUTO_SCOPE_ALL_INGESTED:
        try:
            auto_ids = get_document_store().list_ingested_document_ids()
            if auto_ids:
                scope_file_ids = auto_ids
                logger.info(
                    ">>> CHAT: auto scope — all ingested documents (count=%s)",
                    len(scope_file_ids),
                )
        except Exception as e:
            logger.warning(">>> CHAT: auto scope lookup failed: %s", e)

    if not scope_file_ids:
        if settings.CHAT_AUTO_SCOPE_ALL_INGESTED:
            detail = (
                "No searchable documents yet. Upload PDFs from the Library or Upload tab "
                "and wait until ingestion completes."
            )
        else:
            detail = (
                "No document scope provided. Send file_ids, or attach documents via "
                "/sessions/{session_id}/documents."
            )
        raise HTTPException(status_code=400, detail=detail)

    # Only record the user message after scope is valid (avoids orphan messages on 400).
    chat_service.add_message(session_id, "user", query)

    # Classify query with conversation context (scope = technologies the question relates to)
    recent_messages = chat_service.get_context_messages(session_id, context_window=6)
    technologies, domain, from_context = classify_query_with_context(query, recent_messages)
    logger.info(
        ">>> RETRIEVAL: query_classification technologies=%s domain=%s from_conversation_context=%s",
        technologies, domain, from_context,
    )

    # If context-based classification yields multiple technologies, try to narrow to the last user topic
    topic_context_text = ""
    if from_context and (not technologies or len(technologies) > 1):
        prev_text, prev_techs, prev_domain = _get_last_user_message_with_tech(recent_messages)
        if prev_text and prev_techs:
            topic_context_text = prev_text
            # Prefer a single clear previous topic if available
            technologies = prev_techs
            domain = prev_domain
            logger.info(
                ">>> RETRIEVAL: narrowed_from_last_user_with_tech technologies=%s domain=%s (prev_user=%s)",
                technologies, domain, repr(prev_text[:80] + ("..." if len(prev_text) > 80 else "")),
            )

    # Search query for embedding: when scope came from previous turn, prepend to bias retrieval
    search_query = _build_search_query(query, technologies, from_context, recent_messages, topic_context_text)
    if from_context and technologies:
        logger.info(">>> RETRIEVAL: expanded search_query=%s", repr(search_query[:80] + ("..." if len(search_query) > 80 else "")))

    keyword_service = get_keyword_search_service()
    enable_reranker = getattr(settings, "ENABLE_RERANKER", True)
    initial_top_k = getattr(settings, "RERANK_INITIAL_TOP_K", 20)
    rerank_top_n = getattr(settings, "RERANK_TOP_N", 5)
    max_chunks_no_rerank = getattr(settings, "MAX_CHUNKS_TO_RETRIEVE", 5)
    enable_keyword = getattr(settings, "ENABLE_KEYWORD_SEARCH", True)
    keyword_top_k = getattr(settings, "KEYWORD_TOP_K", initial_top_k)
    vector_weight = getattr(settings, "HYBRID_VECTOR_WEIGHT", 0.7)
    keyword_weight = getattr(settings, "HYBRID_KEYWORD_WEIGHT", 0.3)
    # Vector store is optional; fall back to keyword-only if Zilliz is stopped/unavailable.
    vector_store = None
    embedding_service = None
    try:
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()
    except HTTPException as e:
        # If vector DB is down, keep the app usable via keyword search.
        logger.warning(">>> RETRIEVAL: vector store unavailable (%s). Falling back to keyword-only.", e.detail)
        vector_store = None
        embedding_service = None

    if vector_store is None or embedding_service is None:
        if not enable_keyword or keyword_service is None:
            raise HTTPException(status_code=503, detail="Retrieval is unavailable (vector DB down and keyword search disabled).")
        # Keyword-only retrieval
        technology = technologies[0] if technologies else None
        # Multi-document scope: query each document separately and merge
        context_chunks = []
        per_doc_k = max(1, rerank_top_n // max(1, len(scope_file_ids)))
        for doc_id in scope_file_ids:
            hits = keyword_service.search(
                query=search_query,
                top_k=per_doc_k,
                technology=technology,
                domain=domain,
                document_id=doc_id,
                file_id=doc_id,
            )
            context_chunks.extend(hits or [])
        # Prefer best scores
        context_chunks.sort(key=lambda x: x.get("score", 0.0))
        context_chunks = context_chunks[:rerank_top_n]
        logger.info(">>> RETRIEVAL: keyword-only scoped docs=%s context_chunks=%s", len(scope_file_ids), len(context_chunks))
    else:
        # Multi-document scope: retrieve per doc, then merge + rerank downstream.
        all_chunks = []
        # Allocate top_k per document to keep latency bounded.
        per_doc_initial = max(5, int(initial_top_k / max(1, len(scope_file_ids))))
        for doc_id in scope_file_ids:
            # Make a shallow request-like object with file_id set for filtering
            try:
                scoped_request = request.model_copy(update={"file_id": doc_id, "file_ids": None})
            except Exception:
                scoped_request = request
                scoped_request.file_id = doc_id  # best-effort
            hits = _retrieve_context_chunks(
                search_query=search_query,
                rerank_query=query,
                technologies=technologies,
                domain=domain,
                request=scoped_request,
                vector_store=vector_store,
                embedding_service=embedding_service,
                keyword_service=keyword_service,
                enable_keyword=enable_keyword,
                keyword_top_k=max(3, int(keyword_top_k / max(1, len(scope_file_ids)))),
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                enable_reranker=enable_reranker,
                initial_top_k=per_doc_initial,
                rerank_top_n=max(2, int(rerank_top_n / max(1, len(scope_file_ids)))),
                max_chunks_no_rerank=max_chunks_no_rerank,
            )
            all_chunks.extend(hits or [])

        # Merge duplicates and keep best hybrid_score
        merged_map = {}
        for c in all_chunks:
            key = _chunk_key(c)
            prev = merged_map.get(key)
            if prev is None or c.get("score", 0.0) > prev.get("score", 0.0):
                merged_map[key] = c
        context_chunks = list(merged_map.values())
        context_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        context_chunks = context_chunks[:rerank_top_n]
        logger.info(">>> RETRIEVAL: scoped docs=%s merged context_chunks=%s", len(scope_file_ids), len(context_chunks))

    # Self-refining retrieval loop (single retry with query rewrite)
    enable_query_rewrite = getattr(settings, "ENABLE_QUERY_REWRITE", True)
    if enable_query_rewrite:
        llm_service = get_llm_service()
        try:
            sufficient = llm_service.is_context_sufficient(query, context_chunks)
        except Exception as e:
            logger.warning(">>> RETRIEVAL: sufficiency check failed, skipping rewrite: %s", e)
            sufficient = True

        if not sufficient:
            try:
                rewritten = llm_service.rewrite_query(query, chat_history=recent_messages)
            except Exception as e:
                logger.warning(">>> RETRIEVAL: rewrite failed, skipping retry: %s", e)
                rewritten = query

            if rewritten and rewritten.strip() and rewritten.strip() != query.strip():
                rewrite_search_query = _build_search_query(
                    rewritten, technologies, from_context, recent_messages, topic_context_text
                )
                logger.info(
                    ">>> RETRIEVAL: rewrite_query=%s",
                    repr(rewritten[:120] + ("..." if len(rewritten) > 120 else ""))
                )
                if vector_store is None or embedding_service is None:
                    technology = technologies[0] if technologies else None
                    per_doc_k_rw = max(1, rerank_top_n // max(1, len(scope_file_ids)))
                    ctx_rw: list = []
                    for doc_id in scope_file_ids:
                        ctx_rw.extend(
                            keyword_service.search(
                                query=rewrite_search_query,
                                top_k=per_doc_k_rw,
                                technology=technology,
                                domain=domain,
                                document_id=doc_id,
                                file_id=doc_id,
                            )
                            or []
                        )
                    ctx_rw.sort(key=lambda x: x.get("score", 0.0))
                    context_chunks = ctx_rw[:rerank_top_n]
                    logger.info(">>> RETRIEVAL: keyword-only (rewritten) scoped docs=%s context_chunks=%s", len(scope_file_ids), len(context_chunks))
                else:
                    all_rewrite_chunks: list = []
                    per_doc_initial_rw = max(5, int(initial_top_k / max(1, len(scope_file_ids))))
                    rerank_n_rw = max(2, int(rerank_top_n / max(1, len(scope_file_ids))))
                    for doc_id in scope_file_ids:
                        try:
                            scoped_rw = request.model_copy(update={"file_id": doc_id, "file_ids": None})
                        except Exception:
                            scoped_rw = request
                            setattr(scoped_rw, "file_id", doc_id)
                        hits_rw = _retrieve_context_chunks(
                            search_query=rewrite_search_query,
                            rerank_query=rewritten,
                            technologies=technologies,
                            domain=domain,
                            request=scoped_rw,
                            vector_store=vector_store,
                            embedding_service=embedding_service,
                            keyword_service=keyword_service,
                            enable_keyword=enable_keyword,
                            keyword_top_k=max(3, int(keyword_top_k / max(1, len(scope_file_ids)))),
                            vector_weight=vector_weight,
                            keyword_weight=keyword_weight,
                            enable_reranker=enable_reranker,
                            initial_top_k=per_doc_initial_rw,
                            rerank_top_n=rerank_n_rw,
                            max_chunks_no_rerank=max_chunks_no_rerank,
                        )
                        all_rewrite_chunks.extend(hits_rw or [])
                    merged_rw: dict = {}
                    for c in all_rewrite_chunks:
                        key = _chunk_key(c)
                        prev = merged_rw.get(key)
                        if prev is None or c.get("score", 0.0) > prev.get("score", 0.0):
                            merged_rw[key] = c
                    context_chunks = list(merged_rw.values())
                    context_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    context_chunks = context_chunks[:rerank_top_n]
                    logger.info(
                        ">>> RETRIEVAL: rewrite merged scoped_docs=%s context_chunks=%s",
                        len(scope_file_ids),
                        len(context_chunks),
                    )

    # Hydrate chunks with raw data (tables/code/images)
    context_chunks = _hydrate_chunks(context_chunks)

    # Chat history for context window
    chat_history = chat_service.get_context_messages(session_id, context_window=6)
    available_technologies = get_document_store().get_technologies()
    logger.info(">>> LLM: context_chunks=%s, chat_history_messages=%s, available_technologies=%s", len(context_chunks), len(chat_history), available_technologies)

    # Retrieval confidence gate
    avg_similarity, score_spread, chunk_count = _compute_retrieval_confidence(context_chunks)
    logger.info(
        ">>> RETRIEVAL: confidence avg_similarity=%.4f spread=%.4f chunks=%s",
        avg_similarity, score_spread, chunk_count,
    )
    # Marginal similarity on huge PDFs often clusters ~0.22–0.28; avoid blocking near threshold.
    if avg_similarity < 0.22 or chunk_count < 2:
        answer = _insufficient_context_message(available_technologies)
        chat_service.add_message(session_id, "assistant", answer)
        detected_technology_str = ", ".join(technologies) if technologies else None
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            sources=None,
            detected_technology=detected_technology_str,
            detected_domain=domain,
        )

    # Generate answer
    llm_service = get_llm_service()
    answer = llm_service.generate_response(
        query=query,
        context_chunks=context_chunks,
        chat_history=chat_history,
        available_technologies=available_technologies,
    )

    # Persist assistant message
    assistant_row = chat_service.add_message(session_id, "assistant", answer)

    # Token/cost logs (per assistant turn). Costs are optional and require env vars.
    usage = getattr(llm_service, "last_usage", None) or {}
    prompt_toks = usage.get("prompt_tokens")
    completion_toks = usage.get("completion_tokens")
    total_toks = usage.get("total_tokens")
    provider = (settings.LLM_PROVIDER or "").lower().strip()
    model_used = getattr(llm_service, "model", None)

    cost_usd = None
    try:
        if prompt_toks is not None and completion_toks is not None:
            if provider == "openai":
                in_rate = getattr(settings, "OPENAI_INPUT_USD_PER_1M", None)
                out_rate = getattr(settings, "OPENAI_OUTPUT_USD_PER_1M", None)
                if in_rate is not None and out_rate is not None:
                    cost_usd = (float(prompt_toks) / 1_000_000.0) * float(in_rate) + (
                        float(completion_toks) / 1_000_000.0
                    ) * float(out_rate)
            elif provider == "groq":
                in_rate = float(getattr(settings, "GROQ_INPUT_USD_PER_1M", 0.59))
                out_rate = float(getattr(settings, "GROQ_OUTPUT_USD_PER_1M", 0.79))
                cost_usd = (float(prompt_toks) / 1_000_000.0) * in_rate + (
                    float(completion_toks) / 1_000_000.0
                ) * out_rate
    except Exception:
        cost_usd = None

    logger.info(
        ">>> TOKENS: session=%s msg_id=%s provider=%s model=%s prompt=%s completion=%s total=%s cost_usd=%s",
        session_id,
        assistant_row.get("id"),
        provider,
        model_used,
        prompt_toks,
        completion_toks,
        total_toks,
        f"{cost_usd:.6f}" if isinstance(cost_usd, (int, float)) else None,
    )

    record_chat_completion(
        session_id=session_id,
        message_id=assistant_row.get("id"),
        query_preview=query,
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
        total_tokens=total_toks,
        model=model_used,
        provider=provider or "openai",
        cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
    )

    # Sources from retrieved chunks (e.g. file names / technologies)
    sources = []
    for c in context_chunks:
        tech = c.get("technology") or c.get("file_name")
        if tech and tech not in sources:
            sources.append(tech if isinstance(tech, str) else str(tech))
    sources = list(sources)[:10]
    logger.info(">>> CHAT: done session_id=%s sources=%s", session_id[:8] + "...", sources)

    detected_technology_str = ", ".join(technologies) if technologies else None
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources if sources else None,
        detected_technology=detected_technology_str,
        detected_domain=domain,
    )
