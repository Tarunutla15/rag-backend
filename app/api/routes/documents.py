"""Document library routes: list, view PDF, replace/update, and full delete."""
from pathlib import Path
from typing import List, Optional, Dict, Any
import os
import shutil
import traceback
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.config import settings
from app.models.schemas import DocumentInfo, UploadResponse
from app.services.document_store import get_document_store
from app.services.chat_service import get_chat_service
from app.services.chunk_store import get_chunk_store
from app.services.raw_block_store import get_raw_block_store
from app.services.keyword_search import get_keyword_search_service
from app.services.document_classifier import get_document_classifier
from app.services.pdf_processor import PDFProcessor
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.api.routes.upload import get_vector_store, best_effort_clear_failed_ingest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def purge_document_everywhere(document_id: str) -> Dict[str, Any]:
    """
    Remove all indexed data and files for this document:
    session scope links, keyword FTS, SQL chunks/raw, Zilliz vectors, local PDF + extracted images, documents row.
    """
    errors: List[str] = []
    store = get_document_store()
    if not store.get_document(document_id):
        return {"deleted": False, "document_id": document_id, "errors": ["Document not found"]}

    try:
        get_chat_service().remove_document_from_all_sessions(document_id)
    except Exception as e:
        errors.append(f"session_documents: {e}")

    try:
        get_keyword_search_service().delete_document(document_id)
    except Exception as e:
        errors.append(f"chunks_fts: {e}")

    try:
        get_chunk_store().delete_document(document_id)
    except Exception as e:
        errors.append(f"chunks: {e}")

    try:
        get_raw_block_store().delete_document(document_id)
    except Exception as e:
        errors.append(f"raw_blocks: {e}")

    try:
        get_vector_store().delete_by_document_id(document_id)
    except Exception as e:
        errors.append(f"zilliz: {e}")

    upload_root = Path(settings.UPLOAD_DIR)
    pdf_path = upload_root / f"{document_id}.pdf"
    try:
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception as e:
        errors.append(f"pdf_file: {e}")

    img_dir = upload_root / "images" / document_id
    try:
        if img_dir.exists() and img_dir.is_dir():
            shutil.rmtree(img_dir, ignore_errors=True)
    except Exception as e:
        errors.append(f"images_dir: {e}")

    try:
        store.delete_document_row(document_id)
    except Exception as e:
        errors.append(f"documents_table: {e}")

    return {"deleted": True, "document_id": document_id, "errors": errors}


def _to_document_info(doc: dict) -> DocumentInfo:
    return DocumentInfo(
        document_id=doc.get("document_id") or doc.get("document_id") or "",
        file_name=doc.get("file_name") or "",
        status=doc.get("status"),
        chunk_count=int(doc.get("chunk_count") or 0),
        technology=doc.get("technology"),
        domain=doc.get("domain"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        pdf_path=doc.get("pdf_path"),
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(status: Optional[str] = None) -> List[DocumentInfo]:
    """List all documents in the registry."""
    store = get_document_store()
    docs = store.list_documents()
    if status:
        docs = [d for d in docs if (d.get("status") or "").upper() == status.upper()]
    return [_to_document_info(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str) -> DocumentInfo:
    """Get a single document's metadata."""
    store = get_document_store()
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_document_info(doc)


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> Dict[str, Any]:
    """
    Permanently delete a document: Supabase/SQLite chunks + FTS + raw blocks,
    Zilliz vectors, session scope links, local PDF + image folder, and the documents row.
    """
    result = purge_document_everywhere(document_id)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.get("/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """Return the stored PDF for a document_id (if present on disk)."""
    # Stored as uploads/{document_id}.pdf in current backend
    upload_dir = Path(settings.UPLOAD_DIR)
    pdf_path = upload_dir / f"{document_id}.pdf"
    if not pdf_path.exists():
        # fallback: if registry stored another path
        store = get_document_store()
        doc = store.get_document(document_id) or {}
        p = doc.get("pdf_path")
        if p:
            pdf_path = Path(p)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found for this document")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=os.path.basename(str(pdf_path)),
    )


@router.post("/{document_id}/replace", response_model=UploadResponse)
async def replace_document(document_id: str, file: UploadFile = File(...)):
    """
    Replace/update an existing document while keeping the same document_id.

    This clears old chunks (SQL + vector + keyword index + raw blocks), then ingests the new PDF.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    store = get_document_store()
    existing = store.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    # Read bytes and compute hash
    file_bytes = await file.read()
    new_hash = store.compute_hash_from_bytes(file_bytes)

    # Prevent collisions: if this file already exists under another doc_id, block replace
    dup = store.get_by_hash(new_hash)
    if dup and dup.get("document_id") != document_id:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate detected: this PDF already exists as document {dup.get('document_id')}",
        )

    # Best-effort cleanup of old content
    try:
        get_keyword_search_service().delete_document(document_id)
    except Exception:
        pass
    try:
        get_chunk_store().delete_document(document_id)
    except Exception:
        pass
    try:
        get_raw_block_store().delete_document(document_id)
    except Exception:
        pass
    try:
        get_vector_store().delete_by_document_id(document_id)
    except Exception:
        # Vector DB may be stopped; replacement can still proceed for RelDB
        pass

    # Update DB document row in-place (keep document_id)
    store.update_document(
        document_id,
        {
            "file_name": file.filename,
            "file_hash": new_hash,
            "status": "UPLOADED",
            "chunk_count": 0,
            "error": None,
        },
    )

    # Save new PDF bytes over the same uploads/{document_id}.pdf path
    from app.utils.helpers import ensure_upload_dir, get_file_path
    ensure_upload_dir(settings.UPLOAD_DIR)
    pdf_path = get_file_path(settings.UPLOAD_DIR, f"{document_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    # Ingest (similar to upload.py but anchored to document_id)
    try:
        pdf_processor = PDFProcessor()

        _chunk_size = settings.CHUNK_SIZE
        _chunk_overlap = settings.CHUNK_OVERLAP
        if getattr(settings, "CHUNK_TARGET_TOKENS", 0) > 0:
            _chunk_size = settings.CHUNK_TARGET_TOKENS * 4
            _chunk_overlap = int(_chunk_size * getattr(settings, "CHUNK_OVERLAP_PERCENT", 12) / 100)
        chunking_service = ChunkingService(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)

        embedding_service = EmbeddingService(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_EMBEDDING_MODEL)

        blocks = pdf_processor.extract_blocks(str(pdf_path))

        # classify (tech/domain) and store back to registry
        sample_text = ""
        for b in blocks[:200]:
            sample_text += (b.get("content") or "") + "\n"
            if len(sample_text) >= 50000:
                break
        technology, domain = get_document_classifier().classify(file.filename, sample_text)

        chunks, chunk_metadata = chunking_service.chunk_blocks(
            blocks,
            document_id=document_id,
            file_name=file.filename,
            file_id=document_id,
        )
        if not chunks:
            store.mark_failed(document_id=document_id, error="No text chunks could be created from PDF")
            raise HTTPException(status_code=400, detail="No text chunks could be created from PDF")

        chunk_ids = get_chunk_store().insert_chunks(chunks, chunk_metadata, document_id)
        embeddings = embedding_service.generate_embeddings(chunks)

        # vector insert (best-effort)
        try:
            vector_store = get_vector_store()
            vector_store.insert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
                file_hash=new_hash,
                file_name=file.filename,
                file_size=len(file_bytes),
                chunk_metadata=chunk_metadata,
                file_id=document_id,
                chunk_ids=chunk_ids,
                technology=technology,
                domain=domain,
            )
        except Exception as ve:
            logger.warning("Vector insert failed during replace: %s", ve)

        # keyword index
        try:
            ks = get_keyword_search_service()
            ks.delete_document(document_id)
            ks.index_chunks(
                chunks=chunks,
                chunk_ids=chunk_ids,
                document_id=document_id,
                file_name=file.filename,
                technology=technology,
                domain=domain,
                file_id=document_id,
                start_chunk_index=0,
            )
        except Exception as ke:
            logger.warning("Keyword indexing failed during replace: %s", ke)

        store.mark_ingested(
            document_id=document_id,
            chunk_count=len(chunks),
            technology=technology,
            domain=domain,
            pdf_path=str(pdf_path) if getattr(settings, "STORE_PDF_AFTER_INGEST", True) else None,
        )

        return UploadResponse(
            message="PDF replaced and re-ingested successfully",
            file_id=document_id,
            chunks_created=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Replace ingestion failed: %s\n%s", e, traceback.format_exc())
        try:
            store.mark_failed(document_id=document_id, error=str(e))
        except Exception:
            pass
        best_effort_clear_failed_ingest(document_id)
        raise HTTPException(status_code=500, detail=f"Error replacing document: {e}")

