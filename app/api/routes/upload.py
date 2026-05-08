"""PDF upload endpoint."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.config import settings
from app.models.schemas import UploadResponse, BatchUploadResponse, SingleFileResult
from app.services.pdf_processor import PDFProcessor
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.document_store import get_document_store
from app.utils.helpers import ensure_upload_dir, generate_file_id, get_file_path
import os
import shutil
import traceback
import logging
import sys
from app.services.document_classifier import get_document_classifier
from app.services.keyword_search import get_keyword_search_service
from app.services.chunk_store import get_chunk_store
from app.services.raw_block_store import get_raw_block_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])
print(f"INFO: Upload router created with prefix: /upload")

# Initialize services (lazy initialization for vector_store to handle errors gracefully)
# Use print statements as fallback since logging might not be configured yet
try:
    print("INFO: Initializing services in upload route...")
    pdf_processor = PDFProcessor()
    print("INFO: PDFProcessor initialized")
    
    _chunk_size = settings.CHUNK_SIZE
    _chunk_overlap = settings.CHUNK_OVERLAP
    if getattr(settings, "CHUNK_TARGET_TOKENS", 0) > 0:
        _chunk_size = settings.CHUNK_TARGET_TOKENS * 4
        _chunk_overlap = int(_chunk_size * getattr(settings, "CHUNK_OVERLAP_PERCENT", 12) / 100)
    chunking_service = ChunkingService(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
    print(f"INFO: ChunkingService initialized (chunk_size={_chunk_size}, overlap={_chunk_overlap})")
    
    embedding_service = EmbeddingService(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    print(f"INFO: EmbeddingService initialized (model={settings.OPENAI_EMBEDDING_MODEL})")
    
    document_store = get_document_store()
    print("INFO: DocumentStore initialized (DB-backed)")
except Exception as e:
    print(f"ERROR: Failed to initialize services: {type(e).__name__}: {str(e)}")
    print(traceback.format_exc())
    raise

# Initialize vector store (will be created on first use if initialization fails)
_vector_store = None

def _prepare_chunk_metadata(chunks, pdf_path: str):
    """
    Prepare metadata for each chunk (page numbers, section titles, etc.).
    
    Args:
        chunks: List of text chunks
        pdf_path: Path to the PDF file
        
    Returns:
        List of metadata dictionaries for each chunk
    """
    import pdfplumber
    
    metadata_list = []
    
    try:
        # Try to extract page information using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # For now, we'll assign chunks to pages based on approximate position
            # This is a simple heuristic - can be improved with better PDF parsing
            total_text_length = sum(len(chunk) for chunk in chunks)
            current_pos = 0
            
            for chunk in chunks:
                chunk_meta = {
                    "file_type": "pdf",
                    "language": "en",  # Default, can be enhanced with language detection
                    "has_tables": False,
                    "has_images": False,
                    "chunk_type": "paragraph",
                    "section_title": "",
                    "page_number": -1,
                    "metadata_json": "{}"
                }
                
                # Estimate page number based on text position
                # This is approximate - better would be to track actual page breaks
                if total_text_length > 0:
                    progress = current_pos / total_text_length
                    estimated_page = int(progress * total_pages) + 1
                    chunk_meta["page_number"] = min(estimated_page, total_pages)
                
                # Check if chunk might contain table-like content (simple heuristic)
                if "\t" in chunk or "|" in chunk or chunk.count("\n") > chunk.count(" ") * 0.3:
                    chunk_meta["has_tables"] = True
                
                # Check if chunk mentions images (simple heuristic)
                image_keywords = ["figure", "image", "diagram", "chart", "graph"]
                if any(keyword in chunk.lower() for keyword in image_keywords):
                    chunk_meta["has_images"] = True
                
                # Detect chunk type (simple heuristics)
                if len(chunk) < 100 and chunk.count("\n") < 3:
                    chunk_meta["chunk_type"] = "heading"
                elif chunk.count("\n") > 10:
                    chunk_meta["chunk_type"] = "list"
                
                current_pos += len(chunk)
                metadata_list.append(chunk_meta)
                
    except Exception as e:
        print(f">>> METADATA: Error extracting metadata: {e}, using defaults", flush=True)
        # If extraction fails, use default metadata
        for chunk in chunks:
            metadata_list.append({
                "file_type": "pdf",
                "language": "en",
                "has_tables": False,
                "has_images": False,
                "chunk_type": "paragraph",
                "section_title": "",
                "page_number": -1,
                "metadata_json": "{}"
            })
    
    return metadata_list


def get_vector_store():
    """Get or initialize vector store."""
    import sys
    global _vector_store
    if _vector_store is None:
        print(f">>> VECTOR STORE: Initializing...", flush=True)
        print(f">>> VECTOR STORE: URI={settings.ZILLIZ_URI[:50]}...", flush=True)
        print(f">>> VECTOR STORE: Collection={settings.ZILLIZ_COLLECTION_NAME}", flush=True)
        sys.stdout.flush()
        try:
            _vector_store = VectorStore(
                uri=settings.ZILLIZ_URI,
                token=settings.ZILLIZ_TOKEN,
                collection_name=settings.ZILLIZ_COLLECTION_NAME
            )
            print(">>> VECTOR STORE: Initialized successfully!", flush=True)
            sys.stdout.flush()
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f">>> VECTOR STORE ERROR: {type(e).__name__}: {str(e)}", flush=True)
            print(f">>> TRACEBACK:\n{error_trace}", flush=True)
            sys.stdout.flush()
            msg = str(e)
            # Common Zilliz serverless state: cluster is paused/stopped
            if "cluster status STOPPED" in msg or "status STOPPED" in msg or "code=90153" in msg:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Vector DB (Zilliz) is STOPPED/paused. Resume the Zilliz cluster "
                        "in Zilliz Cloud, then retry upload/chat."
                    ),
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize vector store: {str(e)}"
            )
    return _vector_store


def best_effort_clear_prior_ingest(document_id: str) -> None:
    """
    Remove any prior chunks/FTS/raw/vectors for this document_id before a fresh ingest.
    Safe for brand-new ids (no rows). Prevents duplicate rows and odd state on re-runs
    (e.g. FAILED → retry) without touching other documents.
    """
    did = (document_id or "").strip()
    if not did:
        logger.warning("best_effort_clear_prior_ingest: skipped empty document_id")
        return
    try:
        get_keyword_search_service().delete_document(did)
    except Exception as e:
        logger.warning("clear_prior_ingest keyword: %s", e)
    try:
        get_chunk_store().delete_document(did)
    except Exception as e:
        logger.warning("clear_prior_ingest chunks: %s", e)
    try:
        get_raw_block_store().delete_document(did)
    except Exception as e:
        logger.warning("clear_prior_ingest raw_blocks: %s", e)
    try:
        get_vector_store().delete_by_document_id(did)
    except Exception as e:
        logger.warning("clear_prior_ingest zilliz: %s", e)


def best_effort_clear_failed_ingest(document_id: str) -> None:
    """After ingest failure, remove partial rows from keyword index, chunks, raw blocks, and Zilliz."""
    did = (document_id or "").strip()
    if not did:
        return
    try:
        get_keyword_search_service().delete_document(did)
    except Exception as e:
        logger.warning("failed_ingest_cleanup keyword: %s", e)
    try:
        get_chunk_store().delete_document(did)
    except Exception as e:
        logger.warning("failed_ingest_cleanup chunks: %s", e)
    try:
        get_raw_block_store().delete_document(did)
    except Exception as e:
        logger.warning("failed_ingest_cleanup raw_blocks: %s", e)
    try:
        get_vector_store().delete_by_document_id(did)
    except Exception as e:
        logger.warning("failed_ingest_cleanup zilliz: %s", e)


@router.post("/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    
    - Computes file hash to prevent duplicates
    - Checks document registry for existing document
    - If new: Extracts text, chunks, generates embeddings, stores in vector database
    - If duplicate: Returns existing document info without re-processing
    """
    import sys
    print(f">>> UPLOAD ROUTE: Received file: {file.filename}", flush=True)
    sys.stdout.flush()
    
    document_id = None
    file_path = None
    
    try:
        print(">>> STEP 1: Starting PDF upload processing...", flush=True)
        sys.stdout.flush()
        # Validate file type
        if not file.filename.endswith('.pdf'):
            print(f">>> VALIDATION ERROR: Invalid file type: {file.filename}", flush=True)
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # STEP 2: Read file content and compute hash BEFORE saving
        print(">>> STEP 2: Reading file and computing hash...", flush=True)
        file_content = await file.read()
        file_hash = document_store.compute_hash_from_bytes(file_content)
        print(f">>> STEP 2 DONE: File hash: {file_hash[:16]}...", flush=True)
        
        # STEP 3: Check for duplicate
        print(">>> STEP 3: Checking for duplicate document...", flush=True)
        existing_doc = document_store.get_by_hash(file_hash)
        if existing_doc:
            existing_doc_id = existing_doc["document_id"]
            existing_status = existing_doc.get("status")
            existing_chunks = existing_doc.get("chunk_count", 0)
            print(f">>> DUPLICATE DETECTED: Document {existing_doc_id} already exists (status: {existing_status}, chunks: {existing_chunks})", flush=True)
            
            # If already ingested, return existing document info
            if (existing_status or "").upper() == "INGESTED":
                return UploadResponse(
                    message="PDF already processed (duplicate detected). Using existing document.",
                    file_id=existing_doc_id,  # Return document_id as file_id for backward compatibility
                    chunks_created=existing_chunks
                )
            # If failed, allow re-processing
            elif (existing_status or "").upper() == "FAILED":
                print(f">>> Previous ingestion failed, re-processing document {existing_doc_id}...", flush=True)
                document_id = existing_doc_id
            # If uploaded but not ingested, continue with same document_id
            else:
                document_id = existing_doc_id
                print(f">>> Document {document_id} was uploaded but not ingested, continuing...", flush=True)
        else:
            # New document - register it
            print(">>> STEP 3a: Registering new document (DB)...", flush=True)
            document_id = document_store.create_document(
                file_name=file.filename,
                file_hash=file_hash,
                status="UPLOADED",
            )
            print(f">>> STEP 3a DONE: Registered document {document_id}", flush=True)
        
        # STEP 4: Save uploaded file
        print(">>> STEP 4: Saving uploaded file...", flush=True)
        ensure_upload_dir(settings.UPLOAD_DIR)
        file_extension = os.path.splitext(file.filename)[1]
        saved_filename = f"{document_id}{file_extension}"
        file_path = get_file_path(settings.UPLOAD_DIR, saved_filename)
        
        # Reset file pointer and save
        await file.seek(0)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        print(f">>> STEP 4 DONE: File saved to {file_path}", flush=True)
        
        # STEP 5: Extract blocks from PDF
        print(">>> STEP 5: Extracting structured blocks from PDF...", flush=True)
        blocks = pdf_processor.extract_blocks(str(file_path))
        if getattr(settings, "STORE_EXTRACTED_IMAGES", True):
            try:
                PDFProcessor.persist_extracted_images(
                    str(file_path), document_id, blocks, settings.UPLOAD_DIR
                )
            except Exception as img_err:
                logger.warning(f"Image extraction/persist failed: {img_err}")
        print(f">>> STEP 5 DONE: Extracted {len(blocks)} blocks", flush=True)

        # STEP 5b: Classify document for technology/domain (same as batch)
        sample_text = ""
        for b in blocks[:200]:
            sample_text += (b.get("content") or "") + "\n"
            if len(sample_text) >= 50000:
                break
        doc_classifier = get_document_classifier()
        technology, domain = doc_classifier.classify(file.filename, sample_text)
        print(f">>> Classified {file.filename} as: {technology}/{domain}", flush=True)
        
        # STEP 6: Chunk blocks with type-aware logic
        print(">>> STEP 6: Chunking blocks...", flush=True)
        chunks, chunk_metadata = chunking_service.chunk_blocks(
            blocks,
            document_id=document_id,
            file_name=file.filename,
            file_id=document_id,
        )
        print(f">>> STEP 6 DONE: Created {len(chunks)} chunks", flush=True)
        
        if not chunks:
            print(">>> ERROR: No chunks created from PDF", flush=True)
            document_store.mark_failed(document_id=document_id, error="No text chunks could be created from PDF")
            raise HTTPException(status_code=400, detail="No text chunks could be created from PDF")
        
        # STEP 7: Persist chunks in canonical chunks table
        print(">>> STEP 6b: Storing chunks in chunks table...", flush=True)
        best_effort_clear_prior_ingest(document_id)
        chunk_store = get_chunk_store()
        chunk_ids = chunk_store.insert_chunks(chunks, chunk_metadata, document_id)
        print(f">>> STEP 6b DONE: Stored {len(chunk_ids)} chunks", flush=True)

        # STEP 7: Generate embeddings
        print(">>> STEP 7: Generating embeddings...", flush=True)
        embeddings = embedding_service.generate_embeddings(chunks)
        print(f">>> STEP 7 DONE: Generated {len(embeddings)} embeddings", flush=True)
        
        # STEP 7a: Chunk metadata already prepared by block chunker
        print(">>> STEP 7a: Using block-derived chunk metadata...", flush=True)
        print(f">>> STEP 7a DONE: Prepared metadata for {len(chunk_metadata)} chunks", flush=True)
        
        # STEP 8: Store in vector database (with classified technology/domain)
        print(">>> STEP 8: Initializing vector store...", flush=True)
        vector_store = get_vector_store()
        print(">>> STEP 8a: Vector store ready, inserting chunks...", flush=True)
        file_size = len(file_content)
        chunks_created = vector_store.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            file_hash=file_hash,
            file_name=file.filename,
            file_size=file_size,
            chunk_metadata=chunk_metadata,
            file_id=document_id,  # Keep for backward compatibility
            chunk_ids=chunk_ids,
            technology=technology,
            domain=domain,
        )
        print(f">>> STEP 8 DONE: Inserted {chunks_created} chunks (technology={technology}, domain={domain})", flush=True)

        # STEP 8b: Index chunks for keyword search (FTS) with same technology/domain
        try:
            keyword_service = get_keyword_search_service()
            # Clear any previous entries for this document (re-ingest)
            keyword_service.delete_document(document_id)
            keyword_service.index_chunks(
                chunks=chunks,
                chunk_ids=chunk_ids,
                document_id=document_id,
                file_name=file.filename,
                technology=technology,
                domain=domain,
                file_id=document_id,
                start_chunk_index=0,
            )
            print(f">>> STEP 8b DONE: Indexed {len(chunks)} chunks for keyword search", flush=True)
        except Exception as e:
            logger.warning(f"Keyword indexing failed (single upload): {e}")
        
        # STEP 9: Mark document as ingested with technology/domain and optional PDF path (object storage)
        print(">>> STEP 9: Marking document as ingested...", flush=True)
        pdf_path_for_registry = str(file_path) if getattr(settings, "STORE_PDF_AFTER_INGEST", True) else None
        document_store.mark_ingested(
            document_id=document_id,
            chunk_count=chunks_created,
            technology=technology,
            domain=domain,
            pdf_path=pdf_path_for_registry,
        )
        if not getattr(settings, "STORE_PDF_AFTER_INGEST", True) and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed PDF after ingest (STORE_PDF_AFTER_INGEST=False): {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove PDF after ingest: {e}")
        print(f">>> STEP 9 DONE: Document {document_id} marked as ingested (technology={technology}, domain={domain})", flush=True)
        
        print(f">>> SUCCESS: PDF processing completed for {file.filename}", flush=True)
        return UploadResponse(
            message="PDF processed and stored successfully",
            file_id=document_id,  # Return document_id as file_id for backward compatibility
            chunks_created=chunks_created
        )
    
    except HTTPException as e:
        # Re-raise HTTP exceptions as-is, but log them
        print(f">>> HTTP ERROR: {e.status_code} - {e.detail}", flush=True)
        sys.stdout.flush()
        # Mark document as failed if we have a document_id
        if document_id:
            try:
                document_store.mark_failed(document_id=document_id, error=str(e.detail))
            except Exception as mark_error:
                logger.warning(f"Failed to mark document as failed: {mark_error}")
            best_effort_clear_failed_ingest(document_id)
        raise
    except Exception as e:
        print(f">>> EXCEPTION: {type(e).__name__}: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        
        # Mark document as failed if we have a document_id
        if document_id:
            try:
                document_store.mark_failed(document_id=document_id, error=str(e))
            except Exception as mark_error:
                logger.warning(f"Failed to mark document as failed: {mark_error}")
            best_effort_clear_failed_ingest(document_id)
        
        # Clean up on error
        if file_path and file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file: {cleanup_error}")
        
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error processing PDF: {type(e).__name__}: {str(e)}")
        logger.error(f"Full traceback:\n{error_trace}")
        
        # Return detailed error message
        error_detail = str(e)
        error_type = type(e).__name__
        
        if "Failed to initialize vector store" in error_detail:
            error_detail += ". Please check your Zilliz credentials in .env file."
        elif "OpenAI" in error_detail or "API key" in error_detail or "authentication" in error_detail.lower():
            error_detail += ". Please check your OpenAI API key in .env file."
        elif "Zilliz" in error_detail or "Milvus" in error_detail:
            error_detail += ". Please check your Zilliz credentials and network connectivity."
        
        logger.error(f"Returning error to client: {error_detail}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing PDF ({error_type}): {error_detail}"
        )


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple PDF files in batch.
    
    - Processes each file independently
    - Returns individual results for each file (success/error/duplicate)
    - Continues processing even if some files fail
    - Includes technology/domain detection for each file
    """

    
    print(f">>> BATCH UPLOAD: Received {len(files)} files", flush=True)
    sys.stdout.flush()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate all files are PDFs
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed"
            )
    
    results = []
    successful = 0
    failed = 0
    
    # Get document classifier for technology detection
    doc_classifier = get_document_classifier()
    
    # Process each file
    for idx, file in enumerate(files, 1):
        print(f"\n>>> BATCH UPLOAD: Processing file {idx}/{len(files)}: {file.filename}", flush=True)
        sys.stdout.flush()
        
        document_id = None
        file_path = None
        
        try:
            # Read file content and compute hash
            file_content = await file.read()
            file_hash = document_store.compute_hash_from_bytes(file_content)
            
            # Check for duplicate
            existing_doc = document_store.get_by_hash(file_hash)
            if existing_doc:
                existing_doc_id = existing_doc["document_id"]
                existing_status = existing_doc.get("status")
                existing_chunks = existing_doc.get("chunk_count", 0)
                existing_tech = existing_doc.get("technology", "general")
                existing_domain = existing_doc.get("domain", "general")
                
                print(f">>> DUPLICATE: {file.filename} already exists as {existing_doc_id}", flush=True)
                
                if (existing_status or "").upper() == "INGESTED":
                    results.append(SingleFileResult(
                        file_name=file.filename,
                        file_id=existing_doc_id,
                        chunks_created=existing_chunks,
                        technology=existing_tech,
                        domain=existing_domain,
                        status="duplicate",
                        message="File already processed (duplicate detected)"
                    ))
                    continue
                else:
                    # If failed or uploaded but not ingested, re-process
                    document_id = existing_doc_id
                    print(f">>> Re-processing existing document {document_id}", flush=True)
            else:
                # New document - create record (will be updated after classification)
                document_id = document_store.create_document(
                    file_name=file.filename,
                    file_hash=file_hash,
                    technology="general",
                    domain="general",
                    status="UPLOADED",
                )
                print(f">>> Registered new document {document_id}", flush=True)
            
            # Save file
            ensure_upload_dir(settings.UPLOAD_DIR)
            file_extension = os.path.splitext(file.filename)[1]
            saved_filename = f"{document_id}{file_extension}"
            file_path = get_file_path(settings.UPLOAD_DIR, saved_filename)
            
            await file.seek(0)
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
            
            # Extract text (for classification) and blocks (for chunking)
            blocks = pdf_processor.extract_blocks(str(file_path))
            if getattr(settings, "STORE_EXTRACTED_IMAGES", True):
                try:
                    PDFProcessor.persist_extracted_images(
                        str(file_path), document_id, blocks, settings.UPLOAD_DIR
                    )
                except Exception as img_err:
                    logger.warning(f"Image extraction failed for {file.filename}: {img_err}")
            text = pdf_processor.extract_text(str(file_path))
            print(f">>> Extracted {len(text)} characters, {len(blocks)} blocks from {file.filename}", flush=True)
            
            # Classify document to detect technology/domain
            technology, domain = doc_classifier.classify(file.filename, text)
            print(f">>> Classified {file.filename} as: {technology}/{domain}", flush=True)
            
            # Chunk blocks
            chunks, chunk_metadata = chunking_service.chunk_blocks(
                blocks,
                document_id=document_id,
                file_name=file.filename,
                file_id=document_id,
            )
            print(f">>> Created {len(chunks)} chunks from {file.filename}", flush=True)
            
            if not chunks:
                document_store.mark_failed(document_id=document_id, error="No text chunks could be created")
                results.append(SingleFileResult(
                    file_name=file.filename,
                    file_id=document_id,
                    chunks_created=0,
                    technology=technology,
                    domain=domain,
                    status="error",
                    message="No text chunks could be created from PDF"
                ))
                failed += 1
                continue
            
            # Store chunks in canonical chunks table
            best_effort_clear_prior_ingest(document_id)
            chunk_store = get_chunk_store()
            chunk_ids = chunk_store.insert_chunks(chunks, chunk_metadata, document_id)

            # Generate embeddings
            embeddings = embedding_service.generate_embeddings(chunks)
            print(f">>> Generated {len(embeddings)} embeddings for {file.filename}", flush=True)
            
            # Prepare metadata (already from blocks)
            
            # Store in vector database
            vector_store = get_vector_store()
            file_size = len(file_content)
            chunks_created = vector_store.insert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
                file_hash=file_hash,
                file_name=file.filename,
                file_size=file_size,
                technology=technology,
                domain=domain,
                chunk_metadata=chunk_metadata,
                file_id=document_id,
                chunk_ids=chunk_ids,
            )

            # Index chunks for keyword search (FTS)
            try:
                keyword_service = get_keyword_search_service()
                keyword_service.delete_document(document_id)
                keyword_service.index_chunks(
                    chunks=chunks,
                    chunk_ids=chunk_ids,
                    document_id=document_id,
                    file_name=file.filename,
                    technology=technology,
                    domain=domain,
                    file_id=document_id,
                    start_chunk_index=0,
                )
            except Exception as e:
                logger.warning(f"Keyword indexing failed (batch upload): {e}")
            
            # Mark as ingested with technology/domain and optional PDF path (object storage)
            pdf_path_for_registry = str(file_path) if getattr(settings, "STORE_PDF_AFTER_INGEST", True) else None
            document_store.mark_ingested(
                document_id=document_id,
                chunk_count=chunks_created,
                technology=technology,
                domain=domain,
                pdf_path=pdf_path_for_registry,
            )
            if not getattr(settings, "STORE_PDF_AFTER_INGEST", True) and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            
            print(f">>> SUCCESS: {file.filename} processed ({chunks_created} chunks, {technology}/{domain})", flush=True)
            
            results.append(SingleFileResult(
                file_name=file.filename,
                file_id=document_id,
                chunks_created=chunks_created,
                technology=technology,
                domain=domain,
                status="success",
                message="PDF processed and stored successfully"
            ))
            successful += 1
            
        except Exception as e:
            error_msg = str(e)
            print(f">>> ERROR processing {file.filename}: {error_msg}", flush=True)
            print(traceback.format_exc(), flush=True)
            sys.stdout.flush()
            
            # Mark as failed if we have a document_id
            if document_id:
                try:
                    document_store.mark_failed(document_id=document_id, error=error_msg)
                except Exception as mark_error:
                    logger.warning(f"Failed to mark document as failed: {mark_error}")
                best_effort_clear_failed_ingest(document_id)
            
            # Clean up file on error
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            
            results.append(SingleFileResult(
                file_name=file.filename,
                file_id=document_id,
                chunks_created=0,
                technology="general",
                domain="general",
                status="error",
                message=f"Error processing PDF: {error_msg}"
            ))
            failed += 1
    
    # Summary
    print(f"\n>>> BATCH UPLOAD COMPLETE: {successful} successful, {failed} failed, {len(files) - successful - failed} duplicates", flush=True)
    sys.stdout.flush()
    
    return BatchUploadResponse(
        message=f"Batch upload completed: {successful} successful, {failed} failed",
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results
    )
