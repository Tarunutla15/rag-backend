from fastapi.testclient import TestClient
from app.main import app
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

client = TestClient(app)

# TEST DATA
SAMPLE_PDF_CONTENT = b"%PDF-1.4 sample content"
MOCK_FILE_NAME_1 = "test_doc_one.pdf"
MOCK_FILE_NAME_2 = "test_doc_two.pdf"

# --- Upload Tests ---

@patch("app.services.document_service.DocumentService.check_duplicate")
@patch("app.services.document_service.DocumentService.register_document")
@patch("app.services.pdf_processor.PDFProcessor.extract_text")
@patch("app.services.chunking.ChunkingService.chunk_text")
@patch("app.services.embedding.EmbeddingService.generate_embeddings")
@patch("app.services.vector_store.VectorStore.insert_chunks") 
@patch("app.services.document_service.DocumentService.mark_ingested")
def test_upload_batch_e2e(
    mock_mark_ingested, 
    mock_insert_chunks, 
    mock_generate_embeddings, 
    mock_chunk_text, 
    mock_extract_text, 
    mock_register_doc, 
    mock_check_duplicate
):
    # Mock return values
    mock_check_duplicate.return_value = None
    mock_register_doc.side_effect = ["doc-id-1", "doc-id-2"]
    mock_extract_text.return_value = "This is some sample text for testing the RAG chatbot."
    mock_chunk_text.return_value = ["This is chunk 1", "This is chunk 2"]
    mock_insert_chunks.return_value = 2
    
    files = [
        ("files", (MOCK_FILE_NAME_1, SAMPLE_PDF_CONTENT, "application/pdf")),
        ("files", (MOCK_FILE_NAME_2, SAMPLE_PDF_CONTENT, "application/pdf")),
    ]
    
    response = client.post("/upload/batch", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["message"].startswith("Batch upload completed")
    assert data["total_files"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2
    
    res1 = data["results"][0]
    assert res1["file_name"] == MOCK_FILE_NAME_1
    assert res1["status"] == "success"
    assert res1["chunks_created"] == 2
    assert res1["technology"] != "general" # Should be classified (mocks to general or tech)

    # Verify calls
    assert mock_register_doc.call_count == 2
    assert mock_insert_chunks.call_count == 2


@patch("app.services.document_service.DocumentService.check_duplicate")
def test_upload_duplicate_file(mock_check_duplicate):
    mock_check_duplicate.return_value = {
        "document_id": "existing-id",
        "status": "INGESTED",
        "chunk_count": 5,
        "technology": "python",
        "domain": "backend"
    }

    files = [("files", (MOCK_FILE_NAME_1, SAMPLE_PDF_CONTENT, "application/pdf"))]
    response = client.post("/upload/batch", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 0 # Since it's a duplicate, successful upload count is 0? Or maybe duplicate counts as success? 
    # Based on implementation: duplicate is separate status. successful += 1 ONLY on actual processing. 
    # Let's check implementation behavior: 
    # Check duplicate -> results.append(status="duplicate") -> continue. 
    # successful not incremented. So successful=0.
    
    assert data["total_files"] == 1
    assert data["failed"] == 0
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "duplicate"


def test_upload_invalid_file_type():
    files = [("files", ("bad_file.txt", b"text content", "text/plain"))]
    response = client.post("/upload/batch", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# --- Chat Tests ---

@patch("app.services.query_classifier.QueryClassifier.classify")
@patch("app.services.embedding.EmbeddingService.generate_embedding")
@patch("app.services.vector_store.VectorStore.search")
@patch("app.services.reranker.RerankerService.rerank")
@patch("app.services.llm.LLMService.generate_response")
def test_chat_flow(
    mock_llm_response,
    mock_rerank,
    mock_search,
    mock_embedding,
    mock_classify
):
    # Mock logic
    mock_classify.return_value = (["python"], "backend")
    mock_embedding.return_value = [0.1] * 1536
    mock_search.return_value = [{"text": "context", "metadata": {"source": "doc1"}}]
    mock_rerank.return_value = [{"text": "context", "metadata": {"source": "doc1"}}]
    mock_llm_response.return_value = "Python is a programming language."

    payload = {
        "query": "What is Python?",
        "file_ids": ["00000000-0000-4000-8000-000000000001"],
    }
    response = client.post("/chat/", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["answer"] == "Python is a programming language."
    assert "session_id" in data
    assert data["detected_technology"] == "python"

