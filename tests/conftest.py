import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# --- Add App to Path ---
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# --- Mock Environment BEFORE App Imports ---
os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
os.environ["ZILLIZ_URI"] = "https://mock.zilliz.uri"
os.environ["ZILLIZ_TOKEN"] = "mock-token"
os.environ["ZILLIZ_COLLECTION_NAME"] = "test_collection"

# --- Import App Modules ---
from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import VectorStore
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService

@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_vector_store():
    with patch("app.api.routes.upload.get_vector_store") as mock:
        store = MagicMock(spec=VectorStore)
        store.insert_chunks.return_value = 5  # Assume 5 chunks created
        store.search.return_value = []
        mock.return_value = store
        yield store

@pytest.fixture
def mock_embedding_service():
    with patch("app.services.embedding.EmbeddingService") as mock:
        instance = mock.return_value
        # Mock embedding response (random floats)
        instance.generate_embeddings.return_value = [[0.1] * 1536] * 5
        instance.generate_embedding.return_value = [0.1] * 1536
        yield instance

@pytest.fixture
def mock_llm_service():
    with patch("app.services.llm.LLMService") as mock:
        instance = asLc_mock.return_value
        instance.generate_response.return_value = "This is a mocked LLM response based on context."
        yield instance

@pytest.fixture
def mock_document_service():
    with patch("app.services.document_service.DocumentService") as mock:
        instance = mock.return_value
        instance.check_duplicate.return_value = None
        instance.register_document.return_value = "test-doc-id"
        yield instance
