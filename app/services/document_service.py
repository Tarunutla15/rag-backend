"""Document registry and fingerprinting service with technology/domain support."""
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DocumentStatus(str, Enum):
    """Document ingestion status."""
    UPLOADED = "UPLOADED"
    INGESTED = "INGESTED"
    FAILED = "FAILED"


class DocumentService:
    """
    Service for managing document registry and preventing duplicates.
    Supports technology and domain metadata for intelligent query routing.
    """
    
    def __init__(self, registry_file: str = "document_registry.json"):
        """
        Initialize document service.
        
        Args:
            registry_file: Path to JSON file storing document registry
        """
        self.registry_file = Path(registry_file)
        self.registry: Dict[str, Dict] = self._load_registry()
        logger.info(f"Document registry initialized with {len(self.registry)} documents")
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load document registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}, starting with empty registry")
                return {}
        return {}
    
    def _save_registry(self):
        """Save document registry to file."""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def compute_file_hash_from_bytes(self, file_bytes: bytes) -> str:
        """
        Compute SHA256 hash from file bytes (for uploaded files).
        
        Args:
            file_bytes: File content as bytes
            
        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(file_bytes).hexdigest()
    
    def check_duplicate(self, file_hash: str) -> Optional[Dict]:
        """
        Check if a document with this hash already exists.
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            Document record if duplicate exists, None otherwise
        """
        for doc_id, doc in self.registry.items():
            if doc.get("file_hash") == file_hash:
                return doc
        return None
    
    def register_document(
        self,
        file_name: str,
        file_hash: str,
        technology: str = "general",
        domain: str = "general",
        status: DocumentStatus = DocumentStatus.UPLOADED
    ) -> str:
        """
        Register a new document in the registry.
        
        Args:
            file_name: Original filename
            file_hash: SHA256 hash of the file
            technology: Technology category (e.g., 'react', 'python')
            domain: Domain category (e.g., 'frontend', 'backend')
            status: Initial status
            
        Returns:
            document_id (UUID)
        """
        document_id = str(uuid.uuid4())
        
        document = {
            "document_id": document_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "technology": technology,  # NEW
            "domain": domain,          # NEW
            "status": status.value,
            "chunk_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.registry[document_id] = document
        self._save_registry()
        
        logger.info(f"Registered document: {document_id} ({file_name}) [tech={technology}, domain={domain}]")
        return document_id
    
    def update_metadata(
        self, 
        document_id: str, 
        technology: Optional[str] = None,
        domain: Optional[str] = None
    ):
        """
        Update document metadata (technology/domain).
        
        Args:
            document_id: Document ID
            technology: Technology category to set
            domain: Domain category to set
        """
        if document_id not in self.registry:
            raise ValueError(f"Document {document_id} not found in registry")
        
        if technology:
            self.registry[document_id]["technology"] = technology
        if domain:
            self.registry[document_id]["domain"] = domain
        
        self.registry[document_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save_registry()
        
        logger.info(f"Updated metadata for {document_id}: tech={technology}, domain={domain}")
    
    def mark_ingested(
        self, 
        document_id: str, 
        chunk_count: int,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        pdf_path: Optional[str] = None
    ):
        """
        Mark document as ingested with chunk count and optional metadata.
        
        Args:
            document_id: Document ID
            chunk_count: Number of chunks created
            technology: Optional technology category
            domain: Optional domain category
            pdf_path: Optional path to stored PDF (object storage)
        """
        if document_id not in self.registry:
            raise ValueError(f"Document {document_id} not found in registry")
        
        self.registry[document_id]["status"] = DocumentStatus.INGESTED.value
        self.registry[document_id]["chunk_count"] = chunk_count
        self.registry[document_id]["updated_at"] = datetime.utcnow().isoformat()
        
        if technology:
            self.registry[document_id]["technology"] = technology
        if domain:
            self.registry[document_id]["domain"] = domain
        if pdf_path is not None:
            self.registry[document_id]["pdf_path"] = pdf_path
        
        self._save_registry()
        
        tech = self.registry[document_id].get("technology", "general")
        dom = self.registry[document_id].get("domain", "general")
        logger.info(f"Marked document {document_id} as ingested with {chunk_count} chunks [tech={tech}, domain={dom}]")
    
    def mark_failed(self, document_id: str, error: str = None):
        """
        Mark document ingestion as failed.
        
        Args:
            document_id: Document ID
            error: Optional error message
        """
        if document_id not in self.registry:
            raise ValueError(f"Document {document_id} not found in registry")
        
        self.registry[document_id]["status"] = DocumentStatus.FAILED.value
        self.registry[document_id]["updated_at"] = datetime.utcnow().isoformat()
        if error:
            self.registry[document_id]["error"] = error
        self._save_registry()
        
        logger.warning(f"Marked document {document_id} as failed: {error}")
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """
        Get document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document record or None
        """
        return self.registry.get(document_id)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document from registry.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        if document_id in self.registry:
            del self.registry[document_id]
            self._save_registry()
            logger.info(f"Deleted document {document_id} from registry")
            return True
        return False
    
    def list_documents(
        self, 
        status: Optional[DocumentStatus] = None,
        technology: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[Dict]:
        """
        List all documents, optionally filtered by status, technology, or domain.
        
        Args:
            status: Optional status filter
            technology: Optional technology filter
            domain: Optional domain filter
            
        Returns:
            List of document records
        """
        documents = list(self.registry.values())
        
        if status:
            documents = [doc for doc in documents if doc.get("status") == status.value]
        if technology:
            documents = [doc for doc in documents if doc.get("technology") == technology]
        if domain:
            documents = [doc for doc in documents if doc.get("domain") == domain]
        
        return documents
    
    def get_technologies(self) -> List[str]:
        """
        Get list of all technologies in the registry.
        
        Returns:
            List of unique technology values
        """
        technologies = set()
        for doc in self.registry.values():
            tech = doc.get("technology")
            if tech:
                technologies.add(tech)
        return sorted(list(technologies))
    
    def get_domains(self) -> List[str]:
        """
        Get list of all domains in the registry.
        
        Returns:
            List of unique domain values
        """
        domains = set()
        for doc in self.registry.values():
            domain = doc.get("domain")
            if domain:
                domains.add(domain)
        return sorted(list(domains))
    
    def get_documents_by_technology(self, technology: str) -> List[Dict]:
        """
        Get all documents for a specific technology.
        
        Args:
            technology: Technology to filter by
            
        Returns:
            List of matching documents
        """
        return self.list_documents(technology=technology)
    
    def get_documents_by_domain(self, domain: str) -> List[Dict]:
        """
        Get all documents for a specific domain.
        
        Args:
            domain: Domain to filter by
            
        Returns:
            List of matching documents
        """
        return self.list_documents(domain=domain)
