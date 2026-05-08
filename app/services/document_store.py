"""DB-backed document registry (Supabase or SQLite)."""
from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional
import logging

from app.services.database import get_database

logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self):
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"

    @staticmethod
    def compute_hash_from_bytes(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def list_documents(self) -> List[Dict[str, Any]]:
        if self._use_supabase:
            resp = self.db.supabase.table("documents").select("*").order("updated_at", desc=True).limit(10000).execute()
            return list(resp.data or [])
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        if not document_id:
            return None
        if self._use_supabase:
            resp = self.db.supabase.table("documents").select("*").eq("document_id", document_id).limit(1).execute()
            return resp.data[0] if resp.data else None
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM documents WHERE document_id = %s", (document_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        if not file_hash:
            return None
        if self._use_supabase:
            resp = self.db.supabase.table("documents").select("*").eq("file_hash", file_hash).limit(1).execute()
            return resp.data[0] if resp.data else None
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM documents WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
            return dict(row) if row else None

    def create_document(
        self,
        *,
        file_name: str,
        file_hash: str,
        status: str = "UPLOADED",
        technology: str = "general",
        domain: str = "general",
        document_id: Optional[str] = None,
    ) -> str:
        doc_id = document_id or str(uuid.uuid4())
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row = {
            "document_id": doc_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "status": status,
            "chunk_count": 0,
            "technology": technology,
            "domain": domain,
            "pdf_path": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        if self._use_supabase:
            try:
                self.db.supabase.table("documents").insert(row).execute()
                return doc_id
            except Exception as e:
                err = str(e).lower()
                if any(
                    x in err
                    for x in ("duplicate", "unique", "violates", "23505", "already exists")
                ):
                    existing = self.get_by_hash(file_hash)
                    if existing:
                        logger.warning(
                            "create_document: duplicate file_hash; using existing document_id=%s",
                            existing.get("document_id"),
                        )
                        return str(existing["document_id"])
                raise
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO documents
                       (document_id, file_name, file_hash, status, chunk_count, technology, domain, pdf_path, error, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                    (doc_id, file_name, file_hash, status, 0, technology, domain, None, None),
                )
        except sqlite3.IntegrityError:
            existing = self.get_by_hash(file_hash)
            if existing:
                logger.warning(
                    "create_document: SQLite unique constraint (file_hash); using existing document_id=%s",
                    existing.get("document_id"),
                )
                return str(existing["document_id"])
            raise
        return doc_id

    def update_document(self, document_id: str, updates: Dict[str, Any]) -> None:
        if not document_id:
            return
        updates = dict(updates or {})
        # always bump updated_at
        updates["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self._use_supabase:
            self.db.supabase.table("documents").update(updates).eq("document_id", document_id).execute()
            return
        # SQLite path: update only known columns
        allowed = {
            "file_name", "file_hash", "status", "chunk_count", "technology", "domain", "pdf_path", "error", "updated_at"
        }
        cols = [(k, v) for k, v in updates.items() if k in allowed]
        if not cols:
            return
        set_sql = ", ".join([f"{k} = %s" for k, _ in cols])
        params = [v for _, v in cols] + [document_id]
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE documents SET {set_sql} WHERE document_id = %s", params)

    def mark_ingested(
        self,
        *,
        document_id: str,
        chunk_count: int,
        technology: str,
        domain: str,
        pdf_path: Optional[str],
    ) -> None:
        self.update_document(
            document_id,
            {
                "status": "INGESTED",
                "chunk_count": int(chunk_count or 0),
                "technology": technology or "general",
                "domain": domain or "general",
                "pdf_path": pdf_path,
                "error": None,
            },
        )

    def mark_failed(self, *, document_id: str, error: str) -> None:
        self.update_document(document_id, {"status": "FAILED", "error": error or "Unknown error"})

    def get_technologies(self) -> List[str]:
        docs = self.list_documents()
        techs = sorted({(d.get("technology") or "general") for d in docs if (d.get("technology") or "").strip()})
        return techs

    def list_ingested_document_ids(self) -> List[str]:
        """Document IDs that finished ingest and are eligible for RAG retrieval."""
        out: List[str] = []
        for d in self.list_documents():
            did = d.get("document_id")
            if not did:
                continue
            if (d.get("status") or "").upper() == "INGESTED":
                out.append(str(did))
        return out

    def delete_document_row(self, document_id: str) -> bool:
        """Remove document metadata row."""
        if not document_id:
            return False
        if self._use_supabase:
            self.db.supabase.table("documents").delete().eq("document_id", document_id).execute()
            return True
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
            return cur.rowcount > 0


_store: Optional[DocumentStore] = None


def get_document_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store

