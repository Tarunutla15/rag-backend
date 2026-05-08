"""
Raw block store for tables, code, and images.
Supports Supabase (REST API) and SQLite fallback.
"""
from typing import Dict, Any, Optional, List
import json
import uuid
import logging
import time

from app.services.database import get_database

logger = logging.getLogger(__name__)


class RawBlockStore:
    def __init__(self):
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"

    def store_table(self, rows: List[List[str]], meta: Dict[str, Any]) -> str:
        table_id = str(uuid.uuid4())
        document_id = meta.get("document_id")
        page_number = meta.get("page_number", -1)
        created_at = int(time.time())
        headers = rows[0] if rows else []
        body_rows = rows[1:] if len(rows) > 1 else []

        if self._use_supabase:
            self.db.supabase.table("raw_tables").insert({
                "table_id": table_id,
                "document_id": document_id,
                "page_number": page_number,
                "headers_json": json.dumps(headers, ensure_ascii=False),
                "rows_json": json.dumps(body_rows, ensure_ascii=False),
                "created_at": created_at,
            }).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO raw_tables (table_id, document_id, page_number, headers_json, rows_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (table_id, document_id, page_number, json.dumps(headers, ensure_ascii=False),
                     json.dumps(body_rows, ensure_ascii=False), created_at),
                )
        return table_id

    def store_code(self, code: str, meta: Dict[str, Any]) -> str:
        code_id = str(uuid.uuid4())
        document_id = meta.get("document_id")
        page_number = meta.get("page_number", -1)
        created_at = int(time.time())
        language = meta.get("language", "unknown")

        if self._use_supabase:
            self.db.supabase.table("raw_code_blocks").insert({
                "code_id": code_id,
                "document_id": document_id,
                "language": language,
                "code_text": code,
                "page_number": page_number,
                "created_at": created_at,
            }).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO raw_code_blocks (code_id, document_id, language, code_text, page_number, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (code_id, document_id, language, code, page_number, created_at),
                )
        return code_id

    def store_image(self, image_meta: Dict[str, Any], meta: Dict[str, Any]) -> str:
        image_id = str(uuid.uuid4())
        document_id = meta.get("document_id")
        page_number = meta.get("page_number", -1)
        created_at = int(time.time())
        caption = meta.get("section_title") or f"Image on page {page_number}"
        image_path = image_meta.get("name") if isinstance(image_meta, dict) else ""

        if self._use_supabase:
            self.db.supabase.table("raw_images").insert({
                "image_id": image_id,
                "document_id": document_id,
                "page_number": page_number,
                "caption": caption,
                "image_path": image_path,
                "created_at": created_at,
            }).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO raw_images (image_id, document_id, page_number, caption, image_path, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (image_id, document_id, page_number, caption, image_path, created_at),
                )
        return image_id

    def get_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        if self._use_supabase:
            response = self.db.supabase.table("raw_tables").select("*").eq("table_id", table_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "table_id": row["table_id"],
                    "document_id": row["document_id"],
                    "page_number": row["page_number"],
                    "headers": json.loads(row["headers_json"]) if row["headers_json"] else [],
                    "rows": json.loads(row["rows_json"]) if row["rows_json"] else [],
                    "created_at": row["created_at"],
                }
            return None
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM raw_tables WHERE table_id = %s", (table_id,))
                row = cur.fetchone()
            if not row:
                return None
            return {
                "table_id": row["table_id"],
                "document_id": row["document_id"],
                "page_number": row["page_number"],
                "headers": json.loads(row["headers_json"]) if row["headers_json"] else [],
                "rows": json.loads(row["rows_json"]) if row["rows_json"] else [],
                "created_at": row["created_at"],
            }

    def get_code(self, code_id: str) -> Optional[Dict[str, Any]]:
        if self._use_supabase:
            response = self.db.supabase.table("raw_code_blocks").select("*").eq("code_id", code_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "code_id": row["code_id"],
                    "document_id": row["document_id"],
                    "language": row["language"],
                    "code_text": row["code_text"],
                    "page_number": row["page_number"],
                    "created_at": row["created_at"],
                }
            return None
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM raw_code_blocks WHERE code_id = %s", (code_id,))
                row = cur.fetchone()
            if not row:
                return None
            return {
                "code_id": row["code_id"],
                "document_id": row["document_id"],
                "language": row["language"],
                "code_text": row["code_text"],
                "page_number": row["page_number"],
                "created_at": row["created_at"],
            }

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        if self._use_supabase:
            response = self.db.supabase.table("raw_images").select("*").eq("image_id", image_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "image_id": row["image_id"],
                    "document_id": row["document_id"],
                    "page_number": row["page_number"],
                    "caption": row["caption"],
                    "image_path": row["image_path"],
                    "created_at": row["created_at"],
                }
            return None
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM raw_images WHERE image_id = %s", (image_id,))
                row = cur.fetchone()
            if not row:
                return None
            return {
                "image_id": row["image_id"],
                "document_id": row["document_id"],
                "page_number": row["page_number"],
                "caption": row["caption"],
                "image_path": row["image_path"],
                "created_at": row["created_at"],
            }

    def delete_document(self, document_id: str) -> None:
        """Delete all raw blocks (tables/code/images) for a document_id."""
        if not document_id:
            return
        if self._use_supabase:
            self.db.supabase.table("raw_tables").delete().eq("document_id", document_id).execute()
            self.db.supabase.table("raw_code_blocks").delete().eq("document_id", document_id).execute()
            self.db.supabase.table("raw_images").delete().eq("document_id", document_id).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM raw_tables WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM raw_code_blocks WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM raw_images WHERE document_id = %s", (document_id,))


_raw_store: Optional[RawBlockStore] = None


def get_raw_block_store() -> RawBlockStore:
    global _raw_store
    if _raw_store is None:
        _raw_store = RawBlockStore()
    return _raw_store
