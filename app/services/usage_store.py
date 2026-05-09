"""Persist LLM usage (tokens, optional cost) for dashboard / observability."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.services.database import get_database

logger = logging.getLogger(__name__)


def record_chat_completion(
    *,
    session_id: str,
    message_id: Optional[Any],
    query_preview: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    model: Optional[str],
    provider: str,
    cost_usd: Optional[float],
) -> None:
    """Best-effort insert; never raises to callers."""
    preview = (query_preview or "").strip()[:500]
    db = get_database()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mid = int(message_id) if message_id is not None else None

    try:
        if db.engine == "supabase":
            db.supabase.table("usage_events").insert({
                "session_id": session_id,
                "message_id": mid,
                "event_type": "chat_completion",
                "query_preview": preview or None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": model,
                "provider": provider,
                "cost_usd": cost_usd,
                "created_at": now_iso,
            }).execute()
        else:
            with db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO usage_events (
                        session_id, message_id, event_type, query_preview,
                        prompt_tokens, completion_tokens, total_tokens,
                        model, provider, cost_usd, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)""",
                    (
                        session_id,
                        mid,
                        "chat_completion",
                        preview or None,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        model,
                        provider,
                        cost_usd,
                    ),
                )
    except Exception as e:
        logger.warning("usage_events insert failed (non-fatal): %s", e)


def get_usage_dashboard(*, days: int = 30, limit: int = 200) -> Dict[str, Any]:
    """
    Aggregate totals and recent chat_completion rows for the dashboard API.
    """
    days = max(1, min(int(days), 365))
    limit = max(1, min(int(limit), 2000))
    db = get_database()

    if db.engine == "supabase":
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows_all: List[Dict[str, Any]] = []
        try:
            resp = (
                db.supabase.table("usage_events")
                .select(
                    "id,session_id,message_id,query_preview,prompt_tokens,completion_tokens,total_tokens,model,provider,cost_usd,created_at"
                )
                .eq("event_type", "chat_completion")
                .gte("created_at", cutoff)
                .order("created_at", desc=True)
                .limit(5000)
                .execute()
            )
            rows_all = resp.data or []
        except Exception as e:
            logger.warning("usage_events supabase select failed: %s", e)

        total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in rows_all)
        total_completion = sum(int(r.get("completion_tokens") or 0) for r in rows_all)
        total_all = sum(int(r.get("total_tokens") or 0) for r in rows_all)
        cost_vals = [float(r["cost_usd"]) for r in rows_all if r.get("cost_usd") is not None]
        total_cost = sum(cost_vals) if cost_vals else None

        recent_rows = rows_all[:limit]
        recent = []
        for r in recent_rows:
            recent.append({
                "id": r.get("id"),
                "session_id": r.get("session_id"),
                "query_preview": r.get("query_preview") or "",
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "total_tokens": r.get("total_tokens"),
                "model": r.get("model"),
                "provider": r.get("provider"),
                "cost_usd": float(r["cost_usd"]) if r.get("cost_usd") is not None else None,
                "created_at": str(r.get("created_at", "")),
            })

        return {
            "days": days,
            "summary": {
                "chat_completions": len(rows_all),
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_all,
                "estimated_cost_usd": total_cost,
            },
            "recent": recent,
        }

    # SQLite
    summary_counts: Dict[str, Any] = {
        "chat_completions": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
    }
    recent: List[Dict[str, Any]] = []
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            # days is sanitized int — safe in SQL fragment
            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS n,
                    COALESCE(SUM(prompt_tokens), 0) AS sp,
                    COALESCE(SUM(completion_tokens), 0) AS sc,
                    COALESCE(SUM(total_tokens), 0) AS st,
                    COALESCE(SUM(cost_usd), 0) AS scost
                FROM usage_events
                WHERE event_type = 'chat_completion'
                  AND datetime(created_at) >= datetime('now', '-{days} days')
                """
            )
            agg = cur.fetchone()
            if agg:
                summary_counts["chat_completions"] = int(agg["n"] or 0)
                summary_counts["prompt_tokens"] = int(agg["sp"] or 0)
                summary_counts["completion_tokens"] = int(agg["sc"] or 0)
                summary_counts["total_tokens"] = int(agg["st"] or 0)
                scost = agg["scost"]
                summary_counts["estimated_cost_usd"] = float(scost) if scost is not None else None

            cur.execute(
                f"""
                SELECT id, session_id, message_id, query_preview, prompt_tokens, completion_tokens,
                       total_tokens, model, provider, cost_usd, created_at
                FROM usage_events
                WHERE event_type = 'chat_completion'
                  AND datetime(created_at) >= datetime('now', '-{days} days')
                ORDER BY datetime(created_at) DESC
                LIMIT %s
                """,
                (limit,),
            )
            for row in cur.fetchall():
                r = dict(row)
                recent.append({
                    "id": r.get("id"),
                    "session_id": r.get("session_id"),
                    "query_preview": r.get("query_preview") or "",
                    "prompt_tokens": r.get("prompt_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                    "total_tokens": r.get("total_tokens"),
                    "model": r.get("model"),
                    "provider": r.get("provider"),
                    "cost_usd": float(r["cost_usd"]) if r.get("cost_usd") is not None else None,
                    "created_at": str(r.get("created_at", "")),
                })
    except Exception as e:
        logger.warning("usage_events sqlite dashboard failed: %s", e)

    return {
        "days": days,
        "summary": {
            "chat_completions": summary_counts["chat_completions"],
            "prompt_tokens": summary_counts["prompt_tokens"],
            "completion_tokens": summary_counts["completion_tokens"],
            "total_tokens": summary_counts["total_tokens"],
            "estimated_cost_usd": summary_counts["estimated_cost_usd"],
        },
        "recent": recent,
    }
