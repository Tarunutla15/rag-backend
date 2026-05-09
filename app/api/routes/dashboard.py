"""Usage / observability dashboard API."""
from fastapi import APIRouter, Query

from app.models.schemas import DashboardUsageResponse, UsageEventItem, UsageSummary
from app.services.usage_store import get_usage_dashboard

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/usage", response_model=DashboardUsageResponse)
async def get_usage(
    days: int = Query(30, ge=1, le=365, description="Rolling window for aggregates"),
    limit: int = Query(100, ge=1, le=500, description="Max recent rows returned"),
):
    """
    Token usage for chat completions (main LLM answer per query).
    Requires `usage_events` table (created automatically on SQLite; Supabase via schema sync).
    """
    raw = get_usage_dashboard(days=days, limit=limit)
    s = raw.get("summary") or {}
    summary = UsageSummary(
        chat_completions=int(s.get("chat_completions") or 0),
        prompt_tokens=int(s.get("prompt_tokens") or 0),
        completion_tokens=int(s.get("completion_tokens") or 0),
        total_tokens=int(s.get("total_tokens") or 0),
        estimated_cost_usd=s.get("estimated_cost_usd"),
    )
    recent = [UsageEventItem(**item) for item in raw.get("recent") or []]
    return DashboardUsageResponse(
        days=raw.get("days", days),
        summary=summary,
        recent=recent,
    )
