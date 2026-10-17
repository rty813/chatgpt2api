from __future__ import annotations

import copy
import time
from typing import Any

from services.account_service import account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.helper import CODEX_IMAGE_MODEL
from utils.log import logger

# 上游模型列表的缓存时长。/v1/models 会被下游客户端反复轮询，没必要每次都打官方接口。
MODEL_LIST_TTL_SECS = 300.0
_model_list_cache: dict[str, tuple[float, dict[str, Any]]] = {}

DISABLED_ACCOUNT_STATUSES = {"禁用", "异常"}


def reset_model_list_cache() -> None:
    _model_list_cache.clear()


def _model_list_token(accounts: list[dict[str, Any]]) -> str:
    """挑一个用来列模型的账号：优先 Pro（只有 Pro 号的模型列表里才有 `gpt-6-pro` 这类 `*-pro` slug），
    其次任意可用账号；都没有就返回空串，退回匿名列表。

    这里刻意不用 `get_text_access_token`：那个方法会推进账号轮转游标、并偏向 Plus，
    仅仅为了列模型而打乱对话链路的调度不合算。
    """
    candidates = [
        account
        for account in accounts
        if isinstance(account, dict)
           and account.get("status") not in DISABLED_ACCOUNT_STATUSES
           and str(account.get("access_token") or "")
           # codex 账号拿的是 codex token，问不了网页版的模型列表。
           and account_service._normalize_source_type(account.get("source_type")) == "web"
    ]
    if not candidates:
        return ""
    preferred = next(
        (
            account
            for account in candidates
            if account_service._normalize_account_type(account.get("type")) == "Pro"
        ),
        candidates[0],
    )
    access_token = str(preferred.get("access_token") or "")
    # token 未过期时是空操作，不会额外产生刷新请求。
    return account_service.refresh_access_token(access_token, event="list_models") or access_token


def _upstream_models(access_token: str) -> dict[str, Any]:
    cached = _model_list_cache.get(access_token)
    now = time.monotonic()
    if cached and now - cached[0] < MODEL_LIST_TTL_SECS:
        return copy.deepcopy(cached[1])
    try:
        result = OpenAIBackendAPI(access_token=access_token).list_models()
    except Exception as exc:
        if not access_token:
            raise
        # 账号侧列表拿不到（token 失效、限流等）时退回匿名列表：宁可少列几个模型，也别让 /v1/models 整个挂掉。
        logger.warning({"event": "model_list_account_failed", "error": repr(exc)[:200]})
        return _upstream_models("")
    if isinstance(result.get("data"), list):
        _model_list_cache[access_token] = (now, copy.deepcopy(result))
    return result


def list_models() -> dict[str, Any]:
    accounts = account_service.list_accounts()
    result = _upstream_models(_model_list_token(accounts))
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    web_image_accounts = [
        account
        for account in accounts
        if isinstance(account, dict)
    ]
    codex_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if web_image_accounts:
        dynamic_models.add("gpt-image-2")
    if codex_types & {"Plus", "Team", "Pro"}:
        dynamic_models.add(CODEX_IMAGE_MODEL)
    if "Plus" in codex_types:
        dynamic_models.add(f"plus-{CODEX_IMAGE_MODEL}")
    if "Team" in codex_types:
        dynamic_models.add(f"team-{CODEX_IMAGE_MODEL}")
    if "Pro" in codex_types:
        dynamic_models.add(f"pro-{CODEX_IMAGE_MODEL}")

    for model in sorted(dynamic_models):
        if model not in seen:
            data.append({
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt2api",
                "permission": [],
                "root": model,
                "parent": None,
            })
    return result
