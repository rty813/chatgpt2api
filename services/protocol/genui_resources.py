"""读取 GenUI 资源数据；编译后的代码只用于匹配常量引用，从不执行。"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit


def _url(value: Any) -> str:
    if not isinstance(value, str) or re.search(r"[\s<>\x00-\x1f]", value):
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme in ("https", "http") and parsed.netloc and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return ""


def collect_genui_resources(message: dict[str, Any], resources: dict[str, dict]) -> None:
    metadata = message.get("metadata")
    dil = metadata.get("model_dil_v2") if isinstance(metadata, dict) else None
    if not isinstance(dil, dict):
        return
    constants = dil.get("constants") or {}
    app = dil.get("appData") or {}
    results = (app.get("opGenui") or {}).get("componentResults") or {}
    if not isinstance(constants, dict) or not isinstance(results, dict):
        return
    # 上游 AsyncImage 的 query 与 __resolutionId 都来自明确的编译属性/常量。
    pattern = r'__dil\.jsx\(AsyncImage,\{[^\n]*?"query":("(?:[^"\\]|\\.)*"|__dilConstants\["\d+"\])[^\n]*?"__resolutionId":__dilConstants\["(\d+)"\]'
    for match in re.finditer(pattern, str(dil.get("code") or "")):
        query_value = match[1]
        try:
            query = constants.get(re.search(r'"(\d+)"', query_value)[1]) if query_value.startswith("__dilConstants") else json.loads(query_value)
        except (ValueError, TypeError):
            continue
        result = results.get(constants.get(match[2]), {})
        if not isinstance(query, str) or not isinstance(result, dict):
            continue
        images = (result.get("state") or {}).get("images") or []
        for image in images:
            if not isinstance(image, dict):
                continue
            src = _url(image.get("content_url")) or _url(image.get("thumbnail_url"))
            if src:
                resources["images"][query] = {"src": src, "title": str(image.get("title") or "参考图片")}
                break
    for result in results.values():
        if not isinstance(result, dict) or result.get("componentName") != "Cite":
            continue
        for item in (result.get("state") or {}).get("items") or []:
            if not isinstance(item, dict):
                continue
            ref, url = item.get("ref_id"), _url(item.get("url"))
            if isinstance(ref, str) and url:
                resources["citations"][ref] = {"url": url, "title": str(item.get("title") or item.get("source_label") or "来源")}


def genui_resource_block(resources: dict[str, dict]) -> str:
    if not any(resources.values()):
        return ""
    # 以正文尾部的声明携带资源，兼容只支持追加文本的 OpenAI / Dify 流。
    data = json.dumps(resources, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"\n\n<genui-resources data={{{data}}}/>"
