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
    # 只读取明确的编译属性/常量；图片组查询也可能是字符串数组。
    query_pattern = r'("(?:[^"\\]|\\.)*"|__dilConstants\["\d+"\]|\[(?:\s*"(?:[^"\\]|\\.)*"\s*,?)*\])'
    pattern = r'__dil\.jsx\((AsyncImageGroup|AsyncImage),\{[^\n]*?"query":' + query_pattern + r'[^\n]*?"__resolutionId":__dilConstants\["(\d+)"\]'
    for match in re.finditer(pattern, str(dil.get("code") or "")):
        component, query_value = match[1], match[2]
        try:
            query = constants.get(re.search(r'"(\d+)"', query_value)[1]) if query_value.startswith("__dilConstants") else json.loads(query_value)
        except (ValueError, TypeError):
            continue
        group = component == "AsyncImageGroup"
        valid_query = isinstance(query, str) and bool(query.strip())
        if group and isinstance(query, list):
            valid_query = bool(query) and len(query) <= 100 and all(isinstance(item, str) and item.strip() for item in query)
        resolution_id = constants.get(match[3])
        result = results.get(resolution_id, {}) if isinstance(resolution_id, str) else {}
        if not valid_query or not isinstance(result, dict):
            continue
        state = result.get("state")
        images = state.get("images") if isinstance(state, dict) else None
        if not isinstance(images, list):
            continue
        resolved = []
        seen = set()
        for image in images:
            if not isinstance(image, dict):
                continue
            src = _url(image.get("content_url")) or _url(image.get("thumbnail_url"))
            if src and src not in seen:
                seen.add(src)
                resolved.append({"src": src, "title": str(image.get("title") or "参考图片")})
                if not group or len(resolved) >= 100:
                    break
        if resolved:
            key = json.dumps(query, ensure_ascii=False, separators=(",", ":")) if isinstance(query, list) else query
            resource = dict(resolved[0])
            if group:
                resource["images"] = resolved
            resources["images"][key] = resource
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
