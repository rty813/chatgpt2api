from __future__ import annotations

import time
import uuid
from typing import Any, Iterable, Iterator

from fastapi import HTTPException

from services.protocol.chat_completion_cache import cache_key, chat_completion_cache, normalize_text_messages
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    count_message_image_tokens,
    count_message_text_tokens,
    count_text_tokens,
    encode_images,
    normalize_messages,
    resolve_thinking_effort,
    stream_image_outputs_with_pool,
    stream_text_deltas,
    stream_text_events,
    text_backend,
)
from services.protocol import deep_research_protocol
from services.protocol.deep_research_protocol import is_deep_research_request
from services.protocol.web_search_tool import (
    WEB_SEARCH_TOOL_TYPES,
    has_unsupported_tools,
    has_web_search_tool,
    normalized_sources,
    run_web_search,
    search_query_from_messages,
    text_with_url_citations,
)
from utils.helper import extract_image_from_message_content, extract_response_prompt, has_response_image_generation_tool
from utils.image_tokens import (
    count_image_content_tokens,
    count_image_output_items_tokens,
    image_usage,
    token_usage,
)

TOOL_UNAVAILABLE_SYSTEM_MESSAGE = (
    "This compatibility backend cannot execute local tools, shell commands, non-search tools, "
    "or file operations. Do not claim to have run tools or inspected external resources. "
    "If a user asks you to use a tool, say that tool execution is unavailable through this backend."
)

RESPONSE_CONTENT_PART_TYPES = {"text", "input_text", "output_text", "image_url", "input_image", "image"}


def reasoning_text(reasoning: str) -> str:
    reasoning = (reasoning or "").strip()
    return reasoning


def is_text_response_request(body: dict[str, Any]) -> bool:
    return not has_response_image_generation_tool(body)


def has_unsupported_response_tools(body: dict[str, Any]) -> bool:
    return has_unsupported_tools(body, {"image_generation", *WEB_SEARCH_TOOL_TYPES})


def response_image_tool(body: dict[str, Any]) -> dict[str, object]:
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "image_generation":
            return tool
    return {}


def extract_response_image(input_value: object) -> tuple[bytes, str] | None:
    if isinstance(input_value, dict):
        if str(input_value.get("type") or "").strip() == "input_image":
            images = extract_image_from_message_content([input_value])
            return images[0] if images else None
        images = extract_image_from_message_content(input_value.get("content"))
        return images[0] if images else None
    if not isinstance(input_value, list):
        return None
    for item in reversed(input_value):
        if isinstance(item, dict):
            if str(item.get("type") or "").strip() == "input_image":
                images = extract_image_from_message_content([item])
                if images:
                    return images[0]
            images = extract_image_from_message_content(item.get("content"))
            if images:
                return images[0]
    return None


def _input_image_parts(input_value: object) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if isinstance(input_value, dict):
        content = input_value.get("content")
        if isinstance(content, list):
            parts.extend(item for item in content if isinstance(item, dict))
        return parts
    if not isinstance(input_value, list):
        return parts
    if all(isinstance(item, dict) and item.get("type") for item in input_value):
        return [item for item in input_value if isinstance(item, dict)]
    for item in input_value:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                parts.extend(part for part in content if isinstance(part, dict))
    return parts


def _is_response_content_part(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    part_type = str(value.get("type") or "").strip()
    return part_type in RESPONSE_CONTENT_PART_TYPES or ("image_url" in value and part_type != "message")


def _message_content_from_response_item(item: dict[str, Any]) -> object:
    content = item.get("content")
    if isinstance(content, list):
        return [dict(part) if isinstance(part, dict) else part for part in content]
    if isinstance(content, str):
        return content
    return extract_response_prompt([item]) or content or ""


def _append_response_message(messages: list[dict[str, Any]], role: object, content: object) -> None:
    if isinstance(content, str):
        if content.strip():
            messages.append({"role": str(role or "user"), "content": content.strip()})
        return
    if isinstance(content, list) and content:
        messages.append({"role": str(role or "user"), "content": content})


def messages_from_input(input_value: object, instructions: object = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_text = str(instructions or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if isinstance(input_value, str):
        if input_value.strip():
            messages.append({"role": "user", "content": input_value.strip()})
        return messages
    if isinstance(input_value, dict):
        if _is_response_content_part(input_value):
            _append_response_message(messages, "user", [dict(input_value)])
            return messages
        _append_response_message(
            messages,
            input_value.get("role") or "user",
            _message_content_from_response_item(input_value),
        )
        return messages
    if isinstance(input_value, list):
        if all(_is_response_content_part(item) for item in input_value):
            _append_response_message(messages, "user", [dict(item) for item in input_value if isinstance(item, dict)])
            return messages
        pending_parts: list[dict[str, Any]] = []
        for item in input_value:
            if _is_response_content_part(item):
                pending_parts.append(dict(item))
                continue
            if pending_parts:
                _append_response_message(messages, "user", pending_parts)
                pending_parts = []
            if not isinstance(item, dict):
                continue
            _append_response_message(
                messages,
                item.get("role") or "user",
                _message_content_from_response_item(item),
            )
        if pending_parts:
            _append_response_message(messages, "user", pending_parts)
    return messages


def text_output_item(
    text: str,
    item_id: str | None = None,
    status: str = "completed",
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": annotations or []}],
    }


def reasoning_output_item(
    text: str,
    item_id: str | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "id": item_id or f"rs_{uuid.uuid4().hex}",
        "type": "reasoning",
        "status": status,
        "summary": [{"type": "summary_text", "text": text}],
    }


def web_search_call_item(
    query: str,
    item_id: str | None = None,
    status: str = "completed",
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "type": "search",
        "query": query,
        "queries": [query],
    }
    if sources:
        action["sources"] = [
            {"type": "url", "url": source["url"]}
            for source in sources
            if source.get("url")
        ]
    return {
        "id": item_id or f"ws_{uuid.uuid4().hex}",
        "type": "web_search_call",
        "status": status,
        "action": action,
    }


def image_output_items(prompt: str, data: list[dict[str, Any]], item_id: str | None = None) -> list[dict[str, Any]]:
    output = []
    for item in data:
        b64_json = str(item.get("b64_json") or "").strip()
        if b64_json:
            output.append({
                "id": item_id or f"ig_{len(output) + 1}",
                "type": "image_generation_call",
                "status": "completed",
                "result": b64_json,
                "revised_prompt": str(item.get("revised_prompt") or prompt).strip() or prompt,
            })
    return output


def response_created(response_id: str, model: str, created: int) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": [],
            "parallel_tool_calls": False,
        },
    }


def response_completed(
    response_id: str,
    model: str,
    created: int,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": False,
        },
    }
    if usage:
        response["response"]["usage"] = usage
    return response


def text_response_parts(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = normalize_text_messages(normalize_messages(messages_from_input(body.get("input"), body.get("instructions"))))
    if has_unsupported_response_tools(body):
        messages.insert(0, {"role": "system", "content": TOOL_UNAVAILABLE_SYSTEM_MESSAGE})
    return model, messages


def stream_text_response(backend, body: dict[str, Any], messages: list[dict[str, Any]] | None = None, base_url: str = "") -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    reasoning_id = f"rs_{uuid.uuid4().hex}"
    created = int(time.time())
    full_text = ""
    full_reasoning = ""
    reasoning_added = False
    yield response_created(response_id, model, created)
    request = ConversationRequest(model=model, messages=messages, thinking_effort=resolve_thinking_effort(body, model), base_url=base_url)
    text_added = False
    text_output_index = 0
    for item in stream_text_events(backend, request) if request.thinking_effort else ({"content": delta, "reasoning": ""} for delta in stream_text_deltas(backend, request)):
        reasoning = item.get("reasoning") or ""
        content = item.get("content") or ""
        if reasoning:
            if not reasoning_added:
                reasoning_added = True
                yield {"type": "response.output_item.added", "output_index": 0, "item": reasoning_output_item("", reasoning_id, "in_progress")}
            full_reasoning += reasoning
            yield {
                "type": "response.reasoning_summary_text.delta",
                "item_id": reasoning_id,
                "output_index": 0,
                "summary_index": 0,
                "delta": reasoning,
            }
        if content:
            if reasoning_added and not text_added:
                yield {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": reasoning_id,
                    "output_index": 0,
                    "summary_index": 0,
                    "text": full_reasoning,
                }
                yield {"type": "response.output_item.done", "output_index": 0, "item": reasoning_output_item(full_reasoning, reasoning_id, "completed")}
                text_output_index = 1
            if not text_added:
                text_added = True
                yield {"type": "response.output_item.added", "output_index": text_output_index, "item": text_output_item("", item_id, "in_progress")}
            full_text += content
            yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": text_output_index, "content_index": 0, "delta": content}
    if reasoning_added and not text_added:
        yield {
            "type": "response.reasoning_summary_text.done",
            "item_id": reasoning_id,
            "output_index": 0,
            "summary_index": 0,
            "text": full_reasoning,
        }
        yield {"type": "response.output_item.done", "output_index": 0, "item": reasoning_output_item(full_reasoning, reasoning_id, "completed")}
        text_output_index = 1
    if not text_added:
        yield {"type": "response.output_item.added", "output_index": text_output_index, "item": text_output_item("", item_id, "in_progress")}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": text_output_index, "content_index": 0, "text": full_text}
    item = text_output_item(full_text, item_id, "completed")
    yield {"type": "response.output_item.done", "output_index": text_output_index, "item": item}
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(full_text, model),
    )
    output = [reasoning_output_item(full_reasoning, reasoning_id, "completed"), item] if reasoning_added else [item]
    yield response_completed(response_id, model, created, output, usage)


def stream_web_search_response(body: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Iterator[dict[str, Any]]:
    model = str(body.get("model") or "auto").strip() or "auto"
    messages = messages if messages is not None else messages_from_input(body.get("input"), body.get("instructions"))
    query = search_query_from_messages(messages) or extract_response_prompt(body.get("input"))
    if not query:
        raise HTTPException(status_code=400, detail={"error": "input text is required for web_search"})

    response_id = f"resp_{uuid.uuid4().hex}"
    search_id = f"ws_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)

    searching_item = web_search_call_item(query, search_id, "in_progress")
    yield {"type": "response.output_item.added", "output_index": 0, "item": searching_item}
    yield {"type": "response.web_search_call.in_progress", "output_index": 0, "item_id": search_id}
    yield {"type": "response.web_search_call.searching", "output_index": 0, "item_id": search_id}
    result = run_web_search(query)
    search_item = web_search_call_item(query, search_id, "completed", normalized_sources(result))
    yield {"type": "response.web_search_call.completed", "output_index": 0, "item_id": search_id}
    yield {"type": "response.output_item.done", "output_index": 0, "item": search_item}

    text, annotations = text_with_url_citations(result)
    message_item = text_output_item("", item_id, "in_progress", annotations)
    yield {"type": "response.output_item.added", "output_index": 1, "item": message_item}
    if text:
        yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 1, "content_index": 0, "delta": text}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 1, "content_index": 0, "text": text}
    message_item = text_output_item(text, item_id, "completed", annotations)
    yield {"type": "response.output_item.done", "output_index": 1, "item": message_item}
    usage = token_usage(
        input_text_tokens=count_message_text_tokens(messages, model),
        input_image_tokens=count_message_image_tokens(messages, model),
        output_text_tokens=count_text_tokens(text, model),
    )
    yield response_completed(response_id, model, created, [search_item, message_item], usage)


def stream_image_response(
    image_outputs: Iterable[ImageOutput],
    prompt: str,
    model: str,
    input_image_tokens: int = 0,
    size: object = None,
    quality: str = "auto",
) -> Iterator[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)
    for output in image_outputs:
        if output.kind == "message":
            text = output.text
            item = text_output_item(text)
            usage = token_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_text_tokens=count_text_tokens(text, model),
            )
            yield {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0, "content_index": 0, "delta": text}
            yield {"type": "response.output_text.done", "item_id": item["id"], "output_index": 0, "content_index": 0, "text": text}
            yield {"type": "response.output_item.done", "output_index": 0, "item": item}
            yield response_completed(response_id, model, created, [item], usage)
            return
        if output.kind != "result":
            continue
        items = image_output_items(prompt, output.data)
        if items:
            usage = image_usage(
                input_text_tokens=count_text_tokens(prompt, model),
                input_image_tokens=input_image_tokens,
                output_tokens=count_image_output_items_tokens(output.data, size, quality),
            )
            for output_index, item in enumerate(items):
                yield {"type": "response.output_item.done", "output_index": output_index, "item": item}
            yield response_completed(response_id, model, created, items, usage)
            return
    raise RuntimeError("image generation failed")


def _stream_response_text(model: str, chunks: Iterable[str]) -> Iterator[dict[str, Any]]:
    """把一串文本片段包成 responses 事件流（单条文本传 [text] 即可）。"""
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    yield response_created(response_id, model, created)
    yield {"type": "response.output_item.added", "output_index": 0, "item": text_output_item("", item_id, "in_progress")}
    full_text = ""
    for chunk in chunks:
        if not chunk:
            continue
        full_text += chunk
        yield {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": chunk}
    yield {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": full_text}
    item = text_output_item(full_text, item_id, "completed")
    yield {"type": "response.output_item.done", "output_index": 0, "item": item}
    usage = token_usage(input_text_tokens=0, input_image_tokens=0, output_text_tokens=count_text_tokens(full_text, model))
    yield response_completed(response_id, model, created, [item], usage)


def deep_research_response_handle(body: dict[str, Any], identity: dict[str, Any] | None = None,
                                 base_url: str = "") -> dict[str, Any] | Iterator[dict[str, Any]]:
    """深度研究的 responses 两轮处理（model=deep-research）。"""
    messages = messages_from_input(body.get("input"), body.get("instructions"))
    model = str(body.get("model") or deep_research_protocol.DEEP_RESEARCH_MODEL)
    stream = bool(body.get("stream"))
    try:
        result = deep_research_protocol.start_or_resume(
            identity, messages, base_url, fallback_prompt=extract_response_prompt(body.get("input")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc
    phase = result["phase"]
    if phase == "reply":  # hi / ping 等琐碎提示 → 秒回，不进深研
        events = _stream_response_text(model, [result.get("text") or ""])
        return events if stream else collect_response(events)
    task = result["task"]
    if phase == "progress" and stream:
        events = _stream_response_text(model, deep_research_protocol.iter_progress_text(str(task.get("id") or "")))
    elif phase == "plan":
        events = _stream_response_text(model, [deep_research_protocol.render_plan_markdown(task)])
    else:
        events = _stream_response_text(model, [deep_research_protocol.status_markdown(task)])
    return events if stream else collect_response(events)


def collect_response(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    completed = {}
    for event in events:
        if event.get("type") == "response.completed":
            completed = event.get("response") if isinstance(event.get("response"), dict) else {}
    if not completed:
        raise RuntimeError("response generation failed")
    return completed


def response_events(body: dict[str, Any], base_url: str = "") -> Iterator[dict[str, Any]]:
    if is_text_response_request(body):
        model, messages = text_response_parts(body)
        if has_web_search_tool(body) and not has_unsupported_response_tools(body):
            yield from stream_web_search_response(body, messages)
            return
        key = cache_key(body, messages, stream=bool(body.get("stream")))
        yield from chat_completion_cache.get_or_compute_stream(
            key,
            lambda: stream_text_response(text_backend(model), body, messages, base_url=base_url),
        )
        return

    prompt = extract_response_prompt(body.get("input"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "input text is required"})
    model = str(body.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    image_info = extract_response_image(body.get("input"))
    if image_info:
        image_data, mime_type = image_info
        images = encode_images([(image_data, "image.png", mime_type)])
    else:
        images = None
    input_image_tokens = count_image_content_tokens(_input_image_parts(body.get("input")), model)
    tool = response_image_tool(body)
    image_outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        size=tool.get("size"),
        quality=str(tool.get("quality") or "auto"),
        response_format="b64_json",
        images=images,
    ))
    yield from stream_image_response(image_outputs, prompt, model, input_image_tokens, tool.get("size"), str(tool.get("quality") or "auto"))


def handle(body: dict[str, Any], base_url: str = "") -> dict[str, Any] | Iterator[dict[str, Any]]:
    events = response_events(body, base_url)
    if body.get("stream"):
        return events
    return collect_response(events)
