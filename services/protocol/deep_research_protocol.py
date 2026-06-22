"""Deep Research（深度研究）的 OpenAI 兼容两轮交互（chat + responses 共用核心）。

两轮：
  turn 1（用户首条问题）→ 触发深研、返回研究计划（含隐藏 task 标记 + 状态 URL）。
  turn 2（用户在带计划的历史后再发一条）→ 确认开跑、后台轮询，流式吐进度+最终 PDF 链接。

格式无关：本模块只产出纯文本片段，由各协议文件包成 chat chunk / responses event。
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterator

from services.deep_research_service import MAX_RESEARCH_SECS, deep_research_service

DEEP_RESEARCH_MODEL = "deep-research"
# 隐藏在 assistant 计划消息里的 task 标记；客户端把历史回传时据此识别是第二轮（确认/修改）。
_TASK_MARKER_RE = re.compile(r"deep-research-task:([0-9a-f]{32})")
# 第二轮里这种"短确认"才视为开始研究；其余文本一律当作对计划的修改意见 → 重新生成计划。
_CONFIRM_RE = re.compile(
    r"^\s*(开始|开始吧|开始研究|确认|確認|确定|可以|可以了|行|好|好的|就这样|没问题|同意|"
    r"go|go ahead|start|run|begin|confirm|confirmed|ok|okay|yes|y|proceed|looks good|sounds good)"
    r"\s*[!！。.~]*\s*$",
    re.IGNORECASE,
)
_PROGRESS_POLL_SECS = 10.0
_STREAM_MAX_SECS = MAX_RESEARCH_SECS + 120.0


def is_confirmation(text: object) -> bool:
    return bool(_CONFIRM_RE.match(str(text or "")))


# 健康检查/试探类的琐碎提示词：直接秒回，不触发（烧 Pro 额度的）深度研究。
_TRIVIAL_REPLIES = {
    "hi": "hi", "hii": "hi", "hello": "hello", "hey": "hey", "yo": "yo",
    "hi there": "hi", "hello there": "hello", "are you there": "yes, I'm here",
    "ping": "pong", "test": "ok", "testing": "ok", "ok": "ok",
    "你好": "你好", "您好": "您好", "在吗": "在", "在不在": "在", "测试": "ok",
}


def trivial_reply(text: object) -> str | None:
    """琐碎招呼/试探 → 返回秒回内容；否则 None（照常进深研）。"""
    key = str(text or "").strip().lower().rstrip("!?.~,。！？，、 \t")
    return _TRIVIAL_REPLIES.get(key)


def _status_url_callout(url: object) -> str:
    """状态/下载页的统一说明文案——突出"持久化、断流也能查进度+下载结果"。"""
    url = str(url or "").strip()
    if not url:
        return "⚠️ 未配置 base_url，无法生成持久化进度页链接（请设置 CHATGPT2API_BASE_URL）。"
    return ("📌 **任务持久化页面**（深度研究耗时较长，此链接独立保存，**即使本次 SSE 流中断也不受影响**）：\n"
            f"{url}\n"
            "随时打开它即可查看实时研究进度，研究完成后也在该页在线预览并下载 PDF 报告。")


def is_deep_research_model(model: object) -> bool:
    name = str(model or "").strip().lower()
    return name == DEEP_RESEARCH_MODEL or name.startswith("deep-research")


def is_deep_research_request(body: dict[str, Any]) -> bool:
    return is_deep_research_model(body.get("model"))


def _marker(task_id: str) -> str:
    return f"<!-- deep-research-task:{task_id} -->"


def text_of(content: object) -> str:
    """把 chat / responses 的消息 content（str 或 part 列表）摊平成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("input_text") or part.get("output_text") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def find_task_id(messages: list[dict[str, Any]]) -> str:
    """从历史里（assistant 计划消息）找出 task 标记 → 说明这是确认轮。"""
    for message in reversed(messages):
        if str(message.get("role")) != "assistant":
            continue
        match = _TASK_MARKER_RE.search(text_of(message.get("content")))
        if match:
            return match.group(1)
    return ""


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role")) == "user":
            text = text_of(message.get("content")).strip()
            if text:
                return text
    return ""


def render_plan_markdown(task: dict[str, Any]) -> str:
    plan = task.get("plan") or {}
    steps = plan.get("steps") or []
    version = task.get("plan_version") or 1
    suffix = f"（已按你的修改更新 · 第 {version} 版）" if version and version > 1 else ""
    lines = [f"## 🔬 深度研究计划：{task.get('title') or ''}{suffix}", ""]
    for index, step in enumerate(steps, start=1):
        text = step.get("text") if isinstance(step, dict) else str(step)
        lines.append(f"{index}. {text}")
    lines += [
        "",
        "确认请回复 **开始**（或「确认」「go」「ok」）按此计划启动研究；",
        "如需调整，直接发送修改意见（如「只看近一年」「聚焦推理框架」）——**提交后将按你的修改自动开始研究**。",
        "",
        _status_url_callout(task.get("status_url")),
        "",
        _marker(str(task.get("id") or "")),
    ]
    return "\n".join(lines)


def _queued_line(task: dict[str, Any]) -> str:
    ahead = max(0, int(task.get("queue_position") or 1) - 1)
    return f"⏳ 已加入队列（前面还有 {ahead} 个研究），轮到后自动开始。" if ahead else "⏳ 已加入队列，等待开始。"


def status_markdown(task: dict[str, Any]) -> str:
    status = task.get("status")
    if status == "success":
        return (f"✅ 深度研究已完成。\n\n报告 PDF：[{task.get('pdf_filename')}]({task.get('pdf_url')})\n\n"
                f"状态页：{task.get('status_url')}")
    if status == "error":
        return f"❌ 深度研究失败：{task.get('error') or '未知错误'}\n\n状态页：{task.get('status_url')}"
    if status == "queued":
        return f"{_queued_line(task)}\n\n{_status_url_callout(task.get('status_url'))}"
    return f"✅ 深度研究已在后台启动。\n\n{_status_url_callout(task.get('status_url'))}"


def iter_progress_text(task_id: str) -> Iterator[str]:
    """流式进度（纯文本片段）。开场给持久化页面链接（流断了也能查），队列中先报排队，再吐进度，结束吐报告/链接。"""
    task0 = deep_research_service.get_research(task_id)
    intro = "🔬 深度研究任务已提交。\n\n"
    if task0 and task0.get("status_url"):
        intro += _status_url_callout(task0.get("status_url")) + "\n\n"
    yield intro
    last_progress = ""
    queued_notified = False
    started_seen = False
    deadline = time.time() + _STREAM_MAX_SECS
    while time.time() < deadline:
        time.sleep(_PROGRESS_POLL_SECS)
        task = deep_research_service.get_research(task_id)
        if not task:
            yield "\n(任务状态丢失)\n"
            return
        status = task.get("status")
        if status == "queued":  # 排队中：单 worker，前序完成才轮到
            if not queued_notified:
                queued_notified = True
                yield _queued_line(task) + " 进度会在此持续输出…\n"
            continue
        if not started_seen and status in ("running", "exporting"):
            started_seen = True
            deadline = time.time() + _STREAM_MAX_SECS  # 真正开始后再给足时长，排队等待不计入
            yield "🔬 开始研究，正在检索与分析…\n"
        progress = str(task.get("progress") or "").strip()
        if progress and progress != last_progress:
            last_progress = progress
            yield f"- {progress}（{task.get('num_messages', 0)} 步 / {task.get('sources_count', 0)} 来源）\n"
        if status == "success":
            md = str(task.get("report_markdown") or "").strip()
            out = "\n\n"
            if md:  # 完整报告正文也通过 SSE 内联返回
                out += "---\n\n" + md + "\n\n---\n\n"
            out += "✅ 研究完成。\n"
            if task.get("pdf_url"):
                out += f"📄 下载 PDF：[{task.get('pdf_filename') or 'report.pdf'}]({task.get('pdf_url')})\n"
            if task.get("status_url"):
                out += f"🔗 状态/下载页：{task.get('status_url')}\n"
            yield out
            return
        if status == "error":
            yield f"\n❌ 研究失败：{task.get('error') or '未知错误'}\n"
            return
    yield f"\n⏳ 研究仍在进行，请稍后通过状态页查看：{task_id}\n"


def start_or_resume(identity: dict[str, Any] | None, messages: list[dict[str, Any]], base_url: str,
                    fallback_prompt: str = "") -> dict[str, Any]:
    """根据历史判定轮次并执行：
    返回 `{"phase": "plan"|"progress"|"status", "task": <public task>}`。
    - plan：返回计划（待确认）——第一轮，或第二轮发来修改意见→重出计划。
    - progress：已开跑，调用方应流式 iter_progress_text。
    - status：非流式 / 已终态，返回 status_markdown。
    """
    task_id = find_task_id(messages)
    if not task_id:
        prompt = latest_user_text(messages) or fallback_prompt
        if not prompt:
            raise ValueError("messages or input text is required for deep research")
        reply = trivial_reply(prompt)
        if reply is not None:  # hi / ping / test 等 → 秒回，不建任务、不烧额度
            return {"phase": "reply", "text": reply}
        task = deep_research_service.create_research(identity or {}, prompt=prompt, base_url=base_url)
        return {"phase": "plan", "task": task}
    task = deep_research_service.get_research(task_id)
    if not task:
        raise LookupError("deep research task not found or expired")
    user_text = latest_user_text(messages)
    # 第二轮仍在待确认：修改意见（非短确认）→ 原地编辑（同 session）；官方行为：提交修改后**自动开始研究**。
    if task.get("status") == "awaiting_confirmation" and user_text and not is_confirmation(user_text):
        task = deep_research_service.edit_research(identity, task_id, user_text, base_url=base_url)
    elif task.get("status") == "awaiting_confirmation":  # 短确认 → 入队开始研究
        task = deep_research_service.confirm_research(identity, task_id)
    return {"phase": "progress" if task.get("status") in ("queued", "running", "exporting") else "status", "task": task}
