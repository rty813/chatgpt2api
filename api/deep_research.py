from __future__ import annotations

import html
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

# 报告 markdown → HTML 渲染器（html=False 防注入；开启表格/删除线）
_MD = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table").enable("strikethrough")

from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request
from services.deep_research_service import deep_research_service
from services.log_service import LoggedCall


class DeepResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    deep_research_version: str = "standard"


_CSS = """
*{box-sizing:border-box}
:root{--bg:#f5f6f8;--card:#fff;--fg:#1f2328;--muted:#6b7280;--accent:#10a37f;--accent2:#0d8a6c;--border:#e6e8eb}
@media(prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171b21;--fg:#e6e8eb;--muted:#9aa4b2;--accent:#19c39a;--accent2:#10a37f;--border:#262c35}}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--fg);line-height:1.65;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:28px 16px 48px}
.card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:30px 32px;
 box-shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.05)}
h1{font-size:23px;line-height:1.35;margin:0 0 6px;display:flex;align-items:center;gap:10px}
.sub{color:var(--muted);font-size:13px;margin:0 0 18px}
.badge{display:inline-block;padding:4px 14px;border-radius:999px;font-size:13px;font-weight:600;letter-spacing:.2px}
.badge.run{background:#fff4d6;color:#8a5a00}.badge.ok{background:#d8f5e8;color:#0a6b4a}
.badge.err{background:#fde2e2;color:#9b1c1c}.badge.wait{background:#e9edf2;color:#3b4452}
@media(prefers-color-scheme:dark){.badge.run{background:#3a2f10;color:#f2c64d}.badge.ok{background:#103a2c;color:#52e0b0}
 .badge.err{background:#3d1f1f;color:#f08a8a}.badge.wait{background:#252b34;color:#aeb6c2}}
.steps{margin:14px 0;padding-left:24px}.steps li{margin:7px 0}
.progress{display:flex;align-items:center;gap:10px;background:var(--bg);border:1px solid var(--border);
 border-radius:12px;padding:13px 16px;margin:16px 0;font-size:14.5px}
.stats{display:flex;gap:14px;margin:18px 0;flex-wrap:wrap}
.stat{flex:1;min-width:96px;text-align:center;background:var(--bg);border:1px solid var(--border);border-radius:12px;padding:14px 8px}
.stat b{display:block;font-size:25px;font-weight:700}.stat span{color:var(--muted);font-size:12px}
.btn{display:inline-block;background:var(--accent);color:#fff;padding:12px 24px;border-radius:11px;
 text-decoration:none;font-weight:600;font-size:15px;margin:6px 10px 6px 0;transition:background .15s}
.btn:hover{background:var(--accent2)}
.btn.ghost{background:transparent;color:var(--accent);border:1.5px solid var(--accent)}
.spin{width:17px;height:17px;border:2.5px solid var(--border);border-top-color:var(--accent);
 border-radius:50%;display:inline-block;animation:s .9s linear infinite;flex:none}
@keyframes s{to{transform:rotate(360deg)}}
iframe.pdf{width:100%;height:82vh;border:1px solid var(--border);border-radius:14px;margin-top:20px;background:#fff}
.err{background:#fde2e2;color:#9b1c1c;padding:15px 18px;border-radius:12px;margin-top:14px}
@media(prefers-color-scheme:dark){.err{background:#3d1f1f;color:#f3a3a3}}
.foot{color:var(--muted);font-size:12px;text-align:center;margin-top:18px}
.hint{color:var(--muted);font-size:13.5px;margin-top:14px}
.sec{font-weight:600;font-size:15px;margin:22px 0 8px}
.feed{list-style:none;padding:0;margin:0}
.feed li{padding:8px 0 8px 16px;border-left:2px solid var(--border);color:var(--muted);font-size:14px;position:relative}
.feed li::before{content:"";position:absolute;left:-5px;top:14px;width:8px;height:8px;border-radius:50%;background:var(--border)}
.feed li.cur{border-left-color:var(--accent);color:var(--fg);font-weight:600}
.feed li.cur::before{background:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent)}
.report{margin-top:20px;line-height:1.78;font-size:15px;word-wrap:break-word;overflow-wrap:anywhere}
.report h1{font-size:24px;margin:26px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.report h2{font-size:20px;margin:24px 0 10px}
.report h3{font-size:17px;margin:18px 0 8px}
.report p{margin:11px 0}
.report ul,.report ol{padding-left:24px;margin:10px 0}.report li{margin:5px 0}
.report code{background:var(--bg);padding:2px 6px;border-radius:5px;font-size:.9em;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.report pre{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px;overflow:auto}
.report pre code{background:none;padding:0}
.report blockquote{margin:12px 0;padding:4px 16px;border-left:3px solid var(--accent);color:var(--muted)}
.report table{border-collapse:collapse;margin:14px 0;font-size:14px;display:block;overflow-x:auto;max-width:100%}
.report th,.report td{border:1px solid var(--border);padding:8px 12px;text-align:left;vertical-align:top}
.report th{background:var(--bg);font-weight:600}
.report a{color:var(--accent);text-decoration:none}.report a:hover{text-decoration:underline}
.report img{max-width:100%}
.report hr{border:none;border-top:1px solid var(--border);margin:22px 0}
"""

_BADGES = {
    "awaiting_confirmation": ("wait", "⏸ 等待确认"),
    "queued": ("run", "⏳ 排队中"),
    "running": ("run", "🔬 研究进行中"),
    "exporting": ("run", "📤 正在导出报告"),
    "success": ("ok", "✅ 已完成"),
    "error": ("err", "❌ 失败"),
}


def _page(inner: str, auto_refresh: bool) -> str:
    refresh = '<meta http-equiv="refresh" content="8">' if auto_refresh else ""
    foot = "本页每 8 秒自动刷新" if auto_refresh else "深度研究 · 最终结果"
    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}'
            f'<title>深度研究</title><style>{_CSS}</style></head>'
            f'<body><div class="wrap"><div class="card">{inner}</div>'
            f'<p class="foot">{foot}</p></div></body></html>')


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _local_time_html(task: dict[str, Any]) -> str:
    """更新时间：默认显示服务端字符串，JS 用 epoch 转成访问者设备的本地时区时间。"""
    server = _esc(task.get("updated_at"))
    ts = task.get("updated_ts")
    try:
        ms = int(float(ts) * 1000)
    except (TypeError, ValueError):
        ms = 0
    if not ms:
        return f'更新于 {server}'
    return (f'更新于 <span id="ut">{server}</span>'
            f'<script>(function(){{try{{document.getElementById("ut").textContent='
            f'new Date({ms}).toLocaleString();}}catch(e){{}}}})();</script>')


def _render_status_html(task: dict[str, Any]) -> str:
    status = str(task.get("status") or "")
    bcls, btext = _BADGES.get(status, ("wait", _esc(status)))
    head = (f'<h1>🔬 {_esc(task.get("title") or "深度研究")}</h1>'
            f'<p class="sub">任务 {_esc(str(task.get("id") or "")[:8])} · 第 {_esc(task.get("plan_version") or 1)} 版'
            f' · {_local_time_html(task)}</p>'
            f'<span class="badge {bcls}">{btext}</span>')

    if status == "queued":
        pos = int(task.get("queue_position") or 1)
        ahead = max(0, pos - 1)
        msg = (f"前面还有 <b>{ahead}</b> 个研究在进行/排队，轮到后会自动开始。"
               if ahead else "即将开始……")
        inner = (head
                 + '<div class="progress"><span class="spin"></span><span>已加入队列</span></div>'
                 + f'<p class="hint">同一时间只运行一个深度研究。{msg}本页每 8 秒自动刷新。</p>')
        return _page(inner, True)

    if status == "success":
        pdf = _esc(task.get("pdf_url"))
        md = task.get("report_markdown") or ""
        srcs = _esc(task.get("sources_count") or 0)
        if md:  # 首选：渲染 markdown 正文（阅读体验最好），附 PDF 下载
            buttons = ""
            if pdf:
                fn = _esc(task.get("pdf_filename") or "report.pdf")
                buttons = (f'<a class="btn" href="{pdf}" download>📄 下载 PDF</a>'
                           f'<a class="btn ghost" href="{pdf}" target="_blank" rel="noopener">PDF 预览</a>')
            inner = (head
                     + f'<p style="margin:18px 0 10px">参考来源 {srcs} 条。</p>'
                     + buttons
                     + f'<div class="report">{_MD.render(md)}</div>')
        elif pdf:  # 无 markdown 但有 PDF → 退回内嵌 PDF 预览
            fn = _esc(task.get("pdf_filename") or "report.pdf")
            inner = (head
                     + f'<p style="margin:20px 0 6px">报告已生成（参考来源 {srcs} 条）。</p>'
                     + f'<a class="btn" href="{pdf}" download>📄 下载 PDF 报告</a>'
                     + f'<a class="btn ghost" href="{pdf}" target="_blank" rel="noopener">在新标签打开</a>'
                     + f'<iframe class="pdf" src="{pdf}" title="{fn}"></iframe>')
        else:  # 完成但报告取不到 → 给 ChatGPT 查看链接
            cg = _esc(task.get("chatgpt_url"))
            inner = (head
                     + f'<p style="margin:20px 0 6px">{_esc(task.get("report_note") or "研究已完成。")}</p>'
                     + (f'<a class="btn" href="{cg}" target="_blank" rel="noopener">在 ChatGPT 查看报告</a>' if cg else ""))
        return _page(inner, False)

    if status == "error":
        return _page(head + f'<div class="err">{_esc(task.get("error") or "未知错误")}</div>', False)

    if status in ("running", "exporting"):
        elapsed = task.get("elapsed_secs")
        elapsed_s = f"{int(elapsed)}s" if isinstance(elapsed, (int, float)) else "—"
        prog = _esc(task.get("progress") or "正在准备研究…")
        # 研究计划（步骤列表）
        plan_steps = (task.get("plan") or {}).get("steps") or []
        plan_html = ""
        if plan_steps:
            items = "".join(f"<li>{_esc(s.get('text') if isinstance(s, dict) else s)}</li>" for s in plan_steps)
            plan_html = f'<p class="sec">研究计划</p><ol class="steps">{items}</ol>'
        # 研究进展（活动流，最新在上、当前高亮）
        activities = task.get("activities") or []
        feed_html = ""
        if activities:
            lis_parts = []
            for i, a in enumerate(reversed(activities)):
                cls = ' class="cur"' if i == 0 else ""
                lis_parts.append(f"<li{cls}>{_esc(a)}</li>")
            feed_html = (f'<p class="sec">研究进展（最近 {len(activities)} 步）</p>'
                         f'<ul class="feed">{"".join(lis_parts)}</ul>')
        inner = (head
                 + f'<div class="progress"><span class="spin"></span><span>{prog}</span></div>'
                 + '<div class="stats">'
                 + f'<div class="stat"><b>{_esc(task.get("num_messages") or 0)}</b><span>研究步骤</span></div>'
                 + f'<div class="stat"><b>{_esc(task.get("sources_count") or 0)}</b><span>参考来源</span></div>'
                 + f'<div class="stat"><b>{elapsed_s}</b><span>已用时</span></div></div>'
                 + plan_html + feed_html
                 + '<p class="hint">深度研究通常需要 5–30 分钟；本页每 8 秒自动刷新，完成后将在此显示报告并提供下载。</p>')
        return _page(inner, True)

    # awaiting_confirmation
    steps = (task.get("plan") or {}).get("steps") or []
    steps_html = "".join(f"<li>{_esc(s.get('text') if isinstance(s, dict) else s)}</li>" for s in steps)
    inner = (head
             + ('<p style="margin-top:18px;font-weight:600">研究计划</p>' if steps_html else "")
             + (f'<ol class="steps">{steps_html}</ol>' if steps_html else "")
             + '<p class="hint">请在对话中回复 <b>开始</b> 启动研究；或直接发送修改意见调整计划（提交后会按修改自动开始）。</p>')
    return _page(inner, True)


def _wants_html(request: Request, fmt: str) -> bool:
    if fmt == "json":
        return False
    if fmt == "html":
        return True
    return "text/html" in (request.headers.get("accept") or "")


def create_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/deep-research")
    async def create_deep_research(
        body: DeepResearchRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """触发深度研究，返回研究计划（待确认）。需调用 /confirm 才会开始执行。"""
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/deep-research", "deep-research", "深度研究计划", request_text=body.prompt)
        try:
            await run_in_threadpool(check_request, body.prompt)
        except HTTPException as exc:
            call.log("调用失败", status="failed", error=str(exc.detail))
            raise
        try:
            return await run_in_threadpool(
                deep_research_service.create_research,
                identity,
                prompt=body.prompt,
                base_url=resolve_image_base_url(request),
                deep_research_version=body.deep_research_version,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/deep-research/{task_id}/confirm")
    async def confirm_deep_research(
        task_id: str,
        authorization: str | None = Header(default=None),
    ):
        """确认研究计划 → 后台开始执行并轮询。立即返回状态（含 status_url）。"""
        identity = require_identity(authorization)
        try:
            return await run_in_threadpool(deep_research_service.confirm_research, identity, task_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail={"error": str(exc)}) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.get("/v1/deep-research")
    async def list_deep_research(authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        return {"items": await run_in_threadpool(deep_research_service.list_research, identity)}

    @router.get("/api/deep-research/{task_id}")
    async def deep_research_status(task_id: str, request: Request, format: str = ""):
        """状态/结果页：浏览器访问返回精美 HTML（含报告预览+下载）；API 客户端（Accept 非 html 或 ?format=json）返回 JSON。"""
        task = await run_in_threadpool(deep_research_service.get_research, task_id)
        if task is None:
            if _wants_html(request, format):
                body = ('<h1>🔍 未找到该研究任务</h1>'
                        '<p class="hint">任务可能已过期或链接有误。</p>')
                return HTMLResponse(_page(body, False), status_code=404)
            raise HTTPException(status_code=404, detail={"error": "task not found"})
        if _wants_html(request, format):
            return HTMLResponse(_render_status_html(task))
        return task

    @router.get("/api/deep-research/{task_id}/download")
    async def deep_research_download(task_id: str):
        """下载报告 PDF（重定向到 transfer 的下载链接）。"""
        task = await run_in_threadpool(deep_research_service.get_research, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail={"error": "task not found"})
        pdf_url = task.get("pdf_url")
        if not pdf_url:
            raise HTTPException(status_code=409, detail={"error": f"report not ready (status={task.get('status')})"})
        return RedirectResponse(url=pdf_url)

    return router
