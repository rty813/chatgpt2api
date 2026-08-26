from __future__ import annotations

import html
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from utils.markdown_math import KATEX_HEAD, KATEX_JS, MATH_CSS, has_math, install_math

# 报告 markdown → HTML 渲染器（html=False 防注入；开启表格/删除线；\( \) \[ \] 公式交给 KaTeX）
_MD = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table").enable("strikethrough")
install_math(_MD)

# fence 渲染：mermaid / flowchart 代码块输出 <pre class="mermaid">，交给前端 mermaid.js 渲染成流程图；
# 其他语言仍走默认代码块渲染。mermaid 加载失败时原样降级为预格式文本。
_default_fence = _MD.renderer.rules["fence"]


def _render_fence(tokens, idx, options, env):  # noqa: ANN001
    token = tokens[idx]
    info = (token.info or "").strip().split(maxsplit=1)
    lang = info[0].lower() if info else ""
    if lang in ("mermaid", "flowchart"):
        return f'<pre class="mermaid">{html.escape(token.content)}</pre>\n'
    return _default_fence(tokens, idx, options, env)


_MD.renderer.rules["fence"] = _render_fence

# mermaid.js（ESM，按需注入）：跟随系统深浅色，渲染本页 pre.mermaid；
# 渲染完成后为每张流程图加「点击全屏放大（滚轮缩放 / 拖动平移）」与「新标签打开 SVG」。
_MERMAID_JS = """<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme:dark)').matches;
mermaid.initialize({startOnLoad:false, theme:dark?'dark':'default', securityLevel:'loose'});
mermaid.run({querySelector:'pre.mermaid'}).then(enhance).catch(function(e){console.error('mermaid',e);});

function enhance(){
  document.querySelectorAll('pre.mermaid').forEach(function(pre){
    const svg = pre.querySelector('svg');
    if(!svg) return;
    pre.addEventListener('click', function(){ openZoom(svg); });
  });
}

function openZoom(srcSvg){
  const rect = srcSvg.getBoundingClientRect();
  const vw = window.innerWidth, vh = window.innerHeight;
  const ov = document.createElement('div'); ov.className = 'mzoom';
  const stage = document.createElement('div'); stage.className = 'mzstage';
  const svg = srcSvg.cloneNode(true);
  svg.style.width = rect.width + 'px';
  svg.style.height = rect.height + 'px';
  svg.style.maxWidth = 'none'; svg.style.maxHeight = 'none';
  stage.appendChild(svg); ov.appendChild(stage);

  const base = Math.max(0.1, Math.min(vw*0.92/rect.width, vh*0.92/rect.height));
  let scale = base;
  let tx = (vw - rect.width*scale)/2, ty = (vh - rect.height*scale)/2;
  function apply(){ svg.style.transform = 'translate('+tx+'px,'+ty+'px) scale('+scale+')'; }
  function zoomAt(cx, cy, factor){
    const ns = Math.max(0.05, Math.min(20, scale*factor));
    const ix = (cx - tx)/scale, iy = (cy - ty)/scale;
    scale = ns; tx = cx - ix*scale; ty = cy - iy*scale; apply();
  }
  function reset(){ scale = base; tx = (vw-rect.width*scale)/2; ty = (vh-rect.height*scale)/2; apply(); }

  ov.addEventListener('wheel', function(e){ e.preventDefault(); zoomAt(e.clientX, e.clientY, e.deltaY<0?1.15:1/1.15); }, {passive:false});
  let dragging=false, moved=false, lx=0, ly=0;
  ov.addEventListener('pointerdown', function(e){ dragging=true; moved=false; lx=e.clientX; ly=e.clientY; try{ov.setPointerCapture(e.pointerId);}catch(_){} });
  ov.addEventListener('pointermove', function(e){ if(!dragging) return; const dx=e.clientX-lx, dy=e.clientY-ly; if(Math.abs(dx)+Math.abs(dy)>3) moved=true; tx+=dx; ty+=dy; lx=e.clientX; ly=e.clientY; apply(); });
  ov.addEventListener('pointerup', function(){ dragging=false; });
  ov.addEventListener('click', function(e){ if((e.target===ov||e.target===stage) && !moved) close(); });

  function close(){ ov.remove(); document.removeEventListener('keydown', onKey); }
  function onKey(e){ if(e.key==='Escape') close(); else if(e.key==='+'||e.key==='=') zoomAt(vw/2,vh/2,1.2); else if(e.key==='-') zoomAt(vw/2,vh/2,1/1.2); }
  document.addEventListener('keydown', onKey);

  const bar = document.createElement('div'); bar.className='mzbar';
  function mk(txt, title, fn){ const b=document.createElement('button'); b.textContent=txt; b.title=title; b.addEventListener('click', function(e){ e.stopPropagation(); fn(); }); bar.appendChild(b); }
  mk('+','放大', function(){ zoomAt(vw/2,vh/2,1.2); });
  mk('−','缩小', function(){ zoomAt(vw/2,vh/2,1/1.2); });
  mk('↺','重置', reset);
  mk('↗','在新标签打开', function(){ openInTab(srcSvg); });
  mk('✕','关闭', close);
  ov.appendChild(bar);

  const hint = document.createElement('div'); hint.className='mzhint';
  hint.textContent = '滚轮缩放 · 拖动平移 · Esc 关闭';
  ov.appendChild(hint);

  document.body.appendChild(ov); apply();
}

function openInTab(srcSvg){
  try{
    const s = new XMLSerializer().serializeToString(srcSvg);
    const blob = new Blob([s], {type:'image/svg+xml;charset=utf-8'});
    window.open(URL.createObjectURL(blob), '_blank', 'noopener');
  }catch(e){ console.error('open svg', e); }
}
</script>"""

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
.report pre.mermaid{position:relative;background:none;border:none;padding:8px 0;text-align:center;overflow:auto;line-height:normal;cursor:zoom-in}
.report pre.mermaid svg{max-width:100%;height:auto}
.report pre.mermaid::after{content:"🔍 点击放大";position:absolute;top:4px;right:6px;font-size:11px;color:var(--muted);background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1px 7px;opacity:.8;pointer-events:none}
.mzoom{position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,.86);overflow:hidden;touch-action:none;user-select:none}
.mzoom .mzstage{position:absolute;inset:0}
.mzoom svg{position:absolute;left:0;top:0;transform-origin:0 0;cursor:grab}
.mzoom .mzbar{position:fixed;top:14px;right:16px;display:flex;gap:8px;z-index:100000}
.mzoom .mzbar button{min-width:40px;height:40px;padding:0 10px;background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.3);border-radius:10px;font-size:18px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center}
.mzoom .mzbar button:hover{background:rgba(255,255,255,.3)}
.mzoom .mzhint{position:fixed;bottom:16px;left:0;right:0;text-align:center;color:rgba(255,255,255,.72);font-size:12.5px;pointer-events:none}
.report blockquote{margin:12px 0;padding:4px 16px;border-left:3px solid var(--accent);color:var(--muted)}
.report table{border-collapse:collapse;margin:14px 0;font-size:14px;display:block;overflow-x:auto;max-width:100%}
.report th,.report td{border:1px solid var(--border);padding:8px 12px;text-align:left;vertical-align:top}
.report th{background:var(--bg);font-weight:600}
.report a{color:var(--accent);text-decoration:none}.report a:hover{text-decoration:underline}
.report img{max-width:100%}
.report hr{border:none;border-top:1px solid var(--border);margin:22px 0}
""" + MATH_CSS

_BADGES = {
    "awaiting_confirmation": ("wait", "⏸ 等待确认"),
    "queued": ("run", "⏳ 排队中"),
    "running": ("run", "🔬 研究进行中"),
    "exporting": ("run", "📤 正在导出报告"),
    "success": ("ok", "✅ 已完成"),
    "error": ("err", "❌ 失败"),
}


def _page(inner: str, auto_refresh: bool, head_extra: str = "") -> str:
    refresh = '<meta http-equiv="refresh" content="8">' if auto_refresh else ""
    foot = "本页每 8 秒自动刷新" if auto_refresh else "深度研究 · 最终结果"
    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">{refresh}'
            f'<title>深度研究</title><style>{_CSS}</style>{head_extra}</head>'
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
        head_extra = ""
        if md:  # 首选：渲染 markdown 正文（阅读体验最好），附 PDF 下载
            buttons = ""
            if pdf:
                fn = _esc(task.get("pdf_filename") or "report.pdf")
                buttons = (f'<a class="btn" href="{pdf}" download>📄 下载 PDF</a>'
                           f'<a class="btn ghost" href="{pdf}" target="_blank" rel="noopener">PDF 预览</a>')
            report_html = _MD.render(md)
            math = has_math(report_html)  # 含公式时才加载 KaTeX（样式表进 <head>，脚本随正文）
            head_extra = KATEX_HEAD if math else ""
            inner = (head
                     + f'<p style="margin:18px 0 10px">参考来源 {srcs} 条。</p>'
                     + buttons
                     + f'<div class="report">{report_html}</div>'
                     + (_MERMAID_JS if 'class="mermaid"' in report_html else "")
                     + (KATEX_JS if math else ""))
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
        return _page(inner, False, head_extra)

    if status == "error":
        cg = _esc(task.get("chatgpt_url"))
        link = (f'<a class="btn ghost" href="{cg}" target="_blank" rel="noopener" '
                f'style="margin-top:14px">在 ChatGPT 查看</a>' if cg else "")
        return _page(head + f'<div class="err">{_esc(task.get("error") or "未知错误")}</div>' + link, False)

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
        """下载报告 PDF（重定向到本地托管的 /files 下载链接）。"""
        task = await run_in_threadpool(deep_research_service.get_research, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail={"error": "task not found"})
        pdf_url = task.get("pdf_url")
        if not pdf_url:
            raise HTTPException(status_code=409, detail={"error": f"report not ready (status={task.get('status')})"})
        return RedirectResponse(url=pdf_url)

    return router
