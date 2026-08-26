from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from services.account_service import account_service
from services.config import DATA_DIR, config
from services.file_storage_service import file_storage_service
from services.log_service import LOG_TYPE_CALL, log_service
from services.openai_backend_api import DEEP_RESEARCH_STOP_STATUSES, OpenAIBackendAPI
from utils.log import logger

# 状态机：等待确认 → 排队 → 运行中 → 导出中 → 成功 / 失败
STATUS_AWAITING = "awaiting_confirmation"
STATUS_QUEUED = "queued"        # 已确认/已改、在队列里等单一 worker
STATUS_RUNNING = "running"
STATUS_EXPORTING = "exporting"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
TERMINAL_STATUSES = {STATUS_SUCCESS, STATUS_ERROR}
UNFINISHED_STATUSES = {STATUS_AWAITING, STATUS_QUEUED, STATUS_RUNNING, STATUS_EXPORTING}

DEEP_RESEARCH_PLAN_TYPE = "pro"  # 深度研究仅 Pro 账号
POLL_INTERVAL_SECS = 30.0
MAX_RESEARCH_SECS = 2400.0  # 40 分钟上限
REPORT_GRACE_SECS = 120.0  # 收到完成信号后，再宽限多久等 get_state 出现可导出的报告消息
STALL_NO_START_SECS = 480.0  # 开跑后这么久仍无任何研究消息（停在"准备研究"）→ 判定已停止/未开始，别傻等到 40min
AWAITING_EXPIRE_SECS = 1800.0  # awaiting 阶段被 pause 冻结超过这么久 → 上游多半已作废会话，标记过期、拒绝确认（窗口未知，按需调）
RETENTION_DAYS = 7


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _touch(task: dict[str, Any]) -> None:
    """刷新更新时间：updated_at（可读）+ updated_ts（epoch，供前端按设备本地时区显示）。"""
    task["updated_at"] = _now_iso()
    task["updated_ts"] = time.time()


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    try:
        return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object] | None) -> str:
    return _clean((identity or {}).get("id")) or "anonymous"


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    """对外可见字段（不含 access_token 等敏感信息）。"""
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "title": task.get("title"),
        "prompt": task.get("prompt"),
        "plan": task.get("plan"),
        "plan_version": task.get("plan_version", 1),
        "progress": task.get("progress"),
        "activities": task.get("activities") or [],
        "num_messages": task.get("num_messages"),
        "sources_count": task.get("sources_count"),
        "status_url": task.get("status_url"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "updated_ts": task.get("updated_ts"),
    }
    if task.get("pdf_url"):
        item["pdf_url"] = task.get("pdf_url")
        item["pdf_filename"] = task.get("pdf_filename")
    if task.get("report_markdown"):
        item["report_markdown"] = task.get("report_markdown")
    if task.get("report_note"):
        item["report_note"] = task.get("report_note")
    if task.get("chatgpt_url"):
        item["chatgpt_url"] = task.get("chatgpt_url")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("status") in (STATUS_RUNNING, STATUS_EXPORTING) and task.get("started_ts"):
        item["elapsed_secs"] = round(time.time() - task["started_ts"], 1)
    return item


class DeepResearchService:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()  # 单 worker 串行处理，保证同时只跑一个深研
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()
        threading.Thread(target=self._worker_loop, name="deep-research-worker", daemon=True).start()

    # ---------- public API ----------
    def create_research(self, identity: dict[str, object], *, prompt: str, base_url: str = "",
                        deep_research_version: str = "standard") -> dict[str, Any]:
        """触发深度研究，返回研究计划（状态 awaiting_confirmation，尚未开始执行）。"""
        prompt = _clean(prompt)
        if not prompt:
            raise ValueError("prompt is required")
        token = account_service.get_text_access_token(plan_type=DEEP_RESEARCH_PLAN_TYPE)
        if not token:
            raise RuntimeError("号池中没有可用的 Pro 账号（深度研究仅支持 Pro）")
        backend = OpenAIBackendAPI(access_token=token)
        info = backend.start_deep_research(prompt, deep_research_version=deep_research_version)
        token = backend.access_token or token  # token 可能在请求中轮换
        # 立即暂停：trigger 后约 60s 不确认上游会自动开跑；冻结它，强制必须由用户显式确认/修改才开始。
        try:
            backend.deep_research_pause(info["conversation_id"], info["message_id"], info["session_id"])
        except Exception as exc:
            logger.warning({"event": "deep_research_autopause_failed", "error": repr(exc)[:160]})
        plan = info.get("plan") or {}
        task_id = uuid.uuid4().hex
        now = _now_iso()
        account = account_service.get_account(token) or {}
        task = {
            "id": task_id,
            "owner_id": _owner_id(identity),
            "status": STATUS_AWAITING,
            "prompt": prompt,
            "title": _clean(plan.get("title")) or prompt[:40],
            "plan": plan,
            "deep_research_version": deep_research_version,
            "conversation_id": info.get("conversation_id"),
            "message_id": info.get("message_id"),
            "session_id": info.get("session_id"),
            "access_token": token,
            "account_email": _clean(account.get("email")),
            "base_url": base_url,
            "status_url": f"{base_url.rstrip('/')}/api/deep-research/{task_id}" if base_url else "",
            "progress": "",
            "num_messages": 0,
            "sources_count": 0,
            "created_at": now,
            "updated_at": now,
            "created_ts": time.time(),
            "updated_ts": time.time(),
        }
        with self._lock:
            self._tasks[task_id] = task
            self._save_locked()
        return _public_task(task)

    def edit_research(self, identity: dict[str, object] | None, task_id: str, edit_text: str,
                      base_url: str = "") -> dict[str, Any]:
        """修改计划：加入队列（单 worker 串行，同时只跑一个深研）。轮到时发 targeted_reply（上游自动开跑）。
        仅在 awaiting_confirmation 阶段可改。"""
        edit_text = _clean(edit_text)
        if not edit_text:
            raise ValueError("edit text is required")
        return self._enqueue(identity, task_id, "edit", edit_text)

    def confirm_research(self, identity: dict[str, object] | None, task_id: str) -> dict[str, Any]:
        """确认计划：加入队列（单 worker 串行）。轮到时 skip_sleep 开跑。立即返回（状态 queued/running）。"""
        return self._enqueue(identity, task_id, "confirm", "")

    def _enqueue(self, identity: dict[str, object] | None, task_id: str, action: str, edit_text: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(_clean(task_id))
            if task is None:
                raise ValueError("task not found")
            if identity is not None and task.get("owner_id") not in (_owner_id(identity), "anonymous"):
                raise PermissionError("not your task")
            if task.get("status") in (STATUS_QUEUED, STATUS_RUNNING, STATUS_EXPORTING):
                return self._public_with_queue_locked(task)  # 幂等：已在队列/运行中
            if self._maybe_expire_awaiting_locked(task):
                raise ValueError(task.get("error") or "研究计划已过期，请重新发起")
            if task.get("status") != STATUS_AWAITING:
                raise ValueError(f"task is not awaiting confirmation (status={task.get('status')})")
            task["status"] = STATUS_QUEUED
            task["pending_action"] = action
            if action == "edit":
                task["pending_edit"] = edit_text
            task["queued_at"] = time.time()
            _touch(task)
            self._save_locked()
            public = self._public_with_queue_locked(task)
        self._queue.put(_clean(task_id))
        return public

    def get_research(self, task_id: str, identity: dict[str, object] | None = None) -> dict[str, Any] | None:
        """按 task_id 取状态（capability：task_id 即 uuid，状态页无需鉴权）。"""
        with self._lock:
            task = self._tasks.get(_clean(task_id))
            if not task:
                return None
            self._maybe_expire_awaiting_locked(task)
            return self._public_with_queue_locked(task)

    def list_research(self, identity: dict[str, object]) -> list[dict[str, Any]]:
        owner = _owner_id(identity)
        with self._lock:
            for t in self._tasks.values():
                if t.get("owner_id") == owner:
                    self._maybe_expire_awaiting_locked(t)
            items = [self._public_with_queue_locked(t) for t in self._tasks.values() if t.get("owner_id") == owner]
        items.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
        return items

    def _public_with_queue_locked(self, task: dict[str, Any]) -> dict[str, Any]:
        """_public_task + 排队位置（须持锁调用）。"""
        pub = _public_task(task)
        if task.get("status") == STATUS_QUEUED:
            qa = task.get("queued_at") or 0
            ahead = sum(1 for t in self._tasks.values()
                        if t.get("status") in (STATUS_RUNNING, STATUS_EXPORTING)
                        or (t.get("status") == STATUS_QUEUED and (t.get("queued_at") or 0) < qa))
            pub["queue_position"] = ahead + 1
        return pub

    # ---------- 单 worker：串行执行队列里的深研 ----------
    def _worker_loop(self) -> None:
        while True:
            task_id = self._queue.get()
            try:
                self._run_queued(task_id)
            except Exception as exc:
                logger.warning({"event": "deep_research_worker_error", "task": task_id, "error": repr(exc)[:200]})
                self._update(task_id, status=STATUS_ERROR, error=str(exc) or "worker error")
            finally:
                self._queue.task_done()

    def _run_queued(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.get("status") != STATUS_QUEUED:
                return  # 已取消/无效
            action = task.get("pending_action") or "confirm"
            edit_text = str(task.get("pending_edit") or "")
            conv, msg, sess = task["conversation_id"], task["message_id"], task["session_id"]
            token = task["access_token"]
            title = _clean((task.get("plan") or {}).get("title")) or _clean(task.get("title"))
            base_prompt = str(task.get("prompt") or "")
            version = int(task.get("plan_version") or 1)
            task["status"] = STATUS_RUNNING
            task["started_ts"] = time.time()
            task.pop("pending_action", None)
            task.pop("pending_edit", None)
            _touch(task)
            self._save_locked()
        backend = OpenAIBackendAPI(access_token=token)
        try:
            if action == "edit":
                backend.deep_research_edit(conv, msg, sess, edit_text, title)
                with self._lock:
                    t = self._tasks.get(task_id)
                    if t:
                        t["prompt"] = f"{base_prompt}\n\n补充/修改要求：{edit_text}" if base_prompt else edit_text
                        t["plan_version"] = version + 1
                        _touch(t)
                        self._save_locked()
            else:
                backend.deep_research_skip_sleep(conv, msg, sess)
        except Exception as exc:
            self._update(task_id, status=STATUS_ERROR, error=f"启动研究失败：{exc}")
            return
        self._poll_until_done(task_id)

    # ---------- background poller ----------
    def _poll_until_done(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            conv, msg, sess = task["conversation_id"], task["message_id"], task["session_id"]
            token = task["access_token"]
            started = task.get("started_ts") or time.time()
            prompt = task.get("prompt", "")
            account_email = task.get("account_email", "")
            base_url = task.get("base_url", "")
        backend = OpenAIBackendAPI(access_token=token)
        deadline = started + MAX_RESEARCH_SECS
        no_start_deadline = started + STALL_NO_START_SECS
        report_message_id = ""
        export_host = msg
        signal_completed_at: float | None = None
        saw_progress = False  # 是否真正开始产出研究消息/活动（用于识别"停在准备研究"的已停止任务）
        try:
            while time.time() < deadline:
                time.sleep(POLL_INTERVAL_SECS)
                # 进度（仅展示用，best-effort）
                try:
                    state = backend.deep_research_get_state(conv, msg, sess)
                    if state.get("num_messages") or state.get("activities"):
                        saw_progress = True
                    self._update(task_id, progress=_clean(state.get("progress")),
                                 activities=state.get("activities") or [],
                                 num_messages=state.get("num_messages", 0),
                                 sources_count=state.get("sources_count", 0))
                except Exception as exc:
                    self._update(task_id, progress=f"(轮询出错重试：{str(exc)[:80]})")
                # 完成判定 + 报告引用（主会话 tool/code 的 widget_state，确认/编辑流都适用）
                host_id, report_id = backend.deep_research_report_ref(conv)
                if report_id:
                    report_message_id, export_host = report_id, (host_id or msg)
                    break
                # 权威实时判定：widget_state.status 出现终止值（user_stopped / 被作废 / 失败）→ 立刻结束，不等超时。
                # 这是「计划长时间未确认被上游作废」唯一可靠的接口信号（既无 final_widget_status_signal、async_status 也为 null）。
                wstatus = backend.deep_research_widget_status(conv)
                if wstatus in DEEP_RESEARCH_STOP_STATUSES:
                    self._update(task_id, chatgpt_url=f"https://chatgpt.com/c/{conv}")
                    raise RuntimeError(f"研究已停止（{wstatus}）：可能是计划长时间未确认被上游作废，"
                                       "或在 ChatGPT 中被停止，未生成报告。请重新发起这次深度研究。")
                # 兜底：主会话权威完成信号（final_widget_status_signal，终态才出现）防卡死
                signal = backend.deep_research_completion_status(conv)
                if signal == "completed":
                    if signal_completed_at is None:
                        signal_completed_at = time.time()  # 宽限期，等报告引用出现
                    elif time.time() - signal_completed_at >= REPORT_GRACE_SECS:
                        break
                elif signal:
                    # 非空且非 completed 的终态信号 = 失败 / 被用户在 ChatGPT 中停止 / 取消
                    self._update(task_id, chatgpt_url=f"https://chatgpt.com/c/{conv}")
                    if signal == "failed":
                        raise RuntimeError("上游报告：深度研究失败")
                    raise RuntimeError(f"研究已在 ChatGPT 中停止或中断（{signal}），未生成报告")
                # 兜底：开跑后迟迟无任何研究消息（一直停在"准备研究"）→ 会话多半已失效，别傻等到 40min。
                # 常见原因：计划在 awaiting 阶段被 pause 冻结太久，上游已把该研究会话自动作废，
                # 此时确认(skip_sleep)虽返回成功却永远不开跑。
                if not saw_progress and time.time() >= no_start_deadline:
                    self._update(task_id, chatgpt_url=f"https://chatgpt.com/c/{conv}")
                    raise RuntimeError("研究迟迟未开始：研究计划可能因长时间未确认已被上游作废（或已在 ChatGPT 中停止），"
                                       "请重新发起这次深度研究。")
            # 收尾再取一次报告引用（宽限期内可能刚出现）
            if not report_message_id:
                host_id, report_id = backend.deep_research_report_ref(conv)
                if report_id:
                    report_message_id, export_host = report_id, (host_id or msg)
            if report_message_id:
                self._update(task_id, status=STATUS_EXPORTING)
                report_md = backend.deep_research_report_markdown(conv)  # 报告正文 markdown（状态页渲染用）
                pdf_bytes, filename = backend.deep_research_export(conv, export_host, sess, report_message_id, "pdf")
                pdf_url = file_storage_service.save(pdf_bytes, filename, base_url=base_url).url
                if not pdf_url:
                    raise RuntimeError("PDF 本地托管失败")
                self._update(task_id, status=STATUS_SUCCESS, pdf_url=pdf_url, pdf_filename=filename,
                             report_message_id=report_message_id, report_markdown=report_md, error="")
                self._log(task_id, prompt, started, "深度研究完成", account_email=account_email, pdf_url=pdf_url)
            elif signal_completed_at is not None:
                # 已完成但实在取不到报告引用（极少见）→ 标记成功 + 给 ChatGPT 查看链接，绝不卡死
                self._update(task_id, status=STATUS_SUCCESS, error="",
                             report_note="研究已完成，但本接口暂未取到可导出的报告，请在 ChatGPT 中查看原报告。",
                             chatgpt_url=f"https://chatgpt.com/c/{conv}")
                self._log(task_id, prompt, started, "深度研究完成（无 PDF）", account_email=account_email)
            else:
                raise RuntimeError(f"深度研究在 {int(MAX_RESEARCH_SECS)}s 内未完成（已超时）")
        except Exception as exc:
            self._update(task_id, status=STATUS_ERROR, error=str(exc) or "deep research failed")
            self._log(task_id, prompt, started, "深度研究失败", status="failed",
                      error=str(exc), account_email=account_email)

    def _maybe_expire_awaiting_locked(self, task: dict[str, Any]) -> bool:
        """awaiting 阶段被 pause 冻结太久，上游会把研究会话自动作废（实测 widget_state.status→user_stopped）；
        超过 AWAITING_EXPIRE_SECS 仍未确认 → 就地标记过期，避免确认进一个已死会话。须持锁调用。返回是否已过期。"""
        if task.get("status") != STATUS_AWAITING:
            return False
        created = task.get("created_ts") or _timestamp(task.get("created_at"))
        if not created or time.time() - created < AWAITING_EXPIRE_SECS:
            return False
        task["status"] = STATUS_ERROR
        task["error"] = (f"研究计划已过期：超过 {int(AWAITING_EXPIRE_SECS // 60)} 分钟未确认，"
                         "上游可能已作废该会话，请重新发起这次深度研究。")
        if task.get("conversation_id"):
            task["chatgpt_url"] = f"https://chatgpt.com/c/{task['conversation_id']}"
        _touch(task)
        self._save_locked()
        return True

    # ---------- internals ----------
    def _update(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.update(updates)
            _touch(task)
            self._save_locked()

    def _log(self, task_id: str, prompt: str, started: float, summary: str, *,
             status: str = "success", error: str = "", account_email: str = "", pdf_url: str = "") -> None:
        detail = {
            "endpoint": "/v1/deep-research", "task_id": task_id,
            "request_text": prompt[:200],
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(), "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if error:
            detail["error"] = error
        if account_email:
            detail["account_email"] = account_email
        if pdf_url:
            detail["pdf_url"] = pdf_url
        try:
            log_service.add(LOG_TYPE_CALL, summary, detail)
        except Exception:
            pass

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        items = raw.get("tasks") if isinstance(raw, dict) else raw
        tasks: dict[str, dict[str, Any]] = {}
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    tasks[str(item["id"])] = item
        return tasks

    def _save_locked(self) -> None:
        # 持久化时剥离 access_token（敏感、且会轮换）
        items = []
        for task in sorted(self._tasks.values(), key=lambda x: str(x.get("updated_at") or ""), reverse=True):
            items.append({k: v for k, v in task.items() if k != "access_token"})
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            # 数据目录不可写（如本地非 root）不应让服务崩溃：内存态仍可用，仅丢失持久化。
            logger.warning({"event": "deep_research_save_failed", "path": str(self.path), "error": repr(exc)[:200]})

    def _recover_unfinished_locked(self) -> bool:
        # 重启后无法续跑（access_token 未持久化）：未完成任务标记失败
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = STATUS_ERROR
                task["error"] = "服务已重启，未完成的深度研究任务已中断"
                task["updated_at"] = _now_iso()
                changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        cutoff = time.time() - RETENTION_DAYS * 86400
        removed = [tid for tid, t in self._tasks.items()
                   if t.get("status") in TERMINAL_STATUSES and _timestamp(t.get("updated_at")) < cutoff]
        for tid in removed:
            self._tasks.pop(tid, None)
        return bool(removed)


deep_research_service = DeepResearchService(DATA_DIR / "deep_research_tasks.json")
