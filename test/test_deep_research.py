"""Deep research two-turn flow — chat + responses, stream + non-stream. Mocked service (no upstream)."""
import unittest

from services.protocol import deep_research_protocol as dp
from services.protocol import openai_v1_chat_complete as cc
from services.protocol import openai_v1_response as rp

TID = "a" * 32


class FakeService:
    def __init__(self):
        self.tasks = {}
        self.created = []
        self.confirmed = []
        self.edited = []

    def create_research(self, identity, *, prompt, base_url="", deep_research_version="standard"):
        self.created.append(prompt)
        task = {
            "id": TID, "status": "awaiting_confirmation", "title": "时序数据库对比", "prompt": prompt,
            "plan": {"title": "时序数据库对比", "steps": [{"text": "收集文档"}, {"text": "对比性能"}]},
            "status_url": f"{base_url}/api/deep-research/{TID}", "num_messages": 0, "sources_count": 0,
        }
        self.tasks[TID] = task
        return dict(task)

    def edit_research(self, identity, task_id, edit_text, base_url=""):
        self.edited.append((task_id, edit_text))
        task = self.tasks[task_id]
        task["prompt"] = f"{task.get('prompt','')}\n\n补充/修改要求：{edit_text}"
        task["plan_version"] = int(task.get("plan_version", 1)) + 1
        task["status"] = "running"  # 官方：提交修改后自动开始
        task["_polls"] = 0
        return dict(task)

    def confirm_research(self, identity, task_id):
        self.confirmed.append(task_id)
        self.tasks[task_id]["status"] = "running"
        self.tasks[task_id]["_polls"] = 0
        return dict(self.tasks[task_id])

    def get_research(self, task_id, identity=None):
        task = self.tasks.get(task_id)
        if not task:
            return None
        task["_polls"] = task.get("_polls", 0) + 1
        if task.get("status") in ("running", "exporting") and task["_polls"] >= 2:
            task.update(status="success", pdf_url="https://transfer/r.pdf", pdf_filename="报告.pdf",
                        progress="完成", report_markdown="# 研究报告\n\n这是正文内容。")
        elif task.get("status") in ("running", "exporting"):
            task.update(progress=f"step {task['_polls']}", num_messages=task["_polls"] * 5, sources_count=task["_polls"] * 3)
        return dict(task)


def _collect_chat_text(result):
    if isinstance(result, dict):
        return result["choices"][0]["message"]["content"]
    return "".join(
        str((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
        for c in result
    )


def _collect_resp_text(result):
    events = result if not isinstance(result, dict) else [{"type": "response.completed", "response": result}]
    parts = []
    for e in events:
        if e.get("type") == "response.output_text.delta":
            parts.append(e.get("delta") or "")
    if parts:
        return "".join(parts)
    # non-stream dict: read output items
    resp = result if isinstance(result, dict) else {}
    for item in resp.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                parts.append(part.get("text") or "")
    return "".join(parts)


class DeepResearchFlowTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeService()
        dp.deep_research_service = self.fake
        self._orig_poll = dp._PROGRESS_POLL_SECS
        dp._PROGRESS_POLL_SECS = 0.001

    def tearDown(self):
        dp._PROGRESS_POLL_SECS = self._orig_poll

    def _history_with_plan(self):
        return [
            {"role": "user", "content": "研究时序数据库"},
            {"role": "assistant", "content": f"## 计划\n1. a\n2. b\n{dp._marker(TID)}"},
            {"role": "user", "content": "开始"},
        ]

    # ---- chat completions ----
    def test_chat_turn1_returns_plan_with_marker(self):
        body = {"model": "deep-research", "messages": [{"role": "user", "content": "研究时序数据库"}]}
        text = _collect_chat_text(cc.deep_research_chat_handle(body, {"id": "u"}, "http://h"))
        self.assertIn("研究计划", text)
        self.assertIn(TID, text)
        self.assertEqual(self.fake.created, ["研究时序数据库"])
        self.assertEqual(self.fake.confirmed, [])

    def test_chat_turn1_stream(self):
        body = {"model": "deep-research", "stream": True, "messages": [{"role": "user", "content": "Q"}]}
        text = _collect_chat_text(cc.deep_research_chat_handle(body, {"id": "u"}, "http://h"))
        self.assertIn("深度研究计划", text)
        self.assertIn(TID, text)

    def test_chat_turn2_stream_confirms_and_streams_pdf(self):
        self.fake.tasks[TID] = {"id": TID, "status": "awaiting_confirmation", "plan": {}, "status_url": "http://h/x"}
        body = {"model": "deep-research", "stream": True, "messages": self._history_with_plan()}
        text = _collect_chat_text(cc.deep_research_chat_handle(body, {"id": "u"}, "http://h"))
        self.assertEqual(self.fake.confirmed, [TID])
        self.assertIn("https://transfer/r.pdf", text)
        self.assertIn("这是正文内容", text)  # 完整 markdown 报告也通过 SSE 内联返回

    def test_trivial_prompt_shortcircuits(self):
        for prompt, expect in [("ping", "pong"), ("hi", "hi"), ("Hi!", "hi"), ("你好", "你好"), ("test", "ok"), ("PING?", "pong")]:
            self.fake.created = []
            body = {"model": "deep-research", "messages": [{"role": "user", "content": prompt}]}
            text = _collect_chat_text(cc.deep_research_chat_handle(body, {"id": "u"}, "http://h"))
            self.assertEqual(self.fake.created, [], f"{prompt!r} 不应创建深研任务")
            self.assertIn(expect, text, f"{prompt!r} 应秒回 {expect!r}")

    def test_real_prompt_not_shortcircuited(self):
        body = {"model": "deep-research", "messages": [{"role": "user", "content": "深入研究 vLLM 架构"}]}
        cc.deep_research_chat_handle(body, {"id": "u"}, "http://h")
        self.assertEqual(len(self.fake.created), 1)  # 正常进深研、建任务

    def test_is_confirmation(self):
        for yes in ("开始", "开始吧！", "确认", "可以", "go", "OK", "yes", "looks good", "确定。"):
            self.assertTrue(dp.is_confirmation(yes), yes)
        for no in ("只看近一年", "聚焦推理框架", "可以，但只要开源的", "确认一下范围", "再加一步对比成本"):
            self.assertFalse(dp.is_confirmation(no), no)

    def test_chat_turn2_edit_inplace_autostarts(self):
        self.fake.tasks[TID] = {"id": TID, "status": "awaiting_confirmation", "prompt": "研究时序数据库",
                                "title": "时序数据库对比", "plan_version": 1, "plan": {}, "status_url": "http://h/x"}
        body = {"model": "deep-research", "stream": True, "messages": [
            {"role": "user", "content": "研究时序数据库"},
            {"role": "assistant", "content": f"计划 {dp._marker(TID)}"},
            {"role": "user", "content": "只看近一年的数据"},
        ]}
        text = _collect_chat_text(cc.deep_research_chat_handle(body, {"id": "u"}, "http://h"))
        self.assertEqual(self.fake.created, [])                          # 没有新建 task
        self.assertEqual(self.fake.edited, [(TID, "只看近一年的数据")])   # 走原地编辑
        self.assertEqual(self.fake.tasks[TID]["plan_version"], 2)        # 同一 task，版本递增
        self.assertIn("https://transfer/r.pdf", text)                   # 提交修改→自动开始→跑完→PDF

    def test_chat_turn2_nonstream_returns_status_immediately(self):
        self.fake.tasks[TID] = {"id": TID, "status": "awaiting_confirmation", "plan": {}, "status_url": "http://h/x"}
        body = {"model": "deep-research", "messages": self._history_with_plan()}
        result = cc.deep_research_chat_handle(body, {"id": "u"}, "http://h")
        self.assertIsInstance(result, dict)
        self.assertEqual(self.fake.confirmed, [TID])
        self.assertIn("http://h/x", _collect_chat_text(result))

    # ---- responses ----
    def test_resp_turn1_nonstream_plan(self):
        body = {"model": "deep-research", "input": "研究时序数据库"}
        result = rp.deep_research_response_handle(body, {"id": "u"}, "http://h")
        self.assertIsInstance(result, dict)
        self.assertIn(TID, _collect_resp_text(result))
        self.assertEqual(self.fake.created, ["研究时序数据库"])

    def test_resp_turn2_stream_pdf(self):
        self.fake.tasks[TID] = {"id": TID, "status": "awaiting_confirmation", "plan": {}, "status_url": "http://h/x"}
        body = {"model": "deep-research", "stream": True, "input": [
            {"role": "user", "content": "研究"},
            {"role": "assistant", "content": [{"type": "output_text", "text": f"plan {dp._marker(TID)}"}]},
            {"role": "user", "content": "开始"},
        ]}
        events = list(rp.deep_research_response_handle(body, {"id": "u"}, "http://h"))
        self.assertEqual(self.fake.confirmed, [TID])
        self.assertTrue(any(e.get("type") == "response.completed" for e in events))
        self.assertIn("https://transfer/r.pdf", _collect_resp_text(events))


class QueueSerializationTest(unittest.TestCase):
    """真实 DeepResearchService：验证单 worker 串行——同时只跑一个，其余 queued。"""

    def setUp(self):
        import threading
        from pathlib import Path
        import services.deep_research_service as svc_mod
        self.svc_mod = svc_mod
        self._orig_backend = svc_mod.OpenAIBackendAPI
        self._orig_poll = svc_mod.POLL_INTERVAL_SECS
        svc_mod.POLL_INTERVAL_SECS = 0.02
        # 本地文件托管打桩，避免测试真的落盘
        from types import SimpleNamespace
        self._orig_file_store = svc_mod.file_storage_service
        svc_mod.file_storage_service = SimpleNamespace(
            save=lambda data, filename, base_url="": SimpleNamespace(rel="x", url="http://t/r.pdf", size=len(data)))
        acc = svc_mod.account_service
        self._orig_acc = (acc.get_text_access_token, acc.get_account, acc.refresh_access_token)
        acc.get_text_access_token = lambda **k: "tok"
        acc.get_account = lambda t: {"email": "x@x"}
        acc.refresh_access_token = lambda t, event="": t

        self.gate = threading.Event()
        self.first_started = threading.Event()
        started = self.started = []
        gate, first_started = self.gate, self.first_started

        class FakeBackend:
            def __init__(self, access_token=""):
                self.access_token = access_token
            def start_deep_research(self, prompt, deep_research_version="standard"):
                return {"conversation_id": "c" + prompt, "message_id": "m", "session_id": "s",
                        "plan": {"title": "t", "steps": []}}
            def deep_research_pause(self, *a): pass
            def deep_research_skip_sleep(self, conv, msg, sess):
                started.append(conv)
                first_started.set()
                gate.wait(5)  # 阻塞，模拟研究在跑
            def deep_research_report_ref(self, conv): return ("h", "r")
            def deep_research_report_markdown(self, conv): return "# 报告"
            def deep_research_export(self, *a): return (b"%PDF-1.7", "r.pdf")
            def deep_research_completion_status(self, conv): return ""
            def deep_research_get_state(self, *a):
                return {"done": True, "report_message_id": "r", "progress": "", "activities": [],
                        "num_messages": 1, "sources_count": 1}
        svc_mod.OpenAIBackendAPI = FakeBackend
        self.svc = svc_mod.DeepResearchService(Path("/tmp/dr_queue_test.json"))
        self.svc._save_locked = lambda: None

    def tearDown(self):
        self.gate.set()
        self.svc_mod.OpenAIBackendAPI = self._orig_backend
        self.svc_mod.POLL_INTERVAL_SECS = self._orig_poll
        self.svc_mod.file_storage_service = self._orig_file_store
        acc = self.svc_mod.account_service
        acc.get_text_access_token, acc.get_account, acc.refresh_access_token = self._orig_acc

    def test_only_one_runs_others_queued(self):
        import time
        ident = {"id": "u"}
        t1 = self.svc.create_research(ident, prompt="A")["id"]
        t2 = self.svc.create_research(ident, prompt="B")["id"]
        self.svc.confirm_research(ident, t1)
        self.svc.confirm_research(ident, t2)
        self.assertTrue(self.first_started.wait(3), "worker 应已开始处理 t1")
        # worker 卡在 t1 的 skip_sleep → t1 running，t2 必须 queued
        self.assertEqual(self.svc.get_research(t1)["status"], "running")
        self.assertEqual(self.svc.get_research(t2)["status"], "queued")
        self.assertEqual(self.svc.get_research(t2)["queue_position"], 2)
        self.assertEqual(self.started, ["cA"])  # 同时只起了一个
        self.gate.set()  # 放行 t1 → 完成后 worker 取 t2
        for _ in range(200):
            if all(self.svc.get_research(t)["status"] == "success" for t in (t1, t2)):
                break
            time.sleep(0.02)
        self.assertEqual(self.svc.get_research(t1)["status"], "success")
        self.assertEqual(self.svc.get_research(t2)["status"], "success")
        self.assertEqual(self.started, ["cA", "cB"])  # 严格串行、按入队顺序


if __name__ == "__main__":
    unittest.main()
