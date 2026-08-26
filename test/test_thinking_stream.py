from __future__ import annotations

import json
import unittest

from services.openai_backend_api import OpenAIBackendAPI


class FakeResponse:
    """最小 SSE 响应桩：只需 status_code / iter_lines / close。"""

    def __init__(self, payloads: list[str]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._payloads = payloads
        self.closed = False

    def iter_lines(self):
        for payload in self._payloads:
            yield f"data: {payload}".encode("utf-8")
            yield b""

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None, stream=None):  # noqa: A002
        self.calls.append({"url": url, "timeout": timeout, "stream": stream})
        return self.response


def make_backend(payloads: list[str]) -> tuple[OpenAIBackendAPI, FakeResponse]:
    """构造一个不触网的 backend：只装配 _stream_thinking_turn 用到的字段。"""
    backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    backend.base_url = "https://chatgpt.com"
    response = FakeResponse(payloads)
    backend.session = FakeSession(response)
    backend._conversation_headers = lambda path, requirements: {}
    return backend, response


def message_event(text: str) -> str:
    return json.dumps({"message": {"id": "m1", "author": {"role": "assistant"},
                                   "content": {"content_type": "text", "parts": [text]}}})


class ThinkingStreamTests(unittest.TestCase):
    """思考模型链路：内联优先、stream_handoff 兜底。

    回归背景：上游 2026-08 起 gpt-5-5-thinking / gpt-5-6-thinking 不再发 stream_handoff，
    改为直接内联流式返回。旧实现只找 handoff、其余事件全丢弃、找不到就抛
    `thinking turn did not hand off to a websocket stream`，把完整答案丢掉。
    """

    def test_inline_stream_is_passed_through(self):
        """没有 handoff 时：内联事件原样吐出（含 [DONE]），不再抛错。"""
        payloads = [
            json.dumps({"type": "resume_conversation_token", "conversation_id": "c1"}),
            message_event("Hi!"),
            json.dumps({"type": "message_stream_complete", "conversation_id": "c1"}),
            "[DONE]",
        ]
        backend, response = make_backend(payloads)
        backend._stream_ws_topic = lambda topic: iter(["should-not-be-used"])

        got = list(backend._stream_thinking_turn("/backend-api/conversation", object(), {}))

        self.assertEqual(got, payloads)
        self.assertTrue(response.closed)

    def test_non_json_payload_is_passed_through(self):
        """无法解析的 payload 不吞掉，交给下游解析器。"""
        backend, _ = make_backend(["not-json", "[DONE]"])
        backend._stream_ws_topic = lambda topic: iter([])

        self.assertEqual(list(backend._stream_thinking_turn("/p", object(), {})), ["not-json", "[DONE]"])

    def test_handoff_switches_to_websocket(self):
        """收到 handoff 时：切到 WS topic 续流，且不把内联的 [DONE] 提前吐出去。"""
        payloads = [
            json.dumps({"type": "resume_conversation_token", "conversation_id": "c1"}),
            json.dumps({"type": "stream_handoff", "turn_exchange_id": "abc-123"}),
            "[DONE]",
        ]
        backend, response = make_backend(payloads)
        topics: list[str] = []

        def fake_ws(topic: str):
            topics.append(topic)
            yield message_event("from-ws")
            yield "[DONE]"

        backend._stream_ws_topic = fake_ws

        got = list(backend._stream_thinking_turn("/p", object(), {}))

        self.assertEqual(topics, ["conversation-turn-abc-123"])
        self.assertEqual(got[0], payloads[0])
        self.assertEqual(got[1:], [message_event("from-ws"), "[DONE]"])
        # 内联的 [DONE] 属于 handoff 轮次的空壳流，不能混进结果里提前结束解析。
        self.assertEqual(got.count("[DONE]"), 1)
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
