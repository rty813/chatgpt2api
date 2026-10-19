from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from services.protocol import openai_v1_chat_complete
from services.protocol.conversation import iter_conversation_payloads


def message(mid: str, text: str = "", *, channel: str | None = None,
            metadata: object = None, recipient: str | None = None,
            content_type: str = "text", role: str = "assistant") -> dict:
    content = {"content_type": content_type, "parts": [text]}
    if content_type == "thoughts":
        content = {"content_type": "thoughts", "thoughts": [{"content": text}]}
    result = {"id": mid, "author": {"role": role}, "content": content}
    if channel is not None:
        result["channel"] = channel
    if metadata is not None:
        result["metadata"] = metadata
    if recipient is not None:
        result["recipient"] = recipient
    return {"v": {"message": result}}


def marker(mid: str, kind: str = "final_channel_token") -> dict:
    return {"v": {"type": "message_marker", "message_id": mid, "marker": kind}}


class ThinkingChannelClassificationTests(unittest.TestCase):
    def assert_stream(self, events: list[dict | str], answer: str, reasoning: str = "") -> list[dict]:
        payloads = [json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event for event in events]
        if not payloads or payloads[-1] != "[DONE]":
            payloads.append("[DONE]")
        output = list(iter_conversation_payloads(iter(payloads), thinking=True))
        deltas = [event for event in output if event["type"] == "conversation.delta"]
        self.assertEqual("".join(event.get("delta", "") for event in deltas), answer)
        self.assertEqual("".join(event.get("reasoning_delta", "") for event in deltas), reasoning)
        # 同时检查最终快照，避免流式累计与最终结果不一致。
        self.assertEqual(output[-1]["text"], answer)
        self.assertEqual(output[-1]["reasoning"], reasoning)
        return deltas

    def test_delayed_marker_does_not_emit_answer_as_reasoning(self):
        output = self.assert_stream([message("a", "最终结论"), marker("a")], "最终结论")
        self.assertEqual(len(output), 1)
        self.assertNotIn("raw", output[0])

    def test_missing_marker_falls_back_at_done_without_reasoning_copy(self):
        output = self.assert_stream([message("a", "最终结论")], "最终结论")
        self.assertEqual(len(output), 1)
        self.assertNotIn("raw", output[0])

    def test_late_marker_after_text_patches(self):
        self.assert_stream([
            message("a"),
            {"v": "这是"},
            {"p": "/message/content/parts/0", "o": "append", "v": "完整答案"},
            marker("a"),
            {"v": "。"},
        ], "这是完整答案。")

    def test_late_markers_for_multiple_messages(self):
        self.assert_stream([
            message("a", "第一段"),
            message("b", "第二段"),
            marker("a"),
            marker("b"),
        ], "第一段第二段")

    def test_marker_before_message_remains_supported(self):
        self.assert_stream([marker("a"), message("a", "最终结论")], "最终结论")

    def test_final_channel_keeps_the_latest_snapshot_until_done(self):
        for fields in ({"channel": "final"}, {"metadata": {"channel": "final"}}):
            with self.subTest(fields=fields):
                events = [
                    message("a", "最终", **fields),
                    {"p": "/message/content/parts/0", "o": "append", "v": "结论"},
                ]
                payloads = [json.dumps(event, ensure_ascii=False) for event in events]
                stream = iter_conversation_payloads(iter(payloads + ["[DONE]"]), thinking=True)
                interim = list(stream)
                self.assertFalse(any(event["type"] == "conversation.delta" for event in interim[:-2]))
                deltas = [event for event in interim if event["type"] == "conversation.delta"]
                self.assertEqual([event["delta"] for event in deltas], ["最终结论"])
                self.assertEqual(interim[-1]["text"], "最终结论")

    def test_real_thoughts_remain_separate_from_answer(self):
        self.assert_stream([
            message("r", "检查方案", content_type="thoughts"),
            message("a", "最终结论"),
            marker("a"),
        ], "最终结论", "检查方案")

    def test_reasoning_status_text_accepts_parts_patches(self):
        self.assert_stream([
            message("r", "检查", metadata={"reasoning_status": "is_reasoning"}),
            {"o": "patch", "v": [
                {"p": "/message/content/parts/0", "o": "append", "v": "方案"},
            ]},
            marker("r", "last_token"),
            message("a", "最终结论", channel="final"),
        ], "最终结论", "检查方案")

    def test_analysis_text_is_not_promoted_by_last_token(self):
        for fields in ({"channel": "analysis"}, {"metadata": {"channel": "analysis"}}):
            with self.subTest(fields=fields):
                self.assert_stream([
                    message("r", "检查", **fields),
                    {"p": "/message/content/parts/0", "o": "append", "v": "方案"},
                    marker("r", "last_token"),
                    message("a", "最终结论", channel="final"),
                ], "最终结论", "检查方案")

    def test_commentary_text_remains_reasoning(self):
        self.assert_stream([
            message("r", "正在查询", channel="commentary"),
            marker("r", "last_token"),
            message("a", "最终结论", channel="final"),
        ], "最终结论", "正在查询")

    def test_tool_recipient_text_is_not_final_answer(self):
        self.assert_stream([
            message("r", "工具参数", recipient="web.run"),
            marker("r", "last_token"),
            message("a", "最终结论", recipient="all", channel="final"),
        ], "最终结论", "工具参数")

    def test_tool_reasoning_text_accepts_parts_patches(self):
        self.assert_stream([
            message("r", "查询", role="tool", metadata={"reasoning_status": "is_reasoning"}),
            {"p": "/message/content/parts/0", "o": "append", "v": "结果"},
            message("a", "最终结论", channel="final"),
        ], "最终结论", "查询结果")

    def test_unknown_text_does_not_leak_into_reasoning(self):
        self.assert_stream([
            message("pending", "未确定用途的文本"),
            message("a", "最终结论", channel="final"),
        ], "最终结论")

    def test_non_object_metadata_is_tolerated(self):
        self.assert_stream([message("a", "最终结论", metadata="unexpected")], "最终结论")

    @staticmethod
    def captured_fixture() -> dict:
        fixture = Path(__file__).with_name("fixtures") / "chatgpt_thinking_architecture.json"
        return json.loads(fixture.read_text(encoding="utf-8"))

    @staticmethod
    def delay_final_marker(events: list[dict | str]) -> list[dict | str]:
        # 只改变标记到达顺序，保留真实正文快照及 channel=final。
        events = list(events)
        index = next(i for i, event in enumerate(events)
                     if isinstance(event, dict) and event.get("type") == "message_marker"
                     and event.get("marker") == "final_channel_token")
        final_marker = events.pop(index)
        index = next(i for i, event in enumerate(events)
                     if isinstance(event, dict) and (event.get("message") or {}).get("channel") == "final")
        events.insert(index + 1, final_marker)
        return events

    def test_captured_upstream_stream_preserves_both_channels(self):
        fixture = self.captured_fixture()
        self.assert_stream(fixture["events"], fixture["expected_content"], fixture["expected_reasoning"])

    def test_captured_stream_with_delayed_final_marker(self):
        fixture = self.captured_fixture()
        self.assert_stream(self.delay_final_marker(fixture["events"]),
                           fixture["expected_content"], fixture["expected_reasoning"])

    @staticmethod
    def remove_final_markers(events: list[dict | str]) -> list[dict | str]:
        return [event for event in events
                if not (isinstance(event, dict) and event.get("type") == "message_marker"
                        and event.get("marker") in ("final_channel_token", "last_token"))]

    def test_captured_stream_without_final_markers(self):
        fixture = self.captured_fixture()
        events = self.remove_final_markers(fixture["events"])
        self.assert_stream(events, fixture["expected_content"], fixture["expected_reasoning"])

    def test_captured_stream_without_channel_or_final_markers(self):
        fixture = self.captured_fixture()
        events = self.remove_final_markers(fixture["events"])
        # 模拟旧格式正文也没有 channel 字段，必须等 DONE 后兜底输出。
        for event in events:
            if isinstance(event, dict):
                message = event.get("message") or {}
                if message.get("channel") == "final":
                    message.pop("channel")
        deltas = self.assert_stream(events, fixture["expected_content"], fixture["expected_reasoning"])
        self.assertNotIn("raw", deltas[-1])

    def test_captured_stream_through_openai_chat_completion(self):
        fixture = self.captured_fixture()
        scenarios = {
            "original": fixture["events"],
            "delayed_marker": self.delay_final_marker(fixture["events"]),
            "missing_markers": self.remove_final_markers(fixture["events"]),
        }
        for name, events in scenarios.items():
            with self.subTest(scenario=name):

                def upstream_text_events(_backend, _request):
                    payloads = [json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event
                                for event in events]
                    for event in iter_conversation_payloads(iter(payloads), thinking=True):
                        if event["type"] == "conversation.delta":
                            yield {"content": event["delta"], "reasoning": event["reasoning_delta"]}

                with mock.patch("services.protocol.openai_v1_chat_complete.stream_text_events",
                                side_effect=upstream_text_events):
                    chunks = list(openai_v1_chat_complete.stream_text_chat_completion(
                        object(), [{"role": "user", "content": "test"}],
                        "gpt-5-5-thinking", "extended", emit_reasoning=True,
                    ))
                deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
                answer = "".join(delta.get("content", "") for delta in deltas)
                reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
                self.assertEqual(answer, fixture["expected_content"])
                self.assertEqual(reasoning, fixture["expected_reasoning"])
                self.assertNotIn("<think>", answer + reasoning)
                self.assertNotIn("</think>", answer + reasoning)
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
