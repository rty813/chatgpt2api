from __future__ import annotations

import json
import unittest
from unittest import mock

from services.protocol import openai_v1_chat_complete
from services.protocol.conversation import iter_conversation_payloads, sanitize_output_text


def message(mid: str, text: str, *, channel: str = "final") -> dict:
    return {"message": {"id": mid, "author": {"role": "assistant"}, "channel": channel,
                        "content": {"content_type": "text", "parts": [text]}}}


def stream(events: list[dict | str], *, thinking: bool) -> list[dict]:
    payloads = [json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event for event in events]
    return list(iter_conversation_payloads(iter(payloads), thinking=thinking))


class StreamTextIntegrityTests(unittest.TestCase):
    def assert_content(self, events: list[dict | str], expected: str, *, thinking: bool) -> list[dict]:
        output = stream(events, thinking=thinking)
        self.assertEqual("".join(event.get("delta", "") for event in output), expected)
        self.assertEqual(output[-1]["text"], expected)
        self.assertEqual(output[-1]["type"], "conversation.done")
        return output

    def test_sanitizer_preserves_code_and_punctuation_whitespace(self):
        text = ('```bash\n# 启动\n./agent "分析项目"\n```\n\n'
                '```go\nfunc RunAgent() {\n    messages := BuildInitial()\n}\n```\n'
                '<CodeBlock language="mermaid" model_previewable>flowchart TD\n A --> B</CodeBlock>\n'
                '保留原文 : 说明\n.下一行\n')
        self.assertEqual(sanitize_output_text(text), text)

    def test_character_stream_never_resends_the_introduction(self):
        intro = '建议使用单文件启动器。\n\n'
        body = ('## 总体架构\n\n```bash\n# 启动\n./agent "分析项目"\n```\n\n'
                '```go\nfunc RunAgent() {\n    messages := BuildInitial()\n}\n```\n')
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                events = [message("a", (intro + body)[:i]) for i in range(1, len(intro + body) + 1)]
                self.assert_content(events + ["[DONE]"], intro + body, thinking=thinking)

    def test_interleaved_messages_and_replayed_snapshots(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                self.assert_content([
                    message("intro", "引言\n\n"),
                    message("answer", "## 总体"),
                    message("intro", "引言\n\n"),
                    message("answer", "## 总体架构\n"),
                    message("answer", "## 总体"),  # 重连时的过期快照。
                    {"p": "/message/content/parts/0", "o": "append", "v": "正文"},
                    message("intro", "引言\n\n"),
                    message("answer", "## 总体架构\n正文"),
                    "[DONE]",
                ], "引言\n\n## 总体架构\n正文", thinking=thinking)

    def test_reasoning_replayed_snapshot_does_not_repeat(self):
        output = self.assert_content([
            message("r", "检查", channel="analysis"),
            message("r", "检查方案", channel="analysis"),
            message("r", "检查", channel="analysis"),
            {"p": "/message/content/parts/0", "o": "append", "v": "完成"},
            message("a", "结论"), "[DONE]",
        ], "结论", thinking=True)
        self.assertEqual("".join(event.get("reasoning_delta", "") for event in output), "检查方案完成")
        self.assertEqual(output[-1]["reasoning"], "检查方案完成")

    def test_annotation_boundaries_are_stream_safe(self):
        sources = [
            ("说明 \ue200cite\ue202turn0search0\ue201.", "说明."),
            ("说明 \ue200entity\ue202名称\ue201.", "说明 名称."),
            ("说明 \ue200cite\ue202turn0search0\ue201 后文", "说明  后文"),
            ("说明\n\ue200cite\ue202turn0search0\ue201.", "说明\n."),
        ]
        for source, expected in sources:
            for thinking in (False, True):
                with self.subTest(source=source, thinking=thinking):
                    events = [message("a", source[:i]) for i in range(1, len(source) + 1)]
                    self.assert_content(events + ["[DONE]"], expected, thinking=thinking)

    def test_trailing_whitespace_is_flushed_at_done_and_eof(self):
        for thinking in (False, True):
            for suffix in ("  ", "\n\t", "\n\n"):
                for ending in ([], ["[DONE]"]):
                    with self.subTest(thinking=thinking, suffix=suffix, ending=ending):
                        self.assert_content([message("a", "正文"), message("a", "正文" + suffix)] + ending,
                                            "正文" + suffix, thinking=thinking)

    def test_legacy_patches_without_message_ids_remain_supported(self):
        self.assert_content([
            {"p": "/message/content/parts/0", "o": "append", "v": "# 启动\n"},
            {"p": "/message/content/parts/0", "o": "append", "v": "./agent\n"},
            "[DONE]",
        ], "# 启动\n./agent\n", thinking=False)

    def test_revised_thinking_snapshot_is_buffered_and_emitted_once(self):
        snapshots = [
            "有，而且这已经是计算机视觉和三",
            "有，而且现在已经发展到相当实用的阶段了。",
            "有，而且这几年发展很快。",
            "现在有，而且这几年发展很快。",
        ]
        events = stream([*(message("a", snapshot) for snapshot in snapshots), "[DONE]"], thinking=True)
        self.assertEqual("".join(event.get("delta", "") for event in events), snapshots[-1])
        self.assertEqual(events[-1]["text"], snapshots[-1])

    def test_merged_snapshot_does_not_repeat_its_source_fragments(self):
        # 真实上游同时发送隐藏片段和 is_merged_message 汇总快照。
        intro = message("intro", "引言\n\n")
        intro["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
        body = message("body", "## 正文\n完成。")
        body["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
        merged = message("merged", "引言\n\n## 正文\n完成。")
        merged["message"]["metadata"] = {
            "is_merged_message": True, "source_message_ids": ["intro", "body"],
        }
        for events in ([intro, body, merged], [intro, merged, body], [merged, intro, body]):
            with self.subTest(order=[event["message"]["id"] for event in events]):
                self.assert_content([*events, "[DONE]"], "引言\n\n## 正文\n完成。", thinking=True)

    def test_merged_sources_are_recognized_without_fragment_metadata(self):
        merged = message("merged", "改写后的完整正文")
        merged["message"]["metadata"] = {"is_merged_message": True, "source_message_ids": ["a", "b"]}
        self.assert_content([
            message("a", "早期引言"), message("b", "早期正文"), merged, "[DONE]",
        ], "改写后的完整正文", thinking=True)

    def test_fragments_remain_when_merged_snapshot_is_missing_or_empty(self):
        intro = message("intro", "引言\n\n")
        intro["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
        body = message("body", "正文")
        body["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
        for extra in ([], [message("merged", "")]):
            with self.subTest(extra=extra):
                self.assert_content([intro, body, *extra, "[DONE]"], "引言\n\n正文", thinking=True)

    def test_unmarked_fragment_fallback_uses_the_complete_merged_snapshot(self):
        merged = message("merged", "完整正文", channel="")
        fragment = message("fragment", "正文", channel="")
        fragment["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
        self.assert_content([merged, fragment, "[DONE]"], "完整正文", thinking=True)

    def test_unrelated_identical_messages_are_preserved(self):
        self.assert_content([message("a", "重复示例\n"), message("b", "重复示例\n"), "[DONE]"],
                            "重复示例\n重复示例\n", thinking=True)

    def test_revised_non_thinking_snapshot_still_fails_safely(self):
        with self.assertRaisesRegex(RuntimeError, "append-only"):
            stream([message("a", "方案 A"), message("a", "方案 B"), "[DONE]"], thinking=False)

    def test_unconfirmed_thinking_snapshots_are_not_emitted_before_upstream_done(self):
        payloads = iter([
            json.dumps(message("a", "初始快照"), ensure_ascii=False),
            json.dumps(message("a", "修正后的完整答案"), ensure_ascii=False),
            "[DONE]",
        ])
        output = iter_conversation_payloads(payloads, thinking=True)
        first_event = next(output)
        self.assertEqual(first_event["type"], "conversation.event")
        self.assertEqual(first_event["text"], "")
        final = next(event for event in output if event["type"] == "conversation.delta")
        self.assertEqual(final["delta"], "修正后的完整答案")

    def test_reasoning_and_canonical_merged_answer_stream_before_done(self):
        consumed = []
        def payloads():
            reasoning = message("r", "正在查询", channel="analysis")
            fragment = message("fragment", "可改写的早期预告")
            fragment["message"]["metadata"] = {"is_message_fragment": True, "merged_message_id": "merged"}
            merged = message("merged", "正确正文")
            merged["message"]["metadata"] = {"is_merged_message": True, "source_message_ids": ["fragment"]}
            for event in [reasoning, fragment, merged, message("merged", "正确正文，继续生成。"), "[DONE]"]:
                consumed.append(event)
                yield json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event
        output = iter_conversation_payloads(payloads(), thinking=True)
        first = next(output)
        self.assertEqual(first["reasoning_delta"], "正在查询")
        self.assertEqual(len(consumed), 1)
        first_body = next(event for event in output if event.get("delta"))
        self.assertEqual(first_body["delta"], "正确正文")
        self.assertEqual(len(consumed), 3)
        second_body = next(event for event in output if event.get("delta"))
        self.assertEqual(second_body["delta"], "，继续生成。")
        self.assertEqual(len(consumed), 4)
        self.assertEqual(list(output)[-1]["text"], "正确正文，继续生成。")

    def test_openai_chunks_preserve_exact_code(self):
        expected = "# 启动\n./agent\n\nmessages := BuildInitial()\n"
        events = [message("a", expected[:i]) for i in range(1, len(expected) + 1)] + ["[DONE]"]

        def upstream_text_events(_backend, _request):
            for event in stream(events, thinking=True):
                if event["type"] == "conversation.delta":
                    yield {"content": event["delta"], "reasoning": event["reasoning_delta"]}

        with mock.patch("services.protocol.openai_v1_chat_complete.stream_text_events",
                        side_effect=upstream_text_events):
            chunks = list(openai_v1_chat_complete.stream_text_chat_completion(
                object(), [{"role": "user", "content": "test"}], "gpt-6-thinking", "extended",
                emit_reasoning=True,
            ))
        self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), expected)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
