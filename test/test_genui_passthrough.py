from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from services.protocol import conversation, openai_v1_chat_complete


SOURCE = (Path(__file__).with_name("fixtures") / "chatgpt_genui_npd.txt").read_text(encoding="utf-8")


def message(text: str) -> dict:
    return {"message": {"id": "choice", "author": {"role": "assistant"}, "channel": "final",
                        "content": {"content_type": "text", "parts": [text]}}}


def parsed(events: list[dict | str], thinking: bool) -> list[dict]:
    payloads = (json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event for event in events)
    return list(conversation.iter_conversation_payloads(payloads, thinking=thinking))


class GenUIPassthroughTests(unittest.TestCase):
    def test_sanitizer_preserves_the_complete_choice_card(self):
        self.assertEqual(conversation.sanitize_output_text(SOURCE), SOURCE)

    def test_snapshots_and_patches_preserve_genui_without_duplicates(self):
        snapshots = [message(SOURCE[:length]) for length in range(1, len(SOURCE) + 1)]
        patches = [message("")] + [
            {"p": "/message/content/parts/0", "o": "append", "v": character} for character in SOURCE
        ]
        # 重放完整或较短的历史快照，不得把声明或标签重复追加给前端。
        replay = [message(SOURCE), message(SOURCE[:100]), message(SOURCE)]
        for thinking in (False, True):
            for events in (snapshots, patches, replay):
                for ending in ([], ["[DONE]"]):
                    with self.subTest(thinking=thinking, style=len(events), ending=ending):
                        output = parsed(events + ending, thinking)
                        self.assertEqual("".join(item.get("delta", "") for item in output), SOURCE)
                        self.assertEqual("".join(item.get("reasoning_delta", "") for item in output), "")
                        self.assertEqual(output[-1]["text"], SOURCE)

    def test_chat_completion_stream_and_nonstream_keep_the_same_source(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                events = parsed([message(SOURCE[:100]), message(SOURCE), "[DONE]"], thinking)
                request = conversation.ConversationRequest(
                    model="auto", messages=[{"role": "user", "content": "NPD 是什么？"}],
                    thinking_effort="extended" if thinking else "",
                )
                with mock.patch.object(conversation, "_stream_text_failover", side_effect=lambda *_: iter(events)):
                    chunks = list(openai_v1_chat_complete.stream_text_chat_completion(
                        object(), request.messages, request.model, request.thinking_effort, emit_reasoning=True,
                    ))
                    content, reasoning, _seconds = conversation.collect_text_and_reasoning(object(), request)
                self.assertEqual("".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks), SOURCE)
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
                self.assertEqual(content, SOURCE)
                self.assertEqual(reasoning, "")


if __name__ == "__main__":
    unittest.main()
