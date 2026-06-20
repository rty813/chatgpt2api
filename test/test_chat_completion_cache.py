from __future__ import annotations

import unittest
from unittest import mock
import json
import base64

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import openai_v1_chat_complete, openai_v1_response
from services.protocol.chat_completion_cache import chat_completion_cache
from services.protocol.conversation import iter_conversation_payloads, sanitize_output_text
from utils.helper import extract_image_from_message_content


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP8z8BQDwAFgwJ/luzl4wAAAABJRU5ErkJggg=="
)
PNG_1X1_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")


class ChatCompletionCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_cache_settings = config.data.get("chat_completion_cache")
        config.data["chat_completion_cache"] = {
            "enabled": True,
            "ttl_seconds": 60,
            "max_entries": 32,
            "dedupe_inflight": True,
            "stream_cache": True,
            "normalize_messages": True,
            "drop_adjacent_duplicates": True,
            "drop_assistant_history": False,
        }
        chat_completion_cache.clear()

    def tearDown(self) -> None:
        if self.old_cache_settings is None:
            config.data.pop("chat_completion_cache", None)
        else:
            config.data["chat_completion_cache"] = self.old_cache_settings
        chat_completion_cache.clear()

    def test_repeated_non_stream_text_completion_uses_cache(self) -> None:
        calls = 0

        def fake_collect_text(_backend, _request):
            nonlocal calls
            calls += 1
            return f"cached answer {calls}", "", 0

        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "cache this exact prompt"}],
        }

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text_and_reasoning", side_effect=fake_collect_text),
        ):
            first = openai_v1_chat_complete.handle(body)
            second = openai_v1_chat_complete.handle(body)

        self.assertEqual(calls, 1)
        self.assertEqual(
            first["choices"][0]["message"]["content"],
            second["choices"][0]["message"]["content"],
        )

    def test_repeated_stream_text_completion_replays_cached_chunks(self) -> None:
        calls = 0

        def fake_stream_text_events(_backend, _request):
            nonlocal calls
            calls += 1
            yield {"content": "streamed", "reasoning": ""}
            yield {"content": " answer", "reasoning": ""}

        body = {
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "stream cache this exact prompt"}],
        }

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch(
                "services.protocol.openai_v1_chat_complete.stream_text_events",
                side_effect=fake_stream_text_events,
            ),
        ):
            first = list(openai_v1_chat_complete.handle(body))
            second = list(openai_v1_chat_complete.handle(body))

        self.assertEqual(calls, 1)
        self.assertEqual(first, second)
        content = "".join(str(chunk["choices"][0]["delta"].get("content") or "") for chunk in second)
        self.assertEqual(content, "streamed answer")

    def test_adjacent_duplicate_messages_are_removed_before_upstream_call(self) -> None:
        captured_messages = []

        def fake_collect_text(_backend, request):
            captured_messages.extend(request.messages or [])
            return "ok", "", 0

        body = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "repeat me"},
                {"role": "user", "content": "repeat me"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "next prompt"},
            ],
        }

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text_and_reasoning", side_effect=fake_collect_text),
        ):
            openai_v1_chat_complete.handle(body)

        self.assertEqual(
            captured_messages,
            [
                {"role": "user", "content": "repeat me"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "next prompt"},
            ],
        )

    def test_chat_completion_usage_includes_cached_tokens(self) -> None:
        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text_and_reasoning", return_value=("ok", "", 0)),
        ):
            response = openai_v1_chat_complete.handle({
                "model": "auto",
                "messages": [{"role": "user", "content": "usage shape"}],
            })

        details = response["usage"]["prompt_tokens_details"]
        self.assertEqual(details["cached_tokens"], 0)
        output_details = response["usage"]["completion_tokens_details"]
        self.assertEqual(output_details["reasoning_tokens"], 0)

    def test_responses_completed_usage_includes_cached_tokens(self) -> None:
        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", return_value=iter(["ok"])),
        ):
            response = openai_v1_response.handle({
                "model": "auto",
                "input": "usage shape",
            })

        details = response["usage"]["input_tokens_details"]
        self.assertEqual(details["cached_tokens"], 0)
        output_details = response["usage"]["output_tokens_details"]
        self.assertEqual(output_details["reasoning_tokens"], 0)

    def test_repeated_responses_text_request_uses_cache(self) -> None:
        calls = 0

        def fake_stream_text_deltas(_backend, _request):
            nonlocal calls
            calls += 1
            yield f"response cache {calls}"

        body = {
            "model": "auto",
            "input": "cache this responses prompt",
            "stream": True,
        }

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream_text_deltas),
        ):
            first = list(openai_v1_response.handle(body))
            second = list(openai_v1_response.handle(body))

        self.assertEqual(calls, 1)
        self.assertEqual(first, second)

    def test_responses_stream_emits_reasoning_summary_events(self) -> None:
        def fake_stream_text_events(_backend, _request):
            yield {"content": "", "reasoning": "浏览当前医学信息并引用来源", "seconds": 22}
            yield {"content": "青光眼合并白内障的治疗核心", "reasoning": "", "seconds": 22}

        body = {
            "model": "gpt-5-5-thinking",
            "input": "青光眼合并白内障如何治疗？",
            "stream": True,
        }

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_events", side_effect=fake_stream_text_events),
        ):
            events = list(openai_v1_response.handle(body))

        text_deltas = [
            str(event.get("delta") or "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        ]
        reasoning_deltas = [
            str(event.get("delta") or "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        ]
        content = "".join(text_deltas)
        reasoning = "".join(reasoning_deltas)
        self.assertEqual(content, "青光眼合并白内障的治疗核心")
        self.assertIn("浏览当前医学信息并引用来源", reasoning)
        self.assertNotIn("Thought for", reasoning)
        self.assertFalse(any("<think>" in text or "</think>" in text for text in [content, reasoning]))
        completed = [event["response"] for event in events if event.get("type") == "response.completed"][-1]
        self.assertEqual(completed["output"][0]["type"], "reasoning")
        self.assertEqual(completed["output"][1]["type"], "message")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "青光眼合并白内障的治疗核心")

    def test_output_sanitizer_removes_chatgpt_annotation_markup(self) -> None:
        text = (
            "Repo: \ue200url\ue202basketikun/chatgpt2api"
            "\ue202https://github.com/basketikun/chatgpt2api\ue201 "
            "details \ue200cite\ue202turn0search0\ue201."
        )

        self.assertEqual(
            sanitize_output_text(text),
            "Repo: basketikun/chatgpt2api (https://github.com/basketikun/chatgpt2api) details.",
        )

    def test_output_sanitizer_preserves_annotated_entity_text(self) -> None:
        text = (
            "The character is from \ue200entity\ue202Invincible\ue201, "
            "which is based on the comic series \ue200entity\ue202Invincible\ue201."
        )

        self.assertEqual(
            sanitize_output_text(text),
            "The character is from Invincible, which is based on the comic series Invincible.",
        )

    def test_output_sanitizer_preserves_readable_cite_label(self) -> None:
        text = "The character is \ue200cite\ue202Invincible\ue202turn0search0\ue201."

        self.assertEqual(sanitize_output_text(text), "The character is Invincible.")

    def test_stream_sanitizer_does_not_emit_partial_annotation_or_repeat_prefix(self) -> None:
        events = [
            {"p": "/message/content/parts/0", "o": "append", "v": "Repo: \ue200url\ue202chat"},
            {"p": "/message/content/parts/0", "o": "append", "v": "gpt2api\ue202turn0search0\ue201 done \ue200cite\ue202turn0\ue201."},
            "[DONE]",
        ]
        payloads = [json.dumps(event, ensure_ascii=False) if isinstance(event, dict) else event for event in events]
        deltas = [
            str(event.get("delta") or "")
            for event in iter_conversation_payloads(iter(payloads))
            if event.get("type") == "conversation.delta"
        ]

        self.assertEqual("".join(deltas), "Repo: chatgpt2api done.")
        self.assertFalse(any("\ue200" in delta or "\ue202" in delta or "\ue201" in delta for delta in deltas))

    def test_thinking_stream_emits_reasoning_recap_and_seconds(self) -> None:
        recap = {
            "v": {
                "message": {
                    "id": "reasoning-recap-1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "reasoning_recap", "content": "已思考 1m 11s"},
                    "metadata": {
                        "reasoning_status": "reasoning_ended",
                        "finished_duration_sec": 71,
                    },
                    "end_turn": False,
                },
                "conversation_id": "conv-1",
            },
        }
        marker = {"type": "message_marker", "message_id": "answer-1", "marker": "final_channel_token"}
        answer = {
            "v": {
                "message": {
                    "id": "answer-1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["最终结论"]},
                    "end_turn": True,
                },
                "conversation_id": "conv-1",
            },
        }
        payloads = [json.dumps(item, ensure_ascii=False) for item in (recap, marker, answer)] + ["[DONE]"]

        deltas = [
            event
            for event in iter_conversation_payloads(iter(payloads), thinking=True)
            if event.get("type") == "conversation.delta"
        ]

        self.assertEqual(deltas[0]["reasoning_delta"], "已思考 1m 11s")
        self.assertEqual(deltas[0]["reasoning_seconds"], 71)
        self.assertEqual(deltas[-1]["delta"], "最终结论")
        self.assertEqual(deltas[-1]["reasoning_seconds"], 71)

    def test_stream_chat_completion_wraps_recap_seconds(self) -> None:
        def fake_stream_text_events(_backend, _request):
            yield {"content": "", "reasoning": "已思考 1m 11s", "seconds": 71}
            yield {"content": "最终结论", "reasoning": "", "seconds": 71}

        with mock.patch("services.protocol.openai_v1_chat_complete.stream_text_events", side_effect=fake_stream_text_events):
            chunks = list(openai_v1_chat_complete.stream_text_chat_completion(
                object(),
                [{"role": "user", "content": "test"}],
                "gpt-5-5-thinking",
                "extended",
                emit_reasoning=True,
            ))

        content = "".join(str(chunk["choices"][0]["delta"].get("content") or "") for chunk in chunks)
        reasoning = "".join(str(chunk["choices"][0]["delta"].get("reasoning_content") or "") for chunk in chunks)
        self.assertEqual(content, "最终结论")
        self.assertIn("已思考 1m 11s", reasoning)
        self.assertNotIn("Thought for", reasoning)
        self.assertNotIn("<think>", content + reasoning)
        self.assertNotIn("</think>", content + reasoning)

    def test_chat_completion_non_stream_uses_reasoning_content(self) -> None:
        def fake_collect(_backend, _request):
            return "最终结论", "已思考 11s", 11

        body = {
            "model": "gpt-5-5-thinking",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        }

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text_and_reasoning", side_effect=fake_collect),
        ):
            response = openai_v1_chat_complete.handle(body)

        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "最终结论")
        self.assertIn("已思考 11s", message["reasoning_content"])
        self.assertNotIn("Thought for", message["reasoning_content"])
        self.assertNotIn("<think>", message["content"] + message["reasoning_content"])

    def test_thinking_stream_emits_reasoning_tool_activity(self) -> None:
        code = {
            "v": {
                "message": {
                    "id": "tool-call-1",
                    "author": {"role": "assistant"},
                    "recipient": "web.run",
                    "content": {
                        "content_type": "code",
                        "text": json.dumps({
                            "system1_search_query": [
                                {"q": "Docker Engine architecture containerd official docs"},
                            ],
                        }),
                    },
                    "metadata": {
                        "reasoning_status": "is_reasoning",
                        "reasoning_title": "查找官方文档和技术资料",
                    },
                    "end_turn": False,
                },
                "conversation_id": "conv-1",
            },
        }
        marker = {"type": "message_marker", "message_id": "answer-1", "marker": "final_channel_token"}
        answer = {
            "v": {
                "message": {
                    "id": "answer-1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["最终结论"]},
                    "end_turn": True,
                },
                "conversation_id": "conv-1",
            },
        }
        payloads = [json.dumps(item, ensure_ascii=False) for item in (code, marker, answer)] + ["[DONE]"]

        deltas = [
            event
            for event in iter_conversation_payloads(iter(payloads), thinking=True)
            if event.get("type") == "conversation.delta"
        ]

        reasoning = "".join(str(event.get("reasoning_delta") or "") for event in deltas)
        self.assertIn("查找官方文档和技术资料", reasoning)
        self.assertIn("Docker Engine architecture containerd official docs", reasoning)
        self.assertEqual(deltas[-1]["delta"], "最终结论")

    def test_conversation_poll_rebuilds_reasoning_activity_when_thoughts_are_empty(self) -> None:
        conversation = {
            "mapping": {
                "thoughts": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 1,
                        "content": {"content_type": "thoughts", "thoughts": []},
                        "metadata": {"reasoning_status": "is_reasoning"},
                    },
                },
                "tool": {
                    "message": {
                        "author": {"role": "assistant"},
                        "recipient": "web.run",
                        "create_time": 2,
                        "content": {
                            "content_type": "code",
                            "text": json.dumps({"open": [{"ref_id": "turn0search0"}]}),
                        },
                        "metadata": {
                            "reasoning_status": "is_reasoning",
                            "reasoning_title": "浏览官方Docker博客和文档",
                        },
                    },
                },
                "recap": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 3,
                        "content": {"content_type": "reasoning_recap", "content": "已思考 11s"},
                        "metadata": {"finished_duration_sec": 11},
                    },
                },
            },
        }

        reasoning, seconds = OpenAIBackendAPI._latest_assistant_reasoning(conversation)

        self.assertEqual(seconds, 11)
        self.assertIn("浏览官方Docker博客和文档", reasoning)
        self.assertIn("turn0search0", reasoning)

    def test_responses_tools_add_honest_no_tool_guard(self) -> None:
        model, messages = openai_v1_response.text_response_parts({
            "model": "auto",
            "input": "run echo hi",
            "tools": [{"type": "function", "name": "shell"}],
        })

        self.assertEqual(model, "auto")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("cannot execute local tools", str(messages[0]["content"]))

    def test_responses_web_search_tool_returns_search_output(self) -> None:
        search_result = {
            "answer": "Latest answer.",
            "sources": [{"title": "Example", "url": "https://example.com/news", "snippet": "Snippet"}],
        }
        body = {
            "model": "auto",
            "input": "latest example news",
            "tools": [{"type": "web_search"}],
        }

        with mock.patch("services.protocol.openai_v1_response.run_web_search", return_value=search_result) as search:
            response = openai_v1_response.handle(body)

        search.assert_called_once_with("latest example news")
        self.assertEqual(response["output"][0]["type"], "web_search_call")
        self.assertEqual(response["output"][0]["status"], "completed")
        self.assertEqual(response["output"][0]["action"]["query"], "latest example news")
        message = response["output"][1]
        self.assertEqual(message["type"], "message")
        content = message["content"][0]
        self.assertIn("Latest answer.", content["text"])
        self.assertEqual(content["annotations"][0]["type"], "url_citation")
        self.assertEqual(content["annotations"][0]["url"], "https://example.com/news")

    def test_responses_web_search_tool_streams_search_events(self) -> None:
        search_result = {
            "answer": "Streamed search answer.",
            "sources": [{"title": "Example", "url": "https://example.com/stream", "snippet": ""}],
        }
        body = {
            "model": "auto",
            "stream": True,
            "input": "stream search",
            "tools": [{"type": "web_search_preview"}],
        }

        with mock.patch("services.protocol.openai_v1_response.run_web_search", return_value=search_result):
            events = list(openai_v1_response.handle(body))

        event_types = [event["type"] for event in events]
        self.assertIn("response.web_search_call.in_progress", event_types)
        self.assertIn("response.web_search_call.searching", event_types)
        self.assertIn("response.web_search_call.completed", event_types)
        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["type"], "web_search_call")
        self.assertEqual(completed["output"][1]["type"], "message")

    def test_responses_versioned_web_search_tool_returns_search_output(self) -> None:
        search_result = {
            "answer": "Versioned search answer.",
            "sources": [{"title": "Example", "url": "https://example.com/versioned", "snippet": ""}],
        }
        body = {
            "model": "auto",
            "input": "versioned search",
            "tools": [{"type": "web_search_preview_2025_03_11"}],
        }

        with mock.patch("services.protocol.openai_v1_response.run_web_search", return_value=search_result) as search:
            response = openai_v1_response.handle(body)

        search.assert_called_once_with("versioned search")
        self.assertEqual(response["output"][0]["type"], "web_search_call")
        self.assertIn("Versioned search answer.", response["output"][1]["content"][0]["text"])

    def test_chat_completions_web_search_tool_returns_search_answer(self) -> None:
        search_result = {
            "answer": "Chat search answer.",
            "sources": [{"title": "Example", "url": "https://example.com/chat", "snippet": ""}],
        }
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "search chat"}],
            "tools": [{"type": "web_search"}],
        }

        with mock.patch("services.protocol.openai_v1_chat_complete.run_web_search", return_value=search_result) as search:
            response = openai_v1_chat_complete.handle(body)

        search.assert_called_once_with("search chat")
        message = response["choices"][0]["message"]
        self.assertIn("Chat search answer.", message["content"])
        self.assertEqual(message["annotations"][0]["type"], "url_citation")
        self.assertEqual(message["annotations"][0]["url_citation"]["url"], "https://example.com/chat")

    def test_chat_completions_web_search_options_trigger_search(self) -> None:
        search_result = {
            "answer": "Options search answer.",
            "sources": [{"title": "Example", "url": "https://example.com/options", "snippet": ""}],
        }
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "search options"}],
            "web_search_options": {"search_context_size": "low"},
        }

        with mock.patch("services.protocol.openai_v1_chat_complete.run_web_search", return_value=search_result) as search:
            response = openai_v1_chat_complete.handle(body)

        search.assert_called_once_with("search options")
        self.assertIn("Options search answer.", response["choices"][0]["message"]["content"])

    def test_chat_completions_search_model_triggers_search(self) -> None:
        search_result = {
            "answer": "Search model answer.",
            "sources": [{"title": "Example", "url": "https://example.com/model", "snippet": ""}],
        }
        body = {
            "model": "gpt-5-search-api-2026-06-01",
            "messages": [{"role": "user", "content": "search model"}],
        }

        with mock.patch("services.protocol.openai_v1_chat_complete.run_web_search", return_value=search_result) as search:
            response = openai_v1_chat_complete.handle(body)

        search.assert_called_once_with("search model")
        self.assertEqual(response["model"], "gpt-5-search-api-2026-06-01")
        self.assertIn("Search model answer.", response["choices"][0]["message"]["content"])

    def test_chat_completions_search_like_model_does_not_trigger_search(self) -> None:
        body = {
            "model": "gpt-5-search-apiary",
            "messages": [{"role": "user", "content": "not actually a search model"}],
        }

        with (
            mock.patch("services.protocol.openai_v1_chat_complete.run_web_search") as search,
            mock.patch("services.protocol.openai_v1_chat_complete.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_chat_complete.collect_text_and_reasoning", return_value=("plain text answer", "", 0)),
        ):
            response = openai_v1_chat_complete.handle(body)

        search.assert_not_called()
        self.assertIn("plain text answer", response["choices"][0]["message"]["content"])

    def test_chat_completions_accepts_remote_image_url(self) -> None:
        class FakeImageResponse:
            status_code = 200
            headers = {"content-type": "image/png", "content-length": str(len(PNG_1X1))}
            content = PNG_1X1

        with mock.patch("utils.helper.requests.get", return_value=FakeImageResponse()) as request_get:
            model, messages, _thinking_effort = openai_v1_chat_complete.text_chat_parts({
                "model": "auto",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                    ],
                }],
            })

        request_get.assert_called_once()
        self.assertEqual(model, "auto")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe this"})
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["data"], PNG_1X1)
        self.assertEqual(content[1]["mime"], "image/png")

    def test_responses_text_request_preserves_input_image(self) -> None:
        captured = {}

        def fake_stream_text_deltas(_backend, request):
            captured["messages"] = request.messages
            yield "red"

        body = {
            "model": "auto",
            "input": [
                {"type": "input_text", "text": "What color is this image?"},
                {"type": "input_image", "image_url": PNG_1X1_DATA_URL},
            ],
        }

        with (
            mock.patch("services.protocol.openai_v1_response.text_backend", return_value=object()),
            mock.patch("services.protocol.openai_v1_response.stream_text_deltas", side_effect=fake_stream_text_deltas),
        ):
            response = openai_v1_response.handle(body)

        self.assertEqual(response["output"][0]["content"][0]["text"], "red")
        content = captured["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "What color is this image?"})
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["mime"], "image/png")
        self.assertEqual(content[1]["data"], PNG_1X1)
        self.assertGreater(response["usage"]["input_tokens_details"]["image_tokens"], 0)

    def test_responses_text_request_accepts_remote_input_image_url(self) -> None:
        class FakeImageResponse:
            status_code = 200
            headers = {"content-type": "image/png", "content-length": str(len(PNG_1X1))}
            content = PNG_1X1

        with mock.patch("utils.helper.requests.get", return_value=FakeImageResponse()) as request_get:
            _model, messages = openai_v1_response.text_response_parts({
                "model": "auto",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {"type": "input_image", "image_url": {"url": "https://example.test/image.png"}},
                    ],
                }],
            })

        request_get.assert_called_once()
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe this"})
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["data"], PNG_1X1)
        self.assertEqual(content[1]["mime"], "image/png")

    def test_image_extractor_supports_extra_image_object_shapes(self) -> None:
        encoded = base64.b64encode(PNG_1X1).decode("ascii")

        images = extract_image_from_message_content([
            {"type": "image", "data": PNG_1X1, "mime": "image/png"},
            {"type": "input_image", "base64": encoded, "mime_type": "image/png"},
            {"type": "input_image", "source": {"type": "base64", "data": encoded, "media_type": "image/png"}},
        ])

        self.assertEqual(len(images), 3)
        self.assertEqual([image[1] for image in images], ["image/png", "image/png", "image/png"])
        self.assertTrue(all(image[0] == PNG_1X1 for image in images))


if __name__ == "__main__":
    unittest.main()
