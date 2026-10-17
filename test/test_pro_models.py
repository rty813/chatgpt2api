"""Pro 专属文本模型（gpt-6-pro / gpt-5-6-pro / gpt-5-5-pro / o3-pro）的接入回归。

上游对**任何** model slug 都照单全收并原样回显（实测伪造的 `gpt-99-pro` 也会正常作答），
所以「模型不对」不会以报错的形式暴露，只会静默降级。防线只能建在调度侧：
Pro 专属模型必须落在 Pro 账号上，没有 Pro 号就明确报错，绝不悄悄换成别的模型。
"""
from __future__ import annotations

import unittest
from unittest import mock

from fastapi import HTTPException

from services.protocol import conversation, openai_v1_models
from utils.helper import is_pro_text_model


class ProModelDetectionTests(unittest.TestCase):
    def test_pro_text_models_are_detected_by_suffix(self):
        for model in ("gpt-6-pro", "gpt-5-6-pro", "gpt-5-5-pro", "o3-pro", "GPT-6-PRO"):
            self.assertTrue(is_pro_text_model(model), model)
            self.assertEqual(conversation.required_text_plan_type(model), "pro")

    def test_non_pro_models_have_no_plan_requirement(self):
        # `pro-codex-gpt-image-2` 是图片模型的 `pro-` 前缀，不能被当成 Pro 文本模型。
        for model in ("gpt-6", "gpt-5-5", "gpt-5-5-thinking", "auto", "", "pro-codex-gpt-image-2"):
            self.assertFalse(is_pro_text_model(model), model)
            self.assertIsNone(conversation.required_text_plan_type(model))


class TextBackendPlanRoutingTests(unittest.TestCase):
    def test_pro_model_requests_a_pro_account(self):
        with mock.patch.object(
            conversation.account_service, "get_text_access_token", return_value="pro-token"
        ) as get_token:
            backend = conversation.text_backend("gpt-6-pro")
        get_token.assert_called_once_with(plan_type="pro")
        self.assertEqual(backend.access_token, "pro-token")

    def test_plain_model_keeps_default_account_selection(self):
        with mock.patch.object(
            conversation.account_service, "get_text_access_token", return_value="plus-token"
        ) as get_token:
            backend = conversation.text_backend("gpt-5-5")
        get_token.assert_called_once_with(plan_type=None)
        self.assertEqual(backend.access_token, "plus-token")

    def test_pro_model_without_pro_account_fails_loudly(self):
        with mock.patch.object(conversation.account_service, "get_text_access_token", return_value=""):
            with self.assertRaises(HTTPException) as ctx:
                conversation.text_backend("gpt-6-pro")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("gpt-6-pro", str(ctx.exception.detail))
        self.assertIn("Pro", str(ctx.exception.detail))

    def test_plain_model_without_account_still_returns_backend(self):
        # 非 Pro 模型维持既有行为：空 token 交给下游失败切换逻辑处理。
        with mock.patch.object(conversation.account_service, "get_text_access_token", return_value=""):
            self.assertEqual(conversation.text_backend("gpt-5-5").access_token, "")


class ProModelFailoverTests(unittest.TestCase):
    def test_account_rotation_keeps_the_pro_plan_filter(self):
        """换号（原 token 失效）时必须继续只挑 Pro 号，否则会悄悄降级到 Plus。"""
        request = conversation.ConversationRequest(model="gpt-6-pro", messages=[{"role": "user", "content": "hi"}])
        backend = mock.Mock(access_token="dead-token")

        def fake_events(active_backend, **kwargs):
            if active_backend.access_token == "dead-token":
                raise RuntimeError("Encountered invalidated oauth token for user")
            yield {"type": "conversation.delta", "delta": "ok", "reasoning_delta": ""}

        with (
            mock.patch.object(conversation, "conversation_events", side_effect=fake_events),
            mock.patch.object(conversation.account_service, "refresh_access_token", return_value="dead-token"),
            mock.patch.object(conversation.account_service, "remove_invalid_token", return_value=True),
            mock.patch.object(conversation.account_service, "mark_text_used"),
            mock.patch.object(
                conversation.account_service, "get_text_access_token", return_value="next-pro-token"
            ) as get_token,
        ):
            deltas = [event.get("delta") for event in conversation._stream_text_failover(backend, request)]

        self.assertEqual(deltas, ["ok"])
        self.assertEqual(get_token.call_args.kwargs.get("plan_type"), "pro")

    def test_missing_pro_account_mid_failover_reports_pro_requirement(self):
        request = conversation.ConversationRequest(model="gpt-6-pro", messages=[{"role": "user", "content": "hi"}])
        backend = mock.Mock(access_token="")
        with self.assertRaises(HTTPException) as ctx:
            list(conversation._stream_text_failover(backend, request))
        self.assertEqual(ctx.exception.status_code, 400)


class ProModelListingTests(unittest.TestCase):
    def setUp(self):
        openai_v1_models.reset_model_list_cache()
        self.addCleanup(openai_v1_models.reset_model_list_cache)

    @staticmethod
    def _upstream(*slugs):
        return {
            "object": "list",
            "data": [{"id": slug, "object": "model", "created": 0, "owned_by": "chatgpt",
                      "permission": [], "root": slug, "parent": None} for slug in slugs],
        }

    def test_listing_uses_a_pro_account_so_pro_slugs_are_exposed(self):
        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__", return_value=None) as init,
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI, "list_models",
                return_value=self._upstream("gpt-5-5", "gpt-6-pro"),
            ),
            mock.patch.object(openai_v1_models.account_service, "refresh_access_token",
                              side_effect=lambda token, **kwargs: token),
            mock.patch.object(
                openai_v1_models.account_service, "list_accounts",
                return_value=[
                    {"access_token": "plus-token", "type": "Plus", "source_type": "web"},
                    {"access_token": "pro-token", "type": "pro", "source_type": "web"},
                ],
            ),
        ):
            ids = {item["id"] for item in openai_v1_models.list_models()["data"]}

        self.assertIn("gpt-6-pro", ids)
        # 必须用 Pro 号去列，Plus 号看不到 `*-pro`。
        self.assertEqual(init.call_args.kwargs.get("access_token"), "pro-token")

    def test_listing_falls_back_to_anonymous_without_accounts(self):
        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__", return_value=None) as init,
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "list_models", return_value=self._upstream("gpt-5-5")),
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=[]),
        ):
            openai_v1_models.list_models()
        self.assertEqual(init.call_args.kwargs.get("access_token"), "")

    def test_disabled_accounts_are_not_used_for_listing(self):
        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__", return_value=None) as init,
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "list_models", return_value=self._upstream("gpt-5-5")),
            mock.patch.object(openai_v1_models.account_service, "refresh_access_token",
                              side_effect=lambda token, **kwargs: token),
            mock.patch.object(
                openai_v1_models.account_service, "list_accounts",
                return_value=[
                    {"access_token": "dead-pro", "type": "Pro", "status": "异常", "source_type": "web"},
                    {"access_token": "live-plus", "type": "Plus", "status": "正常", "source_type": "web"},
                ],
            ),
        ):
            openai_v1_models.list_models()
        self.assertEqual(init.call_args.kwargs.get("access_token"), "live-plus")

    def test_codex_only_pool_falls_back_to_anonymous_listing(self):
        # codex 账号的 token 问不了网页版模型列表，直接走匿名。
        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__", return_value=None) as init,
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "list_models", return_value=self._upstream("gpt-5-5")),
            mock.patch.object(
                openai_v1_models.account_service, "list_accounts",
                return_value=[{"access_token": "codex-token", "type": "Pro", "source_type": "codex"}],
            ),
        ):
            openai_v1_models.list_models()
        self.assertEqual(init.call_args.kwargs.get("access_token"), "")

    def test_failed_account_listing_falls_back_to_anonymous(self):
        calls: list[str] = []

        def fake_list_models(self):
            calls.append(self.access_token)
            if self.access_token:
                raise RuntimeError("auth_models failed: status=401")
            return ProModelListingTests._upstream("gpt-5-5")

        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__",
                              lambda self, access_token="": setattr(self, "access_token", access_token)),
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "list_models", fake_list_models),
            mock.patch.object(openai_v1_models.account_service, "refresh_access_token",
                              side_effect=lambda token, **kwargs: token),
            mock.patch.object(
                openai_v1_models.account_service, "list_accounts",
                return_value=[{"access_token": "dead-token", "type": "Pro", "source_type": "web"}],
            ),
        ):
            ids = {item["id"] for item in openai_v1_models.list_models()["data"]}

        self.assertEqual(calls, ["dead-token", ""])
        self.assertIn("gpt-5-5", ids)

    def test_upstream_list_is_cached(self):
        accounts = [{"access_token": "pro-token", "type": "Pro", "source_type": "web"}]
        with (
            mock.patch.object(openai_v1_models.OpenAIBackendAPI, "__init__", return_value=None),
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI, "list_models", return_value=self._upstream("gpt-6-pro"),
            ) as list_models,
            mock.patch.object(openai_v1_models.account_service, "refresh_access_token",
                              side_effect=lambda token, **kwargs: token),
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=accounts),
        ):
            first = openai_v1_models.list_models()
            second = openai_v1_models.list_models()

        self.assertEqual(list_models.call_count, 1)
        # 缓存必须发副本：上一次调用往 data 里追加的动态模型不能污染下一次。
        self.assertEqual([item["id"] for item in first["data"]], [item["id"] for item in second["data"]])


if __name__ == "__main__":
    unittest.main()
