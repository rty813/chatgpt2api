from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from services.config import ConfigStore, config
from services.openai_backend_api import OpenAIBackendAPI
from utils.browser_timezone import browser_time, timezone_offset_min, validate_timezone


class BrowserTimezoneTests(unittest.TestCase):
    def test_los_angeles_date_string_matches_browser_userscript(self):
        # 由 en-US Intl.DateTimeFormat(timeZoneName="long") / Date.toString 交叉验证。
        for instant, expected in (
            ("2026-01-15T12:00:00+00:00", "Thu Jan 15 2026 04:00:00 GMT-0800 (Pacific Standard Time)"),
            ("2026-07-15T12:00:00+00:00", "Wed Jul 15 2026 05:00:00 GMT-0700 (Pacific Daylight Time)"),
            ("2026-03-08T09:59:00+00:00", "Sun Mar 08 2026 01:59:00 GMT-0800 (Pacific Standard Time)"),
            ("2026-03-08T10:00:00+00:00", "Sun Mar 08 2026 03:00:00 GMT-0700 (Pacific Daylight Time)"),
            ("2026-11-01T08:59:00+00:00", "Sun Nov 01 2026 01:59:00 GMT-0700 (Pacific Daylight Time)"),
            ("2026-11-01T09:00:00+00:00", "Sun Nov 01 2026 01:00:00 GMT-0800 (Pacific Standard Time)"),
        ):
            with self.subTest(instant=instant):
                self.assertEqual(browser_time("America/Los_Angeles", at=datetime.fromisoformat(instant)), expected)

    def test_offsets_follow_daylight_saving_and_javascript_sign(self):
        winter = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
        summer = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)
        for name, winter_offset, summer_offset in (
            ("America/Los_Angeles", 480, 420),
            ("Europe/Berlin", -60, -120),
            ("Asia/Shanghai", -480, -480),
            ("Asia/Kolkata", -330, -330),
        ):
            with self.subTest(timezone=name):
                self.assertEqual(timezone_offset_min(name, at=winter), winter_offset)
                self.assertEqual(timezone_offset_min(name, at=summer), summer_offset)

    def test_sentinel_time_and_request_offset_agree_at_dst_boundary(self):
        for instant, offset, date_text in (
            (datetime(2026, 3, 8, 9, 59, tzinfo=timezone.utc), 480, "01:59:00 GMT-0800"),
            (datetime(2026, 3, 8, 10, tzinfo=timezone.utc), 420, "03:00:00 GMT-0700"),
        ):
            with self.subTest(instant=instant):
                self.assertEqual(timezone_offset_min("America/Los_Angeles", at=instant), offset)
                self.assertIn(date_text, browser_time("America/Los_Angeles", at=instant))

    def test_invalid_timezone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "IANA"):
            validate_timezone("Not/A_Timezone")

    def test_config_environment_override_and_invalid_update(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"auth-key": "test-auth", "upstream_timezone": "Europe/Berlin"}))
            store = ConfigStore(path)
            self.assertEqual(store.upstream_timezone, "Europe/Berlin")
            with patch.dict(os.environ, {"CHATGPT2API_UPSTREAM_TIMEZONE": "America/Los_Angeles"}):
                self.assertEqual(store.upstream_timezone, "America/Los_Angeles")
            before = path.read_text()
            with self.assertRaises(ValueError):
                store.update({"upstream_timezone": "Not/A_Timezone"})
            self.assertEqual(path.read_text(), before)
            self.assertEqual(store.upstream_timezone, "Europe/Berlin")

    def make_backend(self, account=None):
        with (
            patch.dict(config.data, {"upstream_timezone": "America/Los_Angeles"}),
            patch.dict(os.environ, {"CHATGPT2API_UPSTREAM_TIMEZONE": ""}),
            patch("services.openai_backend_api.account_service.get_account", return_value=account or {}),
        ):
            backend = OpenAIBackendAPI("test-token")
        self.addCleanup(backend.session.close)
        return backend

    def test_account_timezone_overrides_global_for_independent_proxy(self):
        backend = self.make_backend({"timezone": "Asia/Tokyo"})
        self.assertEqual(backend.timezone, "Asia/Tokyo")
        self.assertEqual(backend.timezone_offset_min, -540)

    def test_authenticated_and_anonymous_chat_use_configured_timezone(self):
        backend = self.make_backend()
        for access_token, route in (("test-token", "/backend-api/conversation"), ("", "/backend-anon/conversation")):
            with self.subTest(route=route):
                backend.access_token = access_token
                path, name = backend._chat_target()
                payload = backend._conversation_payload([{"role": "user", "content": "hello"}], "auto", name)
                self.assertEqual(path, route)
                self.assertEqual(payload["timezone"], "America/Los_Angeles")
                self.assertEqual(payload["timezone_offset_min"], timezone_offset_min("America/Los_Angeles"))

    def test_prepare_proof_and_payload_share_timezone(self):
        backend = self.make_backend()
        prepare = Mock(status_code=200)
        prepare.json.return_value = {
            "prepare_token": "prepared",
            "proofofwork": {"required": True, "seed": "test", "difficulty": "ff"},
        }
        finalize = Mock(status_code=200)
        finalize.json.return_value = {"token": "requirements"}
        with patch.object(backend.session, "post", side_effect=[prepare, finalize]) as post:
            backend._get_chat_requirements()
        p_token = post.call_args_list[0].kwargs["json"]["p"]
        proof_token = post.call_args_list[1].kwargs["json"]["proof_token"]
        expected = datetime.now(ZoneInfo(backend.timezone)).strftime("GMT%z")
        for token in (p_token, proof_token):
            fingerprint = json.loads(base64.b64decode(token[7:]))
            self.assertIn(expected, fingerprint[1])
        payload = backend._conversation_payload([{"role": "user", "content": "hello"}], "auto", backend.timezone)
        self.assertEqual(payload["timezone_offset_min"], backend.timezone_offset_min)

    def test_init_and_prepare_use_configured_offset(self):
        backend = self.make_backend()
        response = Mock(status_code=200)
        response.json.return_value = {"conduit_token": "prepared"}
        with patch.object(backend.session, "post", return_value=response) as post:
            backend._get_conversation_init()
            backend._prepare_search_conversation("hello", "auto")
            backend._prepare_editable_conversation("hello", [])
        for call in post.call_args_list:
            payload = call.kwargs["json"]
            self.assertEqual(payload["timezone_offset_min"], backend.timezone_offset_min)
            if "timezone" in payload:
                self.assertEqual(payload["timezone"], backend.timezone)


if __name__ == "__main__":
    unittest.main()
