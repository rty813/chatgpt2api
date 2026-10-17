from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from curl_cffi import requests

from utils.helper import UpstreamHTTPError, ensure_ok


class ErrorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        status, content_type, body = {
            "/json": (404, "application/json", b'{"detail":"Model not found"}'),
            "/text": (502, "text/plain; charset=utf-8", "上游不可用".encode()),
            "/rate-limit": (429, "application/json", b'{"detail":"Rate limited"}'),
            "/ok": (200, "text/event-stream", b"data: [DONE]\n\n"),
        }[self.path]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if status == 429:
            self.send_header("Retry-After", "15")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class UpstreamErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ErrorHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.session = requests.Session(trust_env=False)
        self.addCleanup(self.session.close)

    def test_streamed_json_error_preserves_body_and_closes_response(self):
        response = self.session.get(self.base_url + "/json", stream=True, timeout=5)
        self.addCleanup(response.close)
        # 复现 curl_cffi 未消费流前 text 为空（并会缓存该空值）的情况。
        self.assertEqual(response.text, "")
        with patch.object(response, "close", wraps=response.close) as close:
            with self.assertRaises(UpstreamHTTPError) as caught:
                ensure_ok(response, "/backend-api/conversation")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.body, {"detail": "Model not found"})
        self.assertIn("Model not found", str(caught.exception))
        close.assert_called_once()

    def test_streamed_text_error_preserves_unicode(self):
        response = self.session.get(self.base_url + "/text", stream=True, timeout=5)
        self.addCleanup(response.close)
        with self.assertRaises(UpstreamHTTPError) as caught:
            ensure_ok(response, "upstream")
        self.assertEqual(caught.exception.body, "上游不可用")

    def test_streamed_error_keeps_retry_after(self):
        response = self.session.get(self.base_url + "/rate-limit", stream=True, timeout=5)
        self.addCleanup(response.close)
        with self.assertRaises(UpstreamHTTPError) as caught:
            ensure_ok(response, "upstream")
        self.assertEqual(caught.exception.retry_after, 15)
        self.assertEqual(caught.exception.body, {"detail": "Rate limited"})

    def test_buffered_json_error_still_works(self):
        response = self.session.get(self.base_url + "/json", timeout=5)
        with self.assertRaises(UpstreamHTTPError) as caught:
            ensure_ok(response, "upstream")
        self.assertEqual(caught.exception.body, {"detail": "Model not found"})

    def test_successful_stream_is_not_consumed_or_closed(self):
        response = self.session.get(self.base_url + "/ok", stream=True, timeout=5)
        self.addCleanup(response.close)
        with patch.object(response, "close", wraps=response.close) as close:
            ensure_ok(response, "upstream")
            close.assert_not_called()
        self.assertEqual(b"".join(response.iter_content()), b"data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
