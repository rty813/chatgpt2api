from __future__ import annotations

import json
import unittest

from services.protocol.conversation import iter_conversation_payloads
from services.protocol.genui_resources import collect_genui_resources


class GenUIResourceTests(unittest.TestCase):
    def test_late_resource_metadata_survives_append_only_stream(self):
        body = '<box><AsyncImage query="走廊"/><text>说明<Cite refs={["turn0view0"]}/></text></box>'
        message = {"id": "answer", "author": {"role": "assistant"}, "channel": "final",
                   "status": "finished_successfully", "content": {"content_type": "text", "parts": [body]}}
        metadata = {"model_dil_v2": {
            "code": '__dil.jsx(AsyncImage,{"query":"走廊","__resolutionId":__dilConstants["1"]})',
            "constants": {"1": "image-id"},
            "appData": {"opGenui": {"componentResults": {
                "image-id": {"componentName": "AsyncImage", "status": "resolved", "state": {
                    "images": [{"content_url": "https://images.example.com/corridor.png", "title": "走廊"}]}},
                "cite-id": {"componentName": "Cite", "state": {"items": [
                    {"ref_id": "turn0view0", "url": "https://example.com/paper", "title": "论文"},
                    {"ref_id": "unsafe", "url": "javascript:alert(1)"},
                ]}},
            }}},
        }}
        first = json.dumps({"message": message})
        message = {**message, "metadata": metadata}
        output = list(iter_conversation_payloads(iter([first, json.dumps({"message": message}), "[DONE]"]), thinking=True))
        deltas = [event["delta"] for event in output if event.get("delta")]
        self.assertEqual(deltas[0], body)
        self.assertTrue(deltas[1].startswith("\n\n<genui-resources data={"))
        self.assertIn('"src":"https://images.example.com/corridor.png"', deltas[1])
        self.assertIn('"turn0view0"', deltas[1])
        self.assertNotIn("javascript:", deltas[1])
        self.assertEqual(output[-1]["text"], "".join(deltas))

    def test_missing_resources_do_not_invent_image_urls(self):
        resources = {"images": {}, "citations": {}}
        collect_genui_resources({"metadata": {"model_dil_v2": {"code": "throw new Error()"}}}, resources)
        self.assertEqual(resources, {"images": {}, "citations": {}})


if __name__ == "__main__":
    unittest.main()
