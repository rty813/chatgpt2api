from __future__ import annotations

import json
import unittest

from services.protocol.conversation import iter_conversation_payloads
from services.protocol.genui_resources import collect_genui_resources


class GenUIResourceTests(unittest.TestCase):
    def test_image_group_keeps_all_images_for_literal_and_constant_array_queries(self):
        queries = ["Project Hail Mary 2026 movie Rocky alien injured recovering glass sphere",
                   "Project Hail Mary movie Rocky alien rock creature Ryan Gosling"]
        for query_code, query in [(json.dumps(queries), queries), ('__dilConstants["2"]', queries),
                                  (json.dumps("洛基"), "洛基")]:
            with self.subTest(query=query_code):
                metadata = {"model_dil_v2": {
                    "code": '__dil.jsx(AsyncImageGroup,{"query":' + query_code + ',"aspectRatio":"5:4","layout":"bento","size":"lg","__resolutionId":__dilConstants["1"]})',
                    "constants": {"1": "group-id", "2": queries},
                    "appData": {"opGenui": {"componentResults": {"group-id": {
                        "componentName": "AsyncImageGroup", "state": {"images": [
                            {"content_url": "https://images.example.com/rocky-1.jpg", "title": "洛基 1"},
                            {"content_url": "javascript:alert(1)", "thumbnail_url": "https://images.example.com/rocky-2.jpg", "title": "洛基 2"},
                            {"content_url": "https://images.example.com/rocky-1.jpg"},
                            {"content_url": "javascript:alert(2)"}, None,
                        ]}}}}},
                }}
                body = '<AsyncImageGroup query={' + json.dumps(query, ensure_ascii=False) + '} aspectRatio="5:4" layout="bento" size="lg"/>'
                message = {"id": "group-answer", "author": {"role": "assistant"}, "channel": "final",
                           "status": "finished_successfully", "content": {"content_type": "text", "parts": [body]}}
                for thinking in [True, False]:
                    events = list(iter_conversation_payloads(iter([
                        json.dumps({"message": message}), json.dumps({"message": {**message, "metadata": metadata}}), "[DONE]",
                    ]), thinking=thinking))
                    deltas = [event["delta"] for event in events if event.get("delta")]
                    self.assertEqual(deltas[0], body)
                    resources = json.loads(deltas[1].split('data={', 1)[1].removesuffix('}/>'))
                    key = json.dumps(query, ensure_ascii=False, separators=(",", ":")) if isinstance(query, list) else query
                    images = resources["images"][key]["images"]
                    self.assertEqual([image["src"] for image in images], ["https://images.example.com/rocky-1.jpg", "https://images.example.com/rocky-2.jpg"])
                    self.assertNotIn("javascript:", deltas[1])
                    self.assertEqual(events[-1]["text"], "".join(deltas))

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
