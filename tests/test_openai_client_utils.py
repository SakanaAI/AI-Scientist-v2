import unittest
from unittest import mock

from ai_scientist.openai_client_utils import (
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    _extract_response_payload,
    _summarize_failed_response_payload,
    _response_to_chat_completion,
    create_openai_client,
    get_openai_timeout_seconds,
)


class OpenAIClientUtilsTests(unittest.TestCase):
    def test_get_openai_timeout_seconds_defaults(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(
                get_openai_timeout_seconds(),
                DEFAULT_OPENAI_TIMEOUT_SECONDS,
            )

    def test_get_openai_timeout_seconds_prefers_env_override(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"AI_SCIENTIST_OPENAI_TIMEOUT_SECONDS": "42.5"},
            clear=False,
        ):
            self.assertEqual(get_openai_timeout_seconds(), 42.5)

    def test_create_openai_client_passes_timeout(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"AI_SCIENTIST_OPENAI_TIMEOUT_SECONDS": "17"},
            clear=False,
        ):
            with mock.patch("ai_scientist.openai_client_utils.openai.OpenAI") as mock_openai:
                create_openai_client(max_retries=3)

        _, kwargs = mock_openai.call_args
        self.assertEqual(kwargs["timeout"], 17.0)
        self.assertEqual(kwargs["max_retries"], 3)

    def test_extract_response_payload_prefers_completed_event(self) -> None:
        raw = """event: response.created
data: {"type":"response.created","response":{"id":"resp_created","status":"in_progress","output":[]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_done","status":"completed","model":"gpt-5.4-mini","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}}
"""

        payload = _extract_response_payload(raw)

        self.assertEqual(payload["id"], "resp_done")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "ok")

    def test_extract_response_payload_reconstructs_text_without_completed_event(self) -> None:
        raw = """event: response.created
data: {"type":"response.created","response":{"id":"resp_partial","status":"in_progress","model":"gpt-5.4-mini","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]},"output_index":0}

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_1","content_index":0,"text":"recoverability summary"}
"""

        payload = _extract_response_payload(raw)
        completion = _response_to_chat_completion(payload, fallback_model="fallback-model")

        self.assertEqual(payload["id"], "resp_partial")
        self.assertEqual(
            completion.choices[0].message.content,
            "recoverability summary",
        )

    def test_extract_response_payload_reconstructs_tool_call_without_completed_event(self) -> None:
        raw = """event: response.created
data: {"type":"response.created","response":{"id":"resp_tool_partial","status":"in_progress","model":"gpt-5.4-mini","output":[]}}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_1","name":"add"},"output_index":0}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"a\\":2,\\"b\\":3}","output_index":0}
"""

        payload = _extract_response_payload(raw)
        completion = _response_to_chat_completion(payload, fallback_model="fallback-model")

        tool_call = completion.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "add")
        self.assertEqual(tool_call.function.arguments, '{"a":2,"b":3}')

    def test_summarize_failed_response_payload_strips_large_internal_fields(self) -> None:
        payload = {
            "id": "resp_failed",
            "status": "failed",
            "instructions": "very long hidden prompt",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "leak"}]}],
            "error": {
                "code": "server_error",
                "message": (
                    "An error occurred while processing your request. "
                    "Please include the request ID 1234abcd-5678-efgh-ijkl-1234567890ab."
                ),
            },
        }

        summary = _summarize_failed_response_payload(payload)

        self.assertIn('"response_id": "resp_failed"', summary)
        self.assertIn('"error_code": "server_error"', summary)
        self.assertIn(
            '"request_id": "1234abcd-5678-efgh-ijkl-1234567890ab"',
            summary,
        )
        self.assertNotIn("hidden prompt", summary)
        self.assertNotIn('"output"', summary)

    def test_extract_response_payload_raises_sanitized_failed_response_error(self) -> None:
        raw = """event: response.failed
data: {"type":"response.failed","response":{"id":"resp_failed","status":"failed","instructions":"hidden","output":[{"type":"message"}],"error":{"code":"server_error","message":"An error occurred while processing your request. Please include the request ID deadbeef-0000-1111-2222-333333333333 in your message."}}}
"""

        with self.assertRaisesRegex(
            RuntimeError,
            "response_id",
        ) as exc_info:
            _extract_response_payload(raw)

        self.assertNotIn("instructions", str(exc_info.exception))
        self.assertNotIn('"output"', str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
