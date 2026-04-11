import os
import unittest
from unittest import mock

from ai_scientist.treesearch.backend.utils import backoff_create


class _RetryableError(RuntimeError):
    pass


class BackoffCreateTests(unittest.TestCase):
    def test_backoff_create_returns_on_first_success(self) -> None:
        result = backoff_create(lambda: "ok", (_RetryableError,))
        self.assertEqual(result, "ok")

    def test_backoff_create_raises_after_max_retries(self) -> None:
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise _RetryableError("still failing")

        with mock.patch.dict(os.environ, {"AI_SCIENTIST_BACKOFF_MAX_TRIES": "3"}):
            with mock.patch("time.sleep", return_value=None):
                with self.assertRaises(_RetryableError):
                    backoff_create(always_fail, (_RetryableError,))

        self.assertEqual(call_count, 3)


if __name__ == "__main__":
    unittest.main()
