"""
Unified agent backend for AI-Scientist v3.

Replaces all direct model-provider SDK calls (OpenAI, Anthropic, Bedrock, etc.)
with coding agent orchestration via the `claude` CLI subprocess.
"""

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent backend errors."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent invocation exceeds timeout."""
    pass


class AgentOutputError(AgentError):
    """Raised when agent output cannot be parsed as expected."""
    pass


@dataclass
class AgentInvocationResult:
    """Result of a single agent invocation, including metadata for tracking."""

    content: str
    wall_clock_seconds: float
    input_size: int  # approximate input character count
    output_size: int  # output character count
    success: bool = True
    error: str | None = None


@dataclass
class AgentBackend:
    """Unified interface wrapping the `claude` CLI for agent-based LLM calls.

    Replaces both Path A (llm.py/vlm.py SDK calls) and Path B (treesearch
    backend_openai.py/backend_anthropic.py) with a single agent subprocess
    interface.

    Usage:
        backend = AgentBackend()
        result = backend.query("You are a scientist.", "Generate a hypothesis about X.")
        print(result)  # str or dict (if json_schema provided)
    """

    model: str = "sonnet"
    timeout: int = 300
    max_retries: int = 2
    verbose: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    extra_cli_args: list[str] = field(default_factory=list)

    # Invocation log for tracking
    _invocation_log: list[AgentInvocationResult] = field(
        default_factory=list, repr=False
    )

    def query(
        self,
        system_message: str | None = None,
        user_message: str | None = None,
        json_schema: dict | None = None,
        images: list[str] | None = None,
        timeout: int | None = None,
        model: str | None = None,
    ) -> str | dict:
        """Execute a single agent query via the claude CLI.

        Args:
            system_message: System prompt providing context/role.
            user_message: User prompt with the actual request.
            json_schema: If provided, request structured JSON output matching
                this schema. The response will be parsed and returned as a dict.
            images: List of image file paths to include (for VLM replacement).
            timeout: Override default timeout for this call (seconds).
            model: Override default model for this call.

        Returns:
            str if no json_schema, dict if json_schema provided.

        Raises:
            AgentTimeoutError: If the agent exceeds the timeout.
            AgentOutputError: If JSON parsing fails when json_schema is set.
            AgentError: For other subprocess failures.
        """
        effective_timeout = timeout or self.timeout
        effective_model = model or self.model
        # Map "agent" model name (used by v3 config) to actual CLI model
        if effective_model == "agent":
            effective_model = "sonnet"

        # Build the prompt that will be sent to the agent
        prompt = self._build_prompt(system_message, user_message, images)

        # Build CLI command
        cmd = self._build_command(
            prompt=prompt,
            system_message=system_message,
            json_schema=json_schema,
            model=effective_model,
        )

        # Execute -- pass prompt via stdin
        start_time = time.time()
        input_size = len(prompt)

        try:
            result_text = self._execute_cli(cmd, effective_timeout, stdin_text=prompt)
            elapsed = time.time() - start_time

            invocation = AgentInvocationResult(
                content=result_text,
                wall_clock_seconds=elapsed,
                input_size=input_size,
                output_size=len(result_text),
                success=True,
            )
            self._invocation_log.append(invocation)

            if self.verbose:
                logger.info(
                    "Agent invocation: %.1fs, in=%d chars, out=%d chars",
                    elapsed,
                    input_size,
                    len(result_text),
                )

            # Parse JSON if schema was requested
            if json_schema is not None:
                return self._parse_json_output(result_text, json_schema)

            return result_text

        except AgentTimeoutError:
            elapsed = time.time() - start_time
            invocation = AgentInvocationResult(
                content="",
                wall_clock_seconds=elapsed,
                input_size=input_size,
                output_size=0,
                success=False,
                error="timeout",
            )
            self._invocation_log.append(invocation)
            raise

        except Exception as e:
            elapsed = time.time() - start_time
            invocation = AgentInvocationResult(
                content="",
                wall_clock_seconds=elapsed,
                input_size=input_size,
                output_size=0,
                success=False,
                error=str(e),
            )
            self._invocation_log.append(invocation)
            raise AgentError(f"Agent invocation failed: {e}") from e

    def query_with_retry(
        self,
        system_message: str | None = None,
        user_message: str | None = None,
        json_schema: dict | None = None,
        images: list[str] | None = None,
        timeout: int | None = None,
        model: str | None = None,
    ) -> str | dict:
        """Query with automatic retry on failure.

        Retries up to self.max_retries times on transient errors.
        """
        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                return self.query(
                    system_message=system_message,
                    user_message=user_message,
                    json_schema=json_schema,
                    images=images,
                    timeout=timeout,
                    model=model,
                )
            except AgentTimeoutError:
                raise  # Don't retry timeouts
            except AgentError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Agent query failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        1 + self.max_retries,
                        wait,
                        e,
                    )
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]

    def get_invocation_log(self) -> list[AgentInvocationResult]:
        """Return the log of all invocations made through this backend."""
        return list(self._invocation_log)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics for all invocations."""
        if not self._invocation_log:
            return {
                "total_invocations": 0,
                "successful": 0,
                "failed": 0,
                "total_wall_clock_seconds": 0.0,
                "total_input_chars": 0,
                "total_output_chars": 0,
            }
        successful = [inv for inv in self._invocation_log if inv.success]
        failed = [inv for inv in self._invocation_log if not inv.success]
        return {
            "total_invocations": len(self._invocation_log),
            "successful": len(successful),
            "failed": len(failed),
            "total_wall_clock_seconds": sum(
                inv.wall_clock_seconds for inv in self._invocation_log
            ),
            "total_input_chars": sum(
                inv.input_size for inv in self._invocation_log
            ),
            "total_output_chars": sum(
                inv.output_size for inv in self._invocation_log
            ),
        }

    # --- Internal methods ---

    def _build_prompt(
        self,
        system_message: str | None,
        user_message: str | None,
        images: list[str] | None,
    ) -> str:
        """Build the full prompt string sent to the claude CLI via stdin.

        The system_message is passed via --system-prompt flag, so only the
        user_message (plus any image references) goes into the stdin prompt.
        If user_message is empty but system_message is set, we send a minimal
        prompt so the CLI doesn't reject empty stdin.
        """
        parts = []

        if user_message:
            parts.append(user_message)

        if images:
            parts.append("\n\n[Attached images for analysis:]")
            for img_path in images:
                parts.append(f"- {img_path}")

        prompt = "\n".join(parts) if parts else ""

        # Ensure we never send empty stdin when system_message is present
        if not prompt and system_message:
            prompt = "Please respond according to the system instructions above."

        return prompt

    def _build_command(
        self,
        prompt: str,
        system_message: str | None,
        json_schema: dict | None,
        model: str,
    ) -> list[str]:
        """Build the claude CLI command list.

        The prompt is passed via stdin (not positional argument) to avoid
        shell argument length limits and empty-prompt edge cases.
        """
        cmd = [
            "claude",
            "-p",  # print mode (non-interactive)
            "--model", model,
        ]

        if json_schema:
            # Use json output format for structured output (no --verbose
            # as it changes the JSON format from single-object to array)
            cmd.extend(["--output-format", "json"])
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        else:
            cmd.extend(["--output-format", "text"])

        if system_message:
            cmd.extend(["--system-prompt", system_message])

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if self.extra_cli_args:
            cmd.extend(self.extra_cli_args)

        # Prompt is NOT appended as a positional arg -- it goes via stdin
        # in _execute_cli to handle long prompts and avoid empty-arg issues.

        return cmd

    def _execute_cli(self, cmd: list[str], timeout: int, stdin_text: str = "") -> str:
        """Execute the claude CLI subprocess and return its stdout.

        Args:
            cmd: The command list (without prompt -- prompt comes via stdin).
            timeout: Maximum execution time in seconds.
            stdin_text: The prompt text to pipe via stdin.

        Returns:
            The stdout text from the subprocess.

        Raises:
            AgentTimeoutError: If the process exceeds the timeout.
            AgentError: If the process exits with non-zero status.
        """
        logger.debug("Executing agent CLI: %s", " ".join(cmd[:5]) + "...")

        try:
            result = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise AgentTimeoutError(
                f"Agent timed out after {timeout}s"
            ) from e

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            error_detail = stderr or stdout or "unknown error"
            raise AgentError(
                f"Agent CLI exited with code {result.returncode}: {error_detail}"
            )

        return result.stdout.strip()

    def _parse_json_output(self, text: str, schema: dict) -> dict:
        """Parse JSON from agent output.

        When --json-schema is used with --output-format json, the claude CLI
        returns a JSON envelope. We extract the result field. Falls back to
        extracting JSON from text if direct parse fails.

        Args:
            text: Raw agent output text.
            schema: The expected JSON schema (for error messages).

        Returns:
            Parsed dict.

        Raises:
            AgentOutputError: If JSON cannot be extracted.
        """
        # Try direct JSON parse first (claude --output-format json returns an envelope)
        try:
            parsed = json.loads(text)
            # The claude CLI --output-format json returns an envelope like:
            # {"type":"result", "result":"...", "structured_output":{...}, ...}
            if isinstance(parsed, dict):
                # Prefer structured_output (populated when --json-schema is used)
                if "structured_output" in parsed and isinstance(parsed["structured_output"], dict):
                    return parsed["structured_output"]
                # Fallback to result field
                if "result" in parsed:
                    result = parsed["result"]
                    if isinstance(result, str):
                        try:
                            return json.loads(result)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(result, dict):
                        return result
                # If it's already a dict matching what we want, return it
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON from markdown code blocks or raw braces
        extracted = _extract_json_from_text(text)
        if extracted is not None:
            return extracted

        raise AgentOutputError(
            f"Could not parse JSON from agent output. "
            f"Expected schema: {json.dumps(schema, indent=2)[:200]}... "
            f"Got: {text[:500]}..."
        )


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text, handling code blocks and raw JSON.

    Compatible with ai_scientist/llm.py's extract_json_between_markers().
    """
    import re

    # Try code block extraction first (```json ... ``` or ``` ... ```)
    patterns = [
        r"```json\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON object
    # Find the first { and last } to extract
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        # Clean control characters
        candidate = re.sub(r"[\x00-\x1f]", "", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None
