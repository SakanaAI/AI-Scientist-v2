# Claude Code Changelog

> Release notes for Claude Code, including new features, improvements, and bug fixes by version.

This page is generated from the [CHANGELOG.md on GitHub](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md).

Run `claude --version` to check your installed version.

## Version 2.1.92 (April 4, 2026)

* Added `forceRemoteSettingsRefresh` policy setting: when set, the CLI blocks startup until remote managed settings are freshly fetched, and exits if the fetch fails (fail-closed)
* Added interactive Bedrock setup wizard accessible from the login screen when selecting "3rd-party platform" — guides you through AWS authentication, region configuration, credential verification, and model pinning
* Added per-model and cache-hit breakdown to `/cost` for subscription users
* `/release-notes` is now an interactive version picker
* Remote Control session names now use your hostname as the default prefix (e.g. `myhost-graceful-unicorn`), overridable with `--remote-control-session-name-prefix`
* Pro users now see a footer hint when returning to a session after the prompt cache has expired, showing roughly how many tokens the next turn will send uncached
* Fixed subagent spawning permanently failing with "Could not determine pane count" after tmux windows are killed or renumbered during a long-running session
* Fixed prompt-type Stop hooks incorrectly failing when the small fast model returns `ok:false`, and restored `preventContinuation:true` semantics for non-Stop prompt-type hooks
* Fixed tool input validation failures when streaming emits array/object fields as JSON-encoded strings
* Fixed an API 400 error that could occur when extended thinking produced a whitespace-only text block alongside real content
* Fixed accidental feedback survey submissions from auto-pilot keypresses and consecutive-prompt digit collisions
* Fixed misleading "esc to interrupt" hint appearing alongside "esc to clear" when a text selection exists in fullscreen mode during processing
* Fixed Homebrew install update prompts to use the cask's release channel (`claude-code` → stable, `claude-code@latest` → latest)
* Fixed `ctrl+e` jumping to the end of the next line when already at end of line in multiline prompts
* Fixed an issue where the same message could appear at two positions when scrolling up in fullscreen mode (iTerm2, Ghostty, and other terminals with DEC 2026 support)
* Fixed idle-return "/clear to save X tokens" hint showing cumulative session tokens instead of current context size
* Fixed plugin MCP servers stuck "connecting" on session start when they duplicate a claude.ai connector that is unauthenticated
* Improved Write tool diff computation speed for large files (60% faster on files with tabs/`&`/`$`)
* Removed `/tag` command
* Removed `/vim` command (toggle vim mode via `/config` → Editor mode)
* Linux sandbox now ships the `apply-seccomp` helper in both npm and native builds, restoring unix-socket blocking for sandboxed commands

## Version 2.1.91 (April 2, 2026)

* Added MCP tool result persistence override via `_meta["anthropic/maxResultSizeChars"]` annotation (up to 500K), allowing larger results like DB schemas to pass through without truncation
* Added `disableSkillShellExecution` setting to disable inline shell execution in skills, custom slash commands, and plugin commands
* Added support for multi-line prompts in `claude-cli://open?q=` deep links (encoded newlines `%0A` no longer rejected)
* Plugins can now ship executables under `bin/` and invoke them as bare commands from the Bash tool
* Fixed transcript chain breaks on `--resume` that could lose conversation history when async transcript writes fail silently
* Fixed `cmd+delete` not deleting to start of line on iTerm2, kitty, WezTerm, Ghostty, and Windows Terminal
* Fixed plan mode in remote sessions losing track of the plan file after a container restart, which caused permission prompts on plan edits and an empty plan-approval modal
* Fixed JSON schema validation for `permissions.defaultMode: "auto"` in settings.json
* Fixed Windows version cleanup not protecting the active version's rollback copy
* `/feedback` now explains why it's unavailable instead of disappearing from the slash menu
* Improved `/claude-api` skill guidance for agent design patterns including tool surface decisions, context management, and caching strategy
* Improved performance: faster `stripAnsi` on Bun by routing through `Bun.stripANSI`
* Edit tool now uses shorter `old_string` anchors, reducing output tokens

## Version 2.1.90 (April 1, 2026)

* Added `/powerup` — interactive lessons teaching Claude Code features with animated demos
* Added `CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE` env var to keep the existing marketplace cache when `git pull` fails, useful in offline environments
* Added `.husky` to protected directories (acceptEdits mode)
* Fixed an infinite loop where the rate-limit options dialog would repeatedly auto-open after hitting your usage limit, eventually crashing the session
* Fixed `--resume` causing a full prompt-cache miss on the first request for users with deferred tools, MCP servers, or custom agents (regression since v2.1.69)
* Fixed `Edit`/`Write` failing with "File content has changed" when a PostToolUse format-on-save hook rewrites the file between consecutive edits
* Fixed `PreToolUse` hooks that emit JSON to stdout and exit with code 2 not correctly blocking the tool call
* Fixed collapsed search/read summary badge appearing multiple times in fullscreen scrollback when a CLAUDE.md file auto-loads during a tool call
* Fixed auto mode not respecting explicit user boundaries ("don't push", "wait for X before Y") even when the action would otherwise be allowed
* Fixed click-to-expand hover text being nearly invisible on light terminal themes
* Fixed UI crash when malformed tool input reached the permission dialog
* Fixed headers disappearing when scrolling `/model`, `/config`, and other selection screens
* Hardened PowerShell tool permission checks: fixed trailing `&` background job bypass, `-ErrorAction Break` debugger hang, archive-extraction TOCTOU, and parse-fail fallback deny-rule degradation
* Improved performance: eliminated per-turn JSON.stringify of MCP tool schemas on cache-key lookup
* Improved performance: SSE transport now handles large streamed frames in linear time (was quadratic)
* Improved performance: SDK sessions with long conversations no longer slow down quadratically on transcript writes
* Improved `/resume` all-projects view to load project sessions in parallel, improving load times for users with many projects
* Changed `--resume` picker to no longer show sessions created by `claude -p` or SDK invocations
* Removed `Get-DnsClientCache` and `ipconfig /displaydns` from auto-allow (DNS cache privacy)

## Version 2.1.89 (April 1, 2026)

* Added `"defer"` permission decision to `PreToolUse` hooks — headless sessions can pause at a tool call and resume with `-p --resume` to have the hook re-evaluate
* Added `CLAUDE_CODE_NO_FLICKER=1` environment variable to opt into flicker-free alt-screen rendering with virtualized scrollback
* Added `PermissionDenied` hook that fires after auto mode classifier denials — return `{retry: true}` to tell the model it can retry
* Added named subagents to `@` mention typeahead suggestions
* Auto mode: denied commands now show a notification and appear in `/permissions` → Recent tab where you can retry with `r`
* Fixed `Edit(//path/**)` and `Read(//path/**)` allow rules to check the resolved symlink target, not just the requested path
* Fixed voice push-to-talk not activating for some modifier-combo bindings, and voice mode on Windows failing with "WebSocket upgrade rejected with HTTP 101"
* Fixed Edit/Write tools doubling CRLF on Windows and stripping Markdown hard line breaks (two trailing spaces)
* Fixed `StructuredOutput` schema cache bug causing ~50% failure rate when using multiple schemas
* Fixed memory leak where large JSON inputs were retained as LRU cache keys in long-running sessions
* Fixed a crash when removing a message from very large session files (over 50MB)
* Fixed LSP server zombie state after crash — server now restarts on next request instead of failing until session restart
* Fixed prompt history entries containing CJK or emoji being silently dropped when they fall on a 4KB boundary in `~/.claude/history.jsonl`
* Fixed `/stats` undercounting tokens by excluding subagent usage, and losing historical data beyond 30 days when the stats cache format changes
* Fixed `-p --resume` hangs when the deferred tool input exceeds 64KB or no deferred marker exists, and `-p --continue` not resuming deferred tools
* Fixed `claude-cli://` deep links not opening on macOS
* Fixed MCP tool errors truncating to only the first content block when the server returns multi-element error content
* Fixed skill reminders and other system context being dropped when sending messages with images via the SDK
* Fixed PreToolUse/PostToolUse hooks to receive `file_path` as an absolute path for Write/Edit/Read tools, matching the documented behavior
* Fixed autocompact thrash loop — now detects when context refills to the limit immediately after compacting three times in a row and stops with an actionable error instead of burning API calls
* Fixed prompt cache misses in long sessions caused by tool schema bytes changing mid-session
* Fixed nested CLAUDE.md files being re-injected dozens of times in long sessions that read many files
* Fixed `--resume` crash when transcript contains a tool result from an older CLI version or interrupted write
* Fixed misleading "Rate limit reached" message when the API returned an entitlement error — now shows the actual error with actionable hints
* Fixed hooks `if` condition filtering not matching compound commands (`ls && git push`) or commands with env-var prefixes (`FOO=bar git push`)
* Fixed collapsed search/read group badges duplicating in terminal scrollback during heavy parallel tool use
* Fixed notification `invalidates` not clearing the currently-displayed notification immediately
* Fixed prompt briefly disappearing after submit when background messages arrived during processing
* Fixed Devanagari and other combining-mark text being truncated in assistant output
* Fixed rendering artifacts on main-screen terminals after layout shifts
* Fixed voice mode failing to request microphone permission on macOS Apple Silicon
* Fixed Shift+Enter submitting instead of inserting a newline on Windows Terminal Preview 1.25
* Fixed periodic UI jitter during streaming in iTerm2 when running inside tmux
* Fixed PowerShell tool incorrectly reporting failures when commands like `git push` wrote progress to stderr on Windows PowerShell 5.1
* Fixed a potential out-of-memory crash when the Edit tool was used on very large files (>1 GiB)
* Improved collapsed tool summary to show "Listed N directories" for `ls`/`tree`/`du` instead of "Read N files"
* Improved Bash tool to warn when a formatter/linter command modifies files you have previously read, preventing stale-edit errors
* Improved `@`-mention typeahead to rank source files above MCP resources with similar names
* Improved PowerShell tool prompt with version-appropriate syntax guidance (5.1 vs 7+)
* Changed `Edit` to work on files viewed via `Bash` with `sed -n` or `cat`, without requiring a separate `Read` call first
* Changed hook output over 50K characters to be saved to disk with a file path + preview instead of being injected directly into context
* Changed `cleanupPeriodDays: 0` in settings.json to be rejected with a validation error — it previously silently disabled transcript persistence
* Changed thinking summaries to no longer be generated by default in interactive sessions — set `showThinkingSummaries: true` in settings.json to restore
* Documented `TaskCreated` hook event and its blocking behavior
* Preserved task notifications when backgrounding a running command with Ctrl+B
* PowerShell tool on Windows: external-command arguments containing both a double-quote and whitespace now prompt instead of auto-allowing (PS 5.1 argument-splitting hardening)
* `/env` now applies to PowerShell tool commands (previously only affected Bash)
* `/usage` now hides redundant "Current week (Sonnet only)" bar for Pro and Enterprise plans
* Image paste no longer inserts a trailing space
* Pasting `!command` into an empty prompt now enters bash mode, matching typed `!` behavior

## Version 2.1.87 (March 29, 2026)

* Fixed messages in Cowork Dispatch not getting delivered

## Version 2.1.86 (March 27, 2026)

* Added `X-Claude-Code-Session-Id` header to API requests so proxies can aggregate requests by session without parsing the body
* Added `.jj` and `.sl` to VCS directory exclusion lists so Grep and file autocomplete don't descend into Jujutsu or Sapling metadata
* Fixed `--resume` failing with "tool_use ids were found without tool_result blocks" on sessions created before v2.1.85
* Fixed Write/Edit/Read failing on files outside the project root (e.g., `~/.claude/CLAUDE.md`) when conditional skills or rules are configured
* Fixed unnecessary config disk writes on every skill invocation that could cause performance issues and config corruption on Windows
* Fixed potential out-of-memory crash when using `/feedback` on very long sessions with large transcript files
* Fixed `--bare` mode dropping MCP tools in interactive sessions and silently discarding messages enqueued mid-turn
* Fixed the `c` shortcut copying only ~20 characters of the OAuth login URL instead of the full URL
* Fixed masked input (e.g., OAuth code paste) leaking the start of the token when wrapping across multiple lines on narrow terminals
* Fixed official marketplace plugin scripts failing with "Permission denied" on macOS/Linux since v2.1.83
* Fixed statusline showing another session's model when running multiple Claude Code instances and using `/model` in one of them
* Fixed scroll not following new messages after wheel scroll or click-to-select at the bottom of a long conversation
* Fixed `/plugin` uninstall dialog: pressing `n` now correctly uninstalls the plugin while preserving its data directory
* Fixed a regression where pressing Enter after clicking could leave the transcript blank until the response arrived
* Fixed `ultrathink` hint lingering after deleting the keyword
* Fixed memory growth in long sessions from markdown/highlight render caches retaining full content strings
* Reduced startup event-loop stalls when many claude.ai MCP connectors are configured (macOS keychain cache extended from 5s to 30s)
* Reduced token overhead when mentioning files with `@` — raw string content no longer JSON-escaped
* Improved prompt cache hit rate for Bedrock, Vertex, and Foundry users by removing dynamic content from tool descriptions
* Memory filenames in the "Saved N memories" notice now highlight on hover and open on click
* Skill descriptions in the `/skills` listing are now capped at 250 characters to reduce context usage
* Changed `/skills` menu to sort alphabetically for easier scanning
* Auto mode now shows "unavailable for your plan" when disabled by plan restrictions (was "temporarily unavailable")
* \[VSCode] Fixed extension incorrectly showing "Not responding" during long-running operations
* \[VSCode] Fixed extension defaulting Max plan users to Sonnet after the OAuth token refreshes (8 hours after login)
* Read tool now uses compact line-number format and deduplicates unchanged re-reads, reducing token usage

For the complete details on all versions, features added, and fixes implemented, please refer to the [full CHANGELOG.md on GitHub](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md).
