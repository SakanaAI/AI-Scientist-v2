# Choose a permission mode

Control whether Claude asks before editing files or running commands. Cycle modes with Shift+Tab in the CLI or use the mode selector in VS Code, Desktop, and claude.ai.

When Claude wants to edit a file, run a shell command, or make a network request, it pauses and asks you to approve the action. Permission modes control how often that pause happens. The mode you pick shapes the flow of a session: default mode has you review each action as it comes, while looser modes let Claude work in longer uninterrupted stretches and report back when done. Pick more oversight for sensitive work, or fewer interruptions when you trust the direction.

## Available modes

Each mode makes a different tradeoff between convenience and oversight. The table below shows what Claude can do without a permission prompt in each mode.

| Mode                                                                | What runs without asking                  | Best for                                |
| :------------------------------------------------------------------ | :---------------------------------------- | :-------------------------------------- |
| `default`                                                           | Reads only                                | Getting started, sensitive work         |
| [`acceptEdits`](#auto-approve-file-edits-with-acceptedits-mode)     | Reads and file edits                      | Iterating on code you're reviewing      |
| [`plan`](#analyze-before-you-edit-with-plan-mode)                   | Reads only                                | Exploring a codebase before changing it |
| [`auto`](#eliminate-prompts-with-auto-mode)                         | Everything, with background safety checks | Long tasks, reducing prompt fatigue     |
| [`dontAsk`](#allow-only-pre-approved-tools-with-dontask-mode)       | Only pre-approved tools                   | Locked-down CI and scripts              |
| [`bypassPermissions`](#skip-all-checks-with-bypasspermissions-mode) | Everything except protected paths         | Isolated containers and VMs only        |

Regardless of mode, writes to [protected paths](#protected-paths) are never auto-approved, guarding repository state and Claude's own configuration against accidental corruption.

Modes set the baseline. Layer [permission rules](/en/permissions#manage-permissions) on top to pre-approve or block specific tools in any mode except `bypassPermissions`, which skips the permission layer entirely.

## Switch permission modes

You can switch modes mid-session, at startup, or as a persistent default. The mode is set through these controls, not by asking Claude in chat. Select your interface below to see how to change it.

### CLI

**During a session**: press `Shift+Tab` to cycle `default` → `acceptEdits` → `plan`. The current mode appears in the status bar. Not every mode is in the default cycle:

* `auto`: appears after you opt in with `--enable-auto-mode` or the persisted equivalent in settings
* `bypassPermissions`: appears after you start with `--permission-mode bypassPermissions`, `--dangerously-skip-permissions`, or `--allow-dangerously-skip-permissions`; the `--allow-` variant adds the mode to the cycle without activating it
* `dontAsk`: never appears in the cycle; set it with `--permission-mode dontAsk`

Enabled optional modes slot in after `plan`, with `bypassPermissions` first and `auto` last. If you have both enabled, you will cycle through `bypassPermissions` on the way to `auto`.

**At startup**: pass the mode as a flag.

```bash
claude --permission-mode plan
```

**As a default**: set `defaultMode` in [settings](/en/settings#settings-files).

```json
{
  "permissions": {
    "defaultMode": "acceptEdits"
  }
}
```

The same `--permission-mode` flag works with `-p` for [non-interactive runs](/en/headless).

### VS Code

**During a session**: click the mode indicator at the bottom of the prompt box.

**As a default**: set `claudeCode.initialPermissionMode` in VS Code settings, or use the Claude Code extension settings panel.

The mode indicator shows these labels, mapped to the mode each one applies:

| UI label           | Mode                |
| :----------------- | :------------------ |
| Ask before edits   | `default`           |
| Edit automatically | `acceptEdits`       |
| Plan mode          | `plan`              |
| Auto mode          | `auto`              |
| Bypass permissions | `bypassPermissions` |

Auto mode appears in the mode indicator after you enable **Allow dangerously skip permissions** in the extension settings, but it stays unavailable until your account meets every requirement listed in the [auto mode section](#eliminate-prompts-with-auto-mode). The `claudeCode.initialPermissionMode` setting does not accept `auto`; to start in auto mode by default, set `defaultMode` in your Claude Code [`settings.json`](/en/settings#settings-files) instead.

Bypass permissions also requires the **Allow dangerously skip permissions** toggle before it appears in the mode indicator.

### JetBrains

The JetBrains plugin runs Claude Code in the IDE terminal, so switching modes works the same as in the CLI: press `Shift+Tab` to cycle, or pass `--permission-mode` when launching.

### Desktop

Use the mode selector next to the send button. Auto and Bypass permissions appear only after you enable them in Desktop settings. See the [Desktop guide](/en/desktop#choose-a-permission-mode).

### Web and mobile

Use the mode dropdown next to the prompt box on [claude.ai/code](https://claude.ai/code) or in the mobile app. Permission prompts appear in claude.ai for approval. Which modes appear depends on where the session runs:

* **Cloud sessions** on [Claude Code on the web](/en/claude-code-on-the-web): Auto accept edits and Plan mode. Ask permissions, Auto, and Bypass permissions are not available.
* **[Remote Control](/en/remote-control) sessions** on your local machine: Ask permissions, Auto accept edits, and Plan mode. Auto and Bypass permissions are not available.

For Remote Control, you can also set the starting mode when launching the host:

```bash
claude remote-control --permission-mode acceptEdits
```

## Auto-approve file edits with acceptEdits mode

`acceptEdits` mode lets Claude create and edit files in your working directory without prompting. Writes to [protected paths](#protected-paths) and all non-edit actions still prompt the same as default mode. The status bar shows `⏵⏵ accept edits on` while this mode is active.

Use `acceptEdits` when you want to review changes in your editor or via `git diff` after the fact rather than approving each edit inline. Press `Shift+Tab` once from default mode to enter it, or start with it directly:

```bash
claude --permission-mode acceptEdits
```

## Analyze before you edit with plan mode

Plan mode tells Claude to research and propose changes without making them. Claude reads files, runs shell commands to explore, and writes a plan, but does not edit your source. Permission prompts still apply the same as default mode.

Enter plan mode by pressing `Shift+Tab` or prefixing a single prompt with `/plan`. You can also start in plan mode from the CLI:

```bash
claude --permission-mode plan
```

Press `Shift+Tab` again to leave plan mode without approving a plan.

When the plan is ready, Claude presents it and asks how to proceed. From that prompt you can:

* Approve and start in auto mode
* Approve and accept edits
* Approve and review each edit manually
* Keep planning with feedback
* Refine with [Ultraplan](/en/ultraplan) for browser-based review

Each approve option also offers to clear the planning context first.

## Eliminate prompts with auto mode

Auto mode requires Claude Code v2.1.83 or later.

Auto mode lets Claude execute without permission prompts. A separate classifier model reviews actions before they run, blocking anything that escalates beyond your request, targets unrecognized infrastructure, or appears driven by hostile content Claude read.

Auto mode is a research preview. It reduces prompts but does not guarantee safety. Use it for tasks where you trust the general direction, not as a replacement for review on sensitive operations.

Auto mode is available only when your account meets all of these requirements:

* **Plan**: Team, Enterprise, or API. Not available on Pro or Max.
* **Admin**: on Team and Enterprise, an admin must enable it in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code) before users can turn it on. Admins can also lock it off by setting `permissions.disableAutoMode` to `"disable"` in [managed settings](/en/permissions#managed-settings).
* **Model**: Claude Sonnet 4.6 or Opus 4.6. Not available on Haiku or claude-3 models.
* **Provider**: Anthropic API only. Not available on Bedrock, Vertex, or Foundry.

If Claude Code reports auto mode as unavailable, one of these requirements is unmet; this is not a transient outage.

Once enabled, start with the flag and `auto` joins the `Shift+Tab` cycle:

```bash
claude --enable-auto-mode
```

### What the classifier blocks by default

The classifier trusts your working directory and your repo's configured remotes. Everything else is treated as external until you [configure trusted infrastructure](/en/permissions#configure-the-auto-mode-classifier).

**Blocked by default**:

* Downloading and executing code, like `curl | bash`
* Sending sensitive data to external endpoints
* Production deploys and migrations
* Mass deletion on cloud storage
* Granting IAM or repo permissions
* Modifying shared infrastructure
* Irreversibly destroying files that existed before the session
* Force push, or pushing directly to `main`

**Allowed by default**:

* Local file operations in your working directory
* Installing dependencies declared in your lock files or manifests
* Reading `.env` and sending credentials to their matching API
* Read-only HTTP requests
* Pushing to the branch you started on or one Claude created

Run `claude auto-mode defaults` to see the full rule lists. If routine actions get blocked, an administrator can add trusted repos, buckets, and services via the `autoMode.environment` setting: see [Configure the auto mode classifier](/en/permissions#configure-the-auto-mode-classifier).

### When auto mode falls back

Each denied action shows a notification and appears in `/permissions` under the Recently denied tab, where you can press `r` to retry it with a manual approval.

If the classifier blocks an action 3 times in a row or 20 times total, auto mode pauses and Claude Code resumes prompting. Approving the prompted action resumes auto mode. These thresholds are not configurable. Any allowed action resets the consecutive counter, while the total counter persists for the session and resets only when its own limit triggers a fallback.

In [non-interactive mode](/en/headless) with the `-p` flag, repeated blocks abort the session since there is no user to prompt.

Repeated blocks usually mean the classifier is missing context about your infrastructure. Use `/feedback` to report false positives, or have an administrator [configure trusted infrastructure](/en/permissions#configure-the-auto-mode-classifier).

### How the classifier evaluates actions

Each action goes through a fixed decision order. The first matching step wins:

1. Actions matching your [allow or deny rules](/en/permissions#manage-permissions) resolve immediately
2. Read-only actions and file edits in your working directory are auto-approved, except writes to [protected paths](#protected-paths)
3. Everything else goes to the classifier
4. If the classifier blocks, Claude receives the reason and tries an alternative

On entering auto mode, broad allow rules that grant arbitrary code execution are dropped:

* Blanket `Bash(*)`
* Wildcarded interpreters like `Bash(python*)`
* Package-manager run commands
* `Agent` allow rules

Narrow rules like `Bash(npm test)` carry over. Dropped rules are restored when you leave auto mode.

The classifier sees user messages, tool calls, and your CLAUDE.md content. Tool results are stripped, so hostile content in a file or web page cannot manipulate it directly. A separate server-side probe scans incoming tool results and flags suspicious content before Claude reads it. For more on how these layers work together, see the auto mode announcement and the engineering deep dive.

### How auto mode handles subagents

The classifier checks [subagent](/en/sub-agents) work at three points:

1. Before a subagent starts, the delegated task description is evaluated, so a dangerous-looking task is blocked at spawn time.
2. While the subagent runs, each of its actions goes through the classifier with the same rules as the parent session, and any `permissionMode` in the subagent's frontmatter is ignored.
3. When the subagent finishes, the classifier reviews its full action history; if that return check flags a concern, a security warning is prepended to the subagent's results.

### Cost and latency

The classifier currently runs on Claude Sonnet 4.6 regardless of your main session model. Classifier calls count toward your token usage. Each check sends a portion of the transcript plus the pending action, adding a round-trip before execution. Reads and working-directory edits outside protected paths skip the classifier, so the overhead comes mainly from shell commands and network operations.

## Allow only pre-approved tools with dontAsk mode

`dontAsk` mode auto-denies every tool that is not explicitly allowed. Only actions matching your `permissions.allow` rules can execute; explicit `ask` rules are also denied rather than prompting. This makes the mode fully non-interactive for CI pipelines or restricted environments where you pre-define exactly what Claude may do.

Set it at startup with the flag:

```bash
claude --permission-mode dontAsk
```

## Skip all checks with bypassPermissions mode

`bypassPermissions` mode disables permission prompts and safety checks so tool calls execute immediately. Writes to [protected paths](#protected-paths) are the only actions that still prompt. Only use this mode in isolated environments like containers, VMs, or devcontainers without internet access, where Claude Code cannot damage your host system.

You cannot enter `bypassPermissions` from a session that was started without one of the enabling flags; restart with one to enable it:

```bash
claude --permission-mode bypassPermissions
```

The `--dangerously-skip-permissions` flag is equivalent.

`bypassPermissions` offers no protection against prompt injection or unintended actions. For background safety checks without prompts, use [auto mode](#eliminate-prompts-with-auto-mode) instead. Administrators can block this mode by setting `permissions.disableBypassPermissionsMode` to `"disable"` in [managed settings](/en/permissions#managed-settings).

## Protected paths

Writes to a small set of paths are never auto-approved, in every mode. This prevents accidental corruption of repository state and Claude's own configuration. In `default`, `acceptEdits`, `plan`, and `bypassPermissions` these writes prompt; in `auto` they route to the classifier; in `dontAsk` they are denied.

Protected directories:

* `.git`
* `.vscode`
* `.idea`
* `.husky`
* `.claude`, except for `.claude/commands`, `.claude/agents`, `.claude/skills`, and `.claude/worktrees` where Claude routinely creates content

Protected files:

* `.gitconfig`, `.gitmodules`
* `.bashrc`, `.bash_profile`, `.zshrc`, `.zprofile`, `.profile`
* `.ripgreprc`
* `.mcp.json`, `.claude.json`

## See also

* [Permissions](/en/permissions): allow, ask, and deny rules; auto mode classifier configuration; managed policies
* [Hooks](/en/hooks): custom permission logic via `PreToolUse` and `PermissionRequest` hooks
* [Ultraplan](/en/ultraplan): run plan mode in a Claude Code on the web session with browser-based review
* [Security](/en/security): safeguards and best practices
* [Sandboxing](/en/sandboxing): filesystem and network isolation for Bash commands
* [Non-interactive mode](/en/headless): run Claude Code with the `-p` flag
