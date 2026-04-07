# Continue local sessions from any device with Remote Control

Continue a local Claude Code session from your phone, tablet, or any browser using Remote Control. Works with claude.ai/code and the Claude mobile app.

> Remote Control is available on all plans. On Team and Enterprise, it is off by default until an admin enables the Remote Control toggle in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code).

Remote Control connects [claude.ai/code](https://claude.ai/code) or the Claude app for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) and [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) to a Claude Code session running on your machine. Start a task at your desk, then pick it up from your phone on the couch or a browser on another computer.

When you start a Remote Control session on your machine, Claude keeps running locally the entire time, so nothing moves to the cloud. With Remote Control you can:

* **Use your full local environment remotely**: your filesystem, [MCP servers](/en/mcp), tools, and project configuration all stay available
* **Work from both surfaces at once**: the conversation stays in sync across all connected devices, so you can send messages from your terminal, browser, and phone interchangeably
* **Survive interruptions**: if your laptop sleeps or your network drops, the session reconnects automatically when your machine comes back online

Unlike [Claude Code on the web](/en/claude-code-on-the-web), which runs on cloud infrastructure, Remote Control sessions run directly on your machine and interact with your local filesystem. The web and mobile interfaces are just a window into that local session.

> Remote Control requires Claude Code v2.1.51 or later. Check your version with `claude --version`.

This page covers setup, how to start and connect to sessions, and how Remote Control compares to Claude Code on the web.

## Requirements

Before using Remote Control, confirm that your environment meets these conditions:

* **Subscription**: available on Pro, Max, Team, and Enterprise plans. API keys are not supported. On Team and Enterprise, an admin must first enable the Remote Control toggle in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code).
* **Authentication**: run `claude` and use `/login` to sign in through claude.ai if you haven't already.
* **Workspace trust**: run `claude` in your project directory at least once to accept the workspace trust dialog.

## Start a Remote Control session

You can start a dedicated Remote Control server, start an interactive session with Remote Control enabled, or connect a session that's already running.

### Server mode

Navigate to your project directory and run:

```bash
claude remote-control
```

The process stays running in your terminal in server mode, waiting for remote connections. It displays a session URL you can use to [connect from another device](#connect-from-another-device), and you can press spacebar to show a QR code for quick access from your phone. While a remote session is active, the terminal shows connection status and tool activity.

Available flags:

| Flag                                            | Description                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--name "My Project"`                           | Set a custom session title visible in the session list at claude.ai/code.                                                                                                                                                                                                                                                                                                        |
| `--remote-control-session-name-prefix <prefix>` | Prefix for auto-generated session names when no explicit name is set. Defaults to your machine's hostname, producing names like `myhost-graceful-unicorn`. Set `CLAUDE_REMOTE_CONTROL_SESSION_NAME_PREFIX` for the same effect.                                                                                                                                                  |
| `--spawn <mode>`                                | How concurrent sessions are created. Press `w` at runtime to toggle.<br />• `same-dir` (default): all sessions share the current working directory, so they can conflict if editing the same files.<br />• `worktree`: each on-demand session gets its own [git worktree](/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees). Requires a git repository. |
| `--capacity <N>`                                | Maximum number of concurrent sessions. Default is 32.                                                                                                                                                                                                                                                                                                                            |
| `--verbose`                                     | Show detailed connection and session logs.                                                                                                                                                                                                                                                                                                                                       |
| `--sandbox` / `--no-sandbox`                    | Enable or disable [sandboxing](/en/sandboxing) for filesystem and network isolation. Off by default.                                                                                                                                                                                                                                                                             |

### Interactive session

To start a normal interactive Claude Code session with Remote Control enabled, use the `--remote-control` flag (or `--rc`):

```bash
claude --remote-control
```

Optionally pass a name for the session:

```bash
claude --remote-control "My Project"
```

This gives you a full interactive session in your terminal that you can also control from claude.ai or the Claude app. Unlike `claude remote-control` (server mode), you can type messages locally while the session is also available remotely.

### From an existing session

If you're already in a Claude Code session and want to continue it remotely, use the `/remote-control` (or `/rc`) command:

```
/remote-control
```

Pass a name as an argument to set a custom session title:

```
/remote-control My Project
```

This starts a Remote Control session that carries over your current conversation history and displays a session URL and QR code you can use to [connect from another device](#connect-from-another-device). The `--verbose`, `--sandbox`, and `--no-sandbox` flags are not available with this command.

### Connect from another device

Once a Remote Control session is active, you have a few ways to connect from another device:

* **Open the session URL** in any browser to go directly to the session on [claude.ai/code](https://claude.ai/code). Both `claude remote-control` and `/remote-control` display this URL in the terminal.
* **Scan the QR code** shown alongside the session URL to open it directly in the Claude app. With `claude remote-control`, press spacebar to toggle the QR code display.
* **Open [claude.ai/code](https://claude.ai/code) or the Claude app** and find the session by name in the session list. Remote Control sessions show a computer icon with a green status dot when online.

The remote session title is chosen in this order:

1. The name you passed to `--name`, `--remote-control`, or `/remote-control`
2. The title you set with `/rename`
3. The last meaningful message in existing conversation history
4. An auto-generated name like `myhost-graceful-unicorn`, where `myhost` is your machine's hostname or the prefix you set with `--remote-control-session-name-prefix`

If you didn't set an explicit name, the title updates to reflect your prompt once you send one.

If the environment already has an active session, you'll be asked whether to continue it or start a new one.

If you don't have the Claude app yet, use the `/mobile` command inside Claude Code to display a download QR code for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) or [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude).

### Enable Remote Control for all sessions

By default, Remote Control only activates when you explicitly run `claude remote-control`, `claude --remote-control`, or `/remote-control`. To enable it automatically for every interactive session, run `/config` inside Claude Code and set **Enable Remote Control for all sessions** to `true`. Set it back to `false` to disable.

With this setting on, each interactive Claude Code process registers one remote session. If you run multiple instances, each one gets its own environment and session. To run multiple concurrent sessions from a single process, use server mode with `--spawn` instead.

## Connection and security

Your local Claude Code session makes outbound HTTPS requests only and never opens inbound ports on your machine. When you start Remote Control, it registers with the Anthropic API and polls for work. When you connect from another device, the server routes messages between the web or mobile client and your local session over a streaming connection.

All traffic travels through the Anthropic API over TLS, the same transport security as any Claude Code session. The connection uses multiple short-lived credentials, each scoped to a single purpose and expiring independently.

## Remote Control vs Claude Code on the web

Remote Control and [Claude Code on the web](/en/claude-code-on-the-web) both use the claude.ai/code interface. The key difference is where the session runs: Remote Control executes on your machine, so your local MCP servers, tools, and project configuration stay available. Claude Code on the web executes in Anthropic-managed cloud infrastructure.

Use Remote Control when you're in the middle of local work and want to keep going from another device. Use Claude Code on the web when you want to kick off a task without any local setup, work on a repo you don't have cloned, or run multiple tasks in parallel.

## Limitations

* **One remote session per interactive process**: outside of server mode, each Claude Code instance supports one remote session at a time. Use server mode with `--spawn` to run multiple concurrent sessions from a single process.
* **Terminal must stay open**: Remote Control runs as a local process. If you close the terminal or stop the `claude` process, the session ends. Run `claude remote-control` again to start a new one.
* **Extended network outage**: if your machine is awake but unable to reach the network for more than roughly 10 minutes, the session times out and the process exits. Run `claude remote-control` again to start a new session.
* **Ultraplan disconnects Remote Control**: starting an [ultraplan](/en/ultraplan) session disconnects any active Remote Control session because both features occupy the claude.ai/code interface and only one can be connected at a time.

## Troubleshooting

### "Remote Control requires a claude.ai subscription"

You're not authenticated with a claude.ai account. Run `claude auth login` and choose the claude.ai option. If `ANTHROPIC_API_KEY` is set in your environment, unset it first.

### "Remote Control requires a full-scope login token"

You're authenticated with a long-lived token from `claude setup-token` or the `CLAUDE_CODE_OAUTH_TOKEN` environment variable. These tokens are limited to inference-only and cannot establish Remote Control sessions. Run `claude auth login` to authenticate with a full-scope session token instead.

### "Unable to determine your organization for Remote Control eligibility"

Your cached account information is stale or incomplete. Run `claude auth login` to refresh it.

### "Remote Control is not yet enabled for your account"

The eligibility check can fail with certain environment variables present:

* `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` or `DISABLE_TELEMETRY`: unset them and try again.
* `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, or `CLAUDE_CODE_USE_FOUNDRY`: Remote Control requires claude.ai authentication and does not work with third-party providers.

If none of these are set, run `/logout` then `/login` to refresh.

### "Remote Control is disabled by your organization's policy"

This error has three distinct causes. Run `/status` first to see which login method and subscription you're using.

* **You're authenticated with an API key or Console account**: Remote Control requires claude.ai OAuth. Run `/login` and choose the claude.ai option. If `ANTHROPIC_API_KEY` is set in your environment, unset it.
* **Your Team or Enterprise admin hasn't enabled it**: Remote Control is off by default on these plans. An admin can enable it at [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code) by turning on the **Remote Control** toggle. This is a server-side organization setting, not a [managed settings](/en/permissions#managed-only-settings) key.
* **The admin toggle is grayed out**: your organization has a data retention or compliance configuration that is incompatible with Remote Control. This cannot be changed from the admin panel. Contact Anthropic support to discuss options.

### "Remote credentials fetch failed"

Claude Code could not obtain a short-lived credential from the Anthropic API to establish the connection. Re-run with `--verbose` to see the full error:

```bash
claude remote-control --verbose
```

Common causes:

* Not signed in: run `claude` and use `/login` to authenticate with your claude.ai account. API key authentication is not supported for Remote Control.
* Network or proxy issue: a firewall or proxy may be blocking the outbound HTTPS request. Remote Control requires access to the Anthropic API on port 443.
* Session creation failed: if you also see `Session creation failed — see debug log`, the failure happened earlier in setup. Check that your subscription is active.

## Choose the right approach

Claude Code offers several ways to work when you're not at your terminal. They differ in what triggers the work, where Claude runs, and how much you need to set up.

|                                                | Trigger                                                                                        | Claude runs on                                                                                          | Setup                                                                                                                                                | Best for                                                      |
| :--------------------------------------------- | :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| [Dispatch](/en/desktop#sessions-from-dispatch) | Message a task from the Claude mobile app                                                      | Your machine (Desktop)                                                                                  | [Pair the mobile app with Desktop](https://support.claude.com/en/articles/13947068)                                                  | Delegating work while you're away, minimal setup              |
| [Remote Control](/en/remote-control)           | Drive a running session from [claude.ai/code](https://claude.ai/code) or the Claude mobile app | Your machine (CLI or VS Code)                                                                           | Run `claude remote-control`                                                                                                          | Steering in-progress work from another device                 |
| [Channels](/en/channels)                       | Push events from a chat app like Telegram or Discord, or your own server                       | Your machine (CLI)                                                                                      | [Install a channel plugin](/en/channels#quickstart) or [build your own](/en/channels-reference)                                      | Reacting to external events like CI failures or chat messages |
| [Slack](/en/slack)                             | Mention `@Claude` in a team channel                                                            | Anthropic cloud                                                                                         | [Install the Slack app](/en/slack#setting-up-claude-code-in-slack) with [Claude Code on the web](/en/claude-code-on-the-web) enabled | PRs and reviews from team chat                                |
| [Scheduled tasks](/en/scheduled-tasks)         | Set a schedule                                                                                 | [CLI](/en/scheduled-tasks), [Desktop](/en/desktop-scheduled-tasks), or [cloud](/en/web-scheduled-tasks) | Pick a frequency                                                                                                                     | Recurring automation like daily reviews                       |

## Related resources

* [Claude Code on the web](/en/claude-code-on-the-web): run sessions in Anthropic-managed cloud environments instead of on your machine
* [Ultraplan](/en/ultraplan): launch a cloud planning session from your terminal and review the plan in your browser
* [Channels](/en/channels): forward Telegram, Discord, or iMessage into a session so Claude reacts to messages while you're away
* [Dispatch](/en/desktop#sessions-from-dispatch): message a task from your phone and it can spawn a Desktop session to handle it
* [Authentication](/en/authentication): set up `/login` and manage credentials for claude.ai
* [CLI reference](/en/cli-reference): full list of flags and commands including `claude remote-control`
* [Security](/en/security): how Remote Control sessions fit into the Claude Code security model
* [Data usage](/en/data-usage): what data flows through the Anthropic API during local and remote sessions
