# Push events into a running session with channels

Use channels to push messages, alerts, and webhooks into your Claude Code session from an MCP server. Forward CI results, chat messages, and monitoring events so Claude can react while you're away.

> Channels are in [research preview](#research-preview) and require Claude Code v2.1.80 or later. They require claude.ai login. Console and API key authentication is not supported. Team and Enterprise organizations must [explicitly enable them](#enterprise-controls).

A channel is an MCP server that pushes events into your running Claude Code session, so Claude can react to things that happen while you're not at the terminal. Channels can be two-way: Claude reads the event and replies back through the same channel, like a chat bridge. Events only arrive while the session is open, so for an always-on setup you run Claude in a background process or persistent terminal.

Unlike integrations that spawn a fresh cloud session or wait to be polled, the event arrives in the session you already have open: see [how channels compare](#how-channels-compare).

You install a channel as a plugin and configure it with your own credentials. Telegram, Discord, and iMessage are included in the research preview.

When Claude replies through a channel, you see the inbound message in your terminal but not the reply text. The terminal shows the tool call and a confirmation (like "sent"), and the actual reply appears on the other platform.

This page covers:

* [Supported channels](#supported-channels): Telegram, Discord, and iMessage setup
* [Install and run a channel](#quickstart) with fakechat, a localhost demo
* [Who can push messages](#security): sender allowlists and how you pair
* [Enable channels for your organization](#enterprise-controls) on Team and Enterprise
* [How channels compare](#how-channels-compare) to web sessions, Slack, MCP, and Remote Control

To build your own channel, see the [Channels reference](/en/channels-reference).

## Supported channels

Each supported channel is a plugin that requires [Bun](https://bun.sh). For a hands-on demo of the plugin flow before connecting a real platform, try the [fakechat quickstart](#quickstart).

### Telegram

View the full [Telegram plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/telegram).

1. **Create a Telegram bot**
   Open [BotFather](https://t.me/BotFather) in Telegram and send `/newbot`. Give it a display name and a unique username ending in `bot`. Copy the token BotFather returns.

2. **Install the plugin**
   In Claude Code, run:

   ```
   /plugin install telegram@claude-plugins-official
   ```

   If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

   After installing, run `/reload-plugins` to activate the plugin's configure command.

3. **Configure your token**
   Run the configure command with the token from BotFather:

   ```
   /telegram:configure <token>
   ```

   This saves it to `~/.claude/channels/telegram/.env`. You can also set `TELEGRAM_BOT_TOKEN` in your shell environment before launching Claude Code.

4. **Restart with channels enabled**
   Exit Claude Code and restart with the channel flag. This starts the Telegram plugin, which begins polling for messages from your bot:

   ```bash
   claude --channels plugin:telegram@claude-plugins-official
   ```

5. **Pair your account**
   Open Telegram and send any message to your bot. The bot replies with a pairing code.

   > If your bot doesn't respond, make sure Claude Code is running with `--channels` from the previous step. The bot can only reply while the channel is active.

   Back in Claude Code, run:

   ```
   /telegram:access pair <code>
   ```

   Then lock down access so only your account can send messages:

   ```
   /telegram:access policy allowlist
   ```

### Discord

View the full [Discord plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/discord).

1. **Create a Discord bot**
   Go to the [Discord Developer Portal](https://discord.com/developers/applications), click **New Application**, and name it. In the **Bot** section, create a username, then click **Reset Token** and copy the token.

2. **Enable Message Content Intent**
   In your bot's settings, scroll to **Privileged Gateway Intents** and enable **Message Content Intent**.

3. **Invite the bot to your server**
   Go to **OAuth2 > URL Generator**. Select the `bot` scope and enable these permissions:

   * View Channels
   * Send Messages
   * Send Messages in Threads
   * Read Message History
   * Attach Files
   * Add Reactions

   Open the generated URL to add the bot to your server.

4. **Install the plugin**
   In Claude Code, run:

   ```
   /plugin install discord@claude-plugins-official
   ```

   If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

   After installing, run `/reload-plugins` to activate the plugin's configure command.

5. **Configure your token**
   Run the configure command with the bot token you copied:

   ```
   /discord:configure <token>
   ```

   This saves it to `~/.claude/channels/discord/.env`. You can also set `DISCORD_BOT_TOKEN` in your shell environment before launching Claude Code.

6. **Restart with channels enabled**
   Exit Claude Code and restart with the channel flag. This connects the Discord plugin so your bot can receive and respond to messages:

   ```bash
   claude --channels plugin:discord@claude-plugins-official
   ```

7. **Pair your account**
   DM your bot on Discord. The bot replies with a pairing code.

   > If your bot doesn't respond, make sure Claude Code is running with `--channels` from the previous step. The bot can only reply while the channel is active.

   Back in Claude Code, run:

   ```
   /discord:access pair <code>
   ```

   Then lock down access so only your account can send messages:

   ```
   /discord:access policy allowlist
   ```

### iMessage

View the full [iMessage plugin source](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/imessage).

The iMessage channel reads your Messages database directly and sends replies through AppleScript. It requires macOS and needs no bot token or external service.

1. **Grant Full Disk Access**
   The Messages database at `~/Library/Messages/chat.db` is protected by macOS. The first time the server reads it, macOS prompts for access: click **Allow**. The prompt names whichever app launched Bun, such as Terminal, iTerm, or your IDE.

   If the prompt doesn't appear or you clicked Don't Allow, grant access manually under **System Settings > Privacy & Security > Full Disk Access** and add your terminal. Without this, the server exits immediately with `authorization denied`.

2. **Install the plugin**
   In Claude Code, run:

   ```
   /plugin install imessage@claude-plugins-official
   ```

   If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

3. **Restart with channels enabled**
   Exit Claude Code and restart with the channel flag:

   ```bash
   claude --channels plugin:imessage@claude-plugins-official
   ```

4. **Text yourself**
   Open Messages on any device signed into your Apple ID and send a message to yourself. It reaches Claude immediately: self-chat bypasses access control with no setup.

   > The first reply Claude sends triggers a macOS Automation prompt asking if your terminal can control Messages. Click **OK**.

5. **Allow other senders**
   By default, only your own messages pass through. To let another contact reach Claude, add their handle:

   ```
   /imessage:access allow +15551234567
   ```

   Handles are phone numbers in `+country` format or Apple ID emails like `user@example.com`.

You can also [build your own channel](/en/channels-reference) for systems that don't have a plugin yet.

## Quickstart

Fakechat is an officially supported demo channel that runs a chat UI on localhost, with nothing to authenticate and no external service to configure.

Once you install and enable fakechat, you can type in the browser and the message arrives in your Claude Code session. Claude replies, and the reply shows up back in the browser. After you've tested the fakechat interface, try out [Telegram](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/telegram), [Discord](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/discord), or [iMessage](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins/imessage).

To try the fakechat demo, you'll need:

* Claude Code [installed and authenticated](/en/quickstart#step-1-install-claude-code) with a claude.ai account
* [Bun](https://bun.sh) installed. The pre-built channel plugins are Bun scripts. Check with `bun --version`; if that fails, [install Bun](https://bun.sh/docs/installation).
* **Team/Enterprise users**: your organization admin must [enable channels](#enterprise-controls) in managed settings

1. **Install the fakechat channel plugin**
   Start a Claude Code session and run the install command:

   ```
   /plugin install fakechat@claude-plugins-official
   ```

   If Claude Code reports that the plugin is not found in any marketplace, your marketplace is either missing or outdated. Run `/plugin marketplace update claude-plugins-official` to refresh it, or `/plugin marketplace add anthropics/claude-plugins-official` if you haven't added it before. Then retry the install.

2. **Restart with the channel enabled**
   Exit Claude Code, then restart with `--channels` and pass the fakechat plugin you installed:

   ```bash
   claude --channels plugin:fakechat@claude-plugins-official
   ```

   The fakechat server starts automatically.

   > You can pass several plugins to `--channels`, space-separated.

3. **Push a message in**
   Open the fakechat UI at [http://localhost:8787](http://localhost:8787) and type a message:

   ```
   hey, what's in my working directory?
   ```

   The message arrives in your Claude Code session as a `<channel source="fakechat">` event. Claude reads it, does the work, and calls fakechat's `reply` tool. The answer shows up in the chat UI.

If Claude hits a permission prompt while you're away from the terminal, the session pauses until you respond. Channel servers that declare the [permission relay capability](/en/channels-reference#relay-permission-prompts) can forward these prompts to you so you can approve or deny remotely. For unattended use, [`--dangerously-skip-permissions`](/en/permission-modes#skip-all-checks-with-bypasspermissions-mode) bypasses prompts entirely, but only use it in environments you trust.

## Security

Every approved channel plugin maintains a sender allowlist: only IDs you've added can push messages, and everyone else is silently dropped.

Telegram and Discord bootstrap the list by pairing:

1. Find your bot in Telegram or Discord and send it any message
2. The bot replies with a pairing code
3. In your Claude Code session, approve the code when prompted
4. Your sender ID is added to the allowlist

iMessage works differently: texting yourself bypasses the gate automatically, and you add other contacts by handle with `/imessage:access allow`.

On top of that, you control which servers are enabled each session with `--channels`, and on Team and Enterprise plans your organization controls availability with [`channelsEnabled`](#enterprise-controls).

Being in `.mcp.json` isn't enough to push messages: a server also has to be named in `--channels`.

The allowlist also gates [permission relay](/en/channels-reference#relay-permission-prompts) if the channel declares it. Anyone who can reply through the channel can approve or deny tool use in your session, so only allowlist senders you trust with that authority.

## Enterprise controls

On Team and Enterprise plans, channels are off by default. Admins control availability through two [managed settings](/en/settings) that users cannot override:

| Setting                 | Purpose                                                                                                                                                                                                                                                     | When not configured            |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- |
| `channelsEnabled`       | Master switch. Must be `true` for any channel to deliver messages. Set via the [claude.ai Admin console](https://claude.ai/admin-settings/claude-code) toggle or directly in managed settings. Blocks all channels including the development flag when off. | Channels blocked               |
| `allowedChannelPlugins` | Which plugins can register once channels are enabled. Replaces the Anthropic-maintained list when set. Only applies when `channelsEnabled` is `true`.                                                                                                       | Anthropic default list applies |

Pro and Max users without an organization skip these checks entirely: channels are available and users opt in per session with `--channels`.

### Enable channels for your organization

Admins can enable channels from [**claude.ai → Admin settings → Claude Code → Channels**](https://claude.ai/admin-settings/claude-code), or by setting `channelsEnabled` to `true` in managed settings.

Once enabled, users in your organization can use `--channels` to opt channel servers into individual sessions. If the setting is disabled or unset, the MCP server still connects and its tools work, but channel messages won't arrive. A startup warning tells the user to have an admin enable the setting.

### Restrict which channel plugins can run

By default, any plugin on the Anthropic-maintained allowlist can register as a channel. Admins on Team and Enterprise plans can replace that allowlist with their own by setting `allowedChannelPlugins` in managed settings. Use this to restrict which official plugins are allowed, approve channels from your own internal marketplace, or both. Each entry names a plugin and the marketplace it comes from:

```json
{
  "channelsEnabled": true,
  "allowedChannelPlugins": [
    { "marketplace": "claude-plugins-official", "plugin": "telegram" },
    { "marketplace": "claude-plugins-official", "plugin": "discord" },
    { "marketplace": "acme-corp-plugins", "plugin": "internal-alerts" }
  ]
}
```

When `allowedChannelPlugins` is set, it replaces the Anthropic allowlist entirely: only the listed plugins can register. Leave it unset to fall back to the default Anthropic allowlist. An empty array blocks all channel plugins from the allowlist, but `--dangerously-load-development-channels` can still bypass it for local testing. To block channels entirely including the development flag, leave `channelsEnabled` unset instead.

This setting requires `channelsEnabled: true`. If a user passes a plugin to `--channels` that isn't on your list, Claude Code starts normally but the channel doesn't register, and the startup notice explains that the plugin isn't on the organization's approved list.

## Research preview

Channels are a research preview feature. Availability is rolling out gradually, and the `--channels` flag syntax and protocol contract may change based on feedback.

During the preview, `--channels` only accepts plugins from an Anthropic-maintained allowlist, or from your organization's allowlist if an admin has set [`allowedChannelPlugins`](#restrict-which-channel-plugins-can-run). The channel plugins in [claude-plugins-official](https://github.com/anthropics/claude-plugins-official/tree/main/external_plugins) are the default approved set. If you pass something that isn't on the effective allowlist, Claude Code starts normally but the channel doesn't register, and the startup notice tells you why.

To test a channel you're building, use `--dangerously-load-development-channels`. See [Test during the research preview](/en/channels-reference#test-during-the-research-preview) for information about testing custom channels that you build.

Report issues or feedback on the [Claude Code GitHub repository](https://github.com/anthropics/claude-code/issues).

## How channels compare

Several Claude Code features connect to systems outside the terminal, each suited to a different kind of work:

| Feature                                              | What it does                                                          | Good for                                                  |
| ---------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------- |
| [Claude Code on the web](/en/claude-code-on-the-web) | Runs tasks in a fresh cloud sandbox, cloned from GitHub               | Delegating self-contained async work you check on later   |
| [Claude in Slack](/en/slack)                         | Spawns a web session from an `@Claude` mention in a channel or thread | Starting tasks directly from team conversation context    |
| Standard [MCP server](/en/mcp)                       | Claude queries it during a task; nothing is pushed to the session     | Giving Claude on-demand access to read or query a system  |
| [Remote Control](/en/remote-control)                 | You drive your local session from claude.ai or the Claude mobile app  | Steering an in-progress session while away from your desk |

Channels fill the gap in that list by pushing events from non-Claude sources into your already-running local session.

* **Chat bridge**: ask Claude something from your phone via Telegram, Discord, or iMessage, and the answer comes back in the same chat while the work runs on your machine against your real files.
* **[Webhook receiver](/en/channels-reference#example-build-a-webhook-receiver)**: a webhook from CI, your error tracker, a deploy pipeline, or other external service arrives where Claude already has your files open and remembers what you were debugging.

## Next steps

Once you have a channel running, explore these related features:

* [Build your own channel](/en/channels-reference) for systems that don't have plugins yet
* [Remote Control](/en/remote-control) to drive a local session from your phone instead of forwarding events into it
* [Scheduled tasks](/en/scheduled-tasks) to poll on a timer instead of reacting to pushed events
