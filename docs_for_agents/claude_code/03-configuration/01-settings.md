> ## Documentation Index
> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Claude Code settings

> Configure Claude Code with global and project-level settings, and environment variables.

Claude Code offers a variety of settings to configure its behavior to meet your needs. You can configure Claude Code by running the `/config` command when using the interactive REPL, which opens a tabbed Settings interface where you can view status information and modify configuration options.

## Configuration scopes

Claude Code uses a **scope system** to determine where configurations apply and who they're shared with. Understanding scopes helps you decide how to configure Claude Code for personal use, team collaboration, or enterprise deployment.

### Available scopes

| Scope       | Location                                                                           | Who it affects                       | Shared with team?      |
| :---------- | :--------------------------------------------------------------------------------- | :----------------------------------- | :--------------------- |
| **Managed** | Server-managed settings, plist / registry, or system-level `managed-settings.json` | All users on the machine             | Yes (deployed by IT)   |
| **User**    | `~/.claude/` directory                                                             | You, across all projects             | No                     |
| **Project** | `.claude/` in repository                                                           | All collaborators on this repository | Yes (committed to git) |
| **Local**   | `.claude/settings.local.json`                                                      | You, in this repository only         | No (gitignored)        |

### When to use each scope

**Managed scope** is for:

* Security policies that must be enforced organization-wide
* Compliance requirements that can't be overridden
* Standardized configurations deployed by IT/DevOps

**User scope** is best for:

* Personal preferences you want everywhere (themes, editor settings)
* Tools and plugins you use across all projects
* API keys and authentication (stored securely)

**Project scope** is best for:

* Team-shared settings (permissions, hooks, MCP servers)
* Plugins the whole team should have
* Standardizing tooling across collaborators

**Local scope** is best for:

* Personal overrides for a specific project
* Testing configurations before sharing with the team
* Machine-specific settings that won't work for others

### How scopes interact

When the same setting is configured in multiple scopes, more specific scopes take precedence:

1. **Managed** (highest) - can't be overridden by anything
2. **Command line arguments** - temporary session overrides
3. **Local** - overrides project and user settings
4. **Project** - overrides user settings
5. **User** (lowest) - applies when nothing else specifies the setting

For example, if a permission is allowed in user settings but denied in project settings, the project setting takes precedence and the permission is blocked.

### What uses scopes

Scopes apply to many Claude Code features:

| Feature         | User location             | Project location                   | Local location                 |
| :-------------- | :------------------------ | :--------------------------------- | :----------------------------- |
| **Settings**    | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json`  |
| **Subagents**   | `~/.claude/agents/`       | `.claude/agents/`                  | None                           |
| **MCP servers** | `~/.claude.json`          | `.mcp.json`                        | `~/.claude.json` (per-project) |
| **Plugins**     | `~/.claude/settings.json` | `.claude/settings.json`            | `.claude/settings.local.json`  |
| **CLAUDE.md**   | `~/.claude/CLAUDE.md`     | `CLAUDE.md` or `.claude/CLAUDE.md` | `CLAUDE.local.md`              |

***

## Settings files

The `settings.json` file is the official mechanism for configuring Claude
Code through hierarchical settings:

* **User settings** are defined in `~/.claude/settings.json` and apply to all
  projects.
* **Project settings** are saved in your project directory:
  * `.claude/settings.json` for settings that are checked into source control and shared with your team
  * `.claude/settings.local.json` for settings that are not checked in, useful for personal preferences and experimentation. Claude Code will configure git to ignore `.claude/settings.local.json` when it is created.
* **Managed settings**: For organizations that need centralized control, Claude Code supports multiple delivery mechanisms for managed settings. All use the same JSON format and cannot be overridden by user or project settings:

  * **Server-managed settings**: delivered from Anthropic's servers via the Claude.ai admin console. See [server-managed settings](/en/server-managed-settings).
  * **MDM/OS-level policies**: delivered through native device management on macOS and Windows:
    * macOS: `com.anthropic.claudecode` managed preferences domain (deployed via configuration profiles in Jamf, Kandji, or other MDM tools)
    * Windows: `HKLM\SOFTWARE\Policies\ClaudeCode` registry key with a `Settings` value (REG\_SZ or REG\_EXPAND\_SZ) containing JSON (deployed via Group Policy or Intune)
    * Windows (user-level): `HKCU\SOFTWARE\Policies\ClaudeCode` (lowest policy priority, only used when no admin-level source exists)
  * **File-based**: `managed-settings.json` and `managed-mcp.json` deployed to system directories:

    * macOS: `/Library/Application Support/ClaudeCode/`
    * Linux and WSL: `/etc/claude-code/`
    * Windows: `C:\Program Files\ClaudeCode\`

    <Warning>
      The legacy Windows path `C:\ProgramData\ClaudeCode\managed-settings.json` is no longer supported as of v2.1.75. Administrators who deployed settings to that location must migrate files to `C:\Program Files\ClaudeCode\managed-settings.json`.
    </Warning>

    File-based managed settings also support a drop-in directory at `managed-settings.d/` in the same system directory alongside `managed-settings.json`. This lets separate teams deploy independent policy fragments without coordinating edits to a single file.

    Following the systemd convention, `managed-settings.json` is merged first as the base, then all `*.json` files in the drop-in directory are sorted alphabetically and merged on top. Later files override earlier ones for scalar values; arrays are concatenated and de-duplicated; objects are deep-merged. Hidden files starting with `.` are ignored.

    Use numeric prefixes to control merge order, for example `10-telemetry.json` and `20-security.json`.

  See [managed settings](/en/permissions#managed-only-settings) and [Managed MCP configuration](/en/mcp#managed-mcp-configuration) for details.

  <Note>
    Managed deployments can also restrict **plugin marketplace additions** using
    `strictKnownMarketplaces`. For more information, see [Managed marketplace restrictions](/en/plugin-marketplaces#managed-marketplace-restrictions).
  </Note>
* **Other configuration** is stored in `~/.claude.json`. This file contains your preferences (theme, notification settings, editor mode), OAuth session, [MCP server](/en/mcp) configurations for user and local scopes, per-project state (allowed tools, trust settings), and various caches. Project-scoped MCP servers are stored separately in `.mcp.json`.

<Note>
  Claude Code automatically creates timestamped backups of configuration files and retains the five most recent backups to prevent data loss.
</Note>

```JSON Example settings.json theme={null}
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl *)",
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)"
    ]
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp"
  },
  "companyAnnouncements": [
    "Welcome to Acme Corp! Review our code guidelines at docs.acme.com",
    "Reminder: Code reviews required for all PRs",
    "New security policy in effect"
  ]
}
```

The `$schema` line in the example above points to the [official JSON schema](https://json.schemastore.org/claude-code-settings.json) for Claude Code settings. Adding it to your `settings.json` enables autocomplete and inline validation in VS Code, Cursor, and any other editor that supports JSON schema validation.

### Available settings

`settings.json` supports a number of options:

| Key                               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Example                                                                                                                        |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| `agent`                           | Run the main thread as a named subagent. Applies that subagent's system prompt, tool restrictions, and model. See [Invoke subagents explicitly](/en/sub-agents#invoke-subagents-explicitly)                                                                                                                                                                                                                                                                                                                                                              | `"code-reviewer"`                                                                                                              |
| `allowedChannelPlugins`           | (Managed settings only) Allowlist of channel plugins that may push messages. Replaces the default Anthropic allowlist when set. Undefined = fall back to the default, empty array = block all channel plugins. Requires `channelsEnabled: true`. See [Restrict which channel plugins can run](/en/channels#restrict-which-channel-plugins-can-run)                                                                                                                                                                                                       | `[{ "marketplace": "claude-plugins-official", "plugin": "telegram" }]`                                                         |
| `allowedHttpHookUrls`             | Allowlist of URL patterns that HTTP hooks may target. Supports `*` as a wildcard. When set, hooks with non-matching URLs are blocked. Undefined = no restriction, empty array = block all HTTP hooks. Arrays merge across settings sources. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                | `["https://hooks.example.com/*"]`                                                                                              |
| `allowedMcpServers`               | When set in managed-settings.json, allowlist of MCP servers users can configure. Undefined = no restrictions, empty array = lockdown. Applies to all scopes. Denylist takes precedence. See [Managed MCP configuration](/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                                               | `[{ "serverName": "github" }]`                                                                                                 |
| `allowManagedHooksOnly`           | (Managed settings only) Prevent loading of user, project, and plugin hooks. Only allows managed hooks and SDK hooks. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                                                                                                                                       | `true`                                                                                                                         |
| `allowManagedMcpServersOnly`      | (Managed settings only) Only `allowedMcpServers` from managed settings are respected. `deniedMcpServers` still merges from all sources. Users can still add MCP servers, but only the admin-defined allowlist applies. See [Managed MCP configuration](/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                | `true`                                                                                                                         |
| `allowManagedPermissionRulesOnly` | (Managed settings only) Prevent user and project settings from defining `allow`, `ask`, or `deny` permission rules. Only rules in managed settings apply. See [Managed-only settings](/en/permissions#managed-only-settings)                                                                                                                                                                                                                                                                                                                             | `true`                                                                                                                         |
| `alwaysThinkingEnabled`           | Enable [extended thinking](/en/common-workflows#use-extended-thinking-thinking-mode) by default for all sessions. Typically configured via the `/config` command rather than editing directly                                                                                                                                                                                                                                                                                                                                                            | `true`                                                                                                                         |
| `apiKeyHelper`                    | Custom script, to be executed in `/bin/sh`, to generate an auth value. This value will be sent as `X-Api-Key` and `Authorization: Bearer` headers for model requests                                                                                                                                                                                                                                                                                                                                                                                     | `/bin/generate_temp_api_key.sh`                                                                                                |
| `attribution`                     | Customize attribution for git commits and pull requests. See [Attribution settings](#attribution-settings)                                                                                                                                                                                                                                                                                                                                                                                                                                               | `{"commit": "🤖 Generated with Claude Code", "pr": ""}`                                                                        |
| `autoMemoryDirectory`             | Custom directory for [auto memory](/en/memory#storage-location) storage. Accepts `~/`-expanded paths. Not accepted in project settings (`.claude/settings.json`) to prevent shared repos from redirecting memory writes to sensitive locations. Accepted from policy, local, and user settings                                                                                                                                                                                                                                                           | `"~/my-memory-dir"`                                                                                                            |
| `autoMode`                        | Customize what the [auto mode](/en/permission-modes#eliminate-prompts-with-auto-mode) classifier blocks and allows. Contains `environment`, `allow`, and `soft_deny` arrays of prose rules. See [Configure the auto mode classifier](/en/permissions#configure-the-auto-mode-classifier). Not read from shared project settings                                                                                                                                                                                                                          | `{"environment": ["Trusted repo: github.example.com/acme"]}`                                                                   |
| `autoUpdatesChannel`              | Release channel to follow for updates. Use `"stable"` for a version that is typically about one week old and skips versions with major regressions, or `"latest"` (default) for the most recent release                                                                                                                                                                                                                                                                                                                                                  | `"stable"`                                                                                                                     |
| `availableModels`                 | Restrict which models users can select via `/model`, `--model`, Config tool, or `ANTHROPIC_MODEL`. Does not affect the Default option. See [Restrict model selection](/en/model-config#restrict-model-selection)                                                                                                                                                                                                                                                                                                                                         | `["sonnet", "haiku"]`                                                                                                          |
| `awsAuthRefresh`                  | Custom script that modifies the `.aws` directory (see [advanced credential configuration](/en/amazon-bedrock#advanced-credential-configuration))                                                                                                                                                                                                                                                                                                                                                                                                         | `aws sso login --profile myprofile`                                                                                            |
| `awsCredentialExport`             | Custom script that outputs JSON with AWS credentials (see [advanced credential configuration](/en/amazon-bedrock#advanced-credential-configuration))                                                                                                                                                                                                                                                                                                                                                                                                     | `/bin/generate_aws_grant.sh`                                                                                                   |
| `blockedMarketplaces`             | (Managed settings only) Blocklist of marketplace sources. Blocked sources are checked before downloading, so they never touch the filesystem. See [Managed marketplace restrictions](/en/plugin-marketplaces#managed-marketplace-restrictions)                                                                                                                                                                                                                                                                                                           | `[{ "source": "github", "repo": "untrusted/plugins" }]`                                                                        |
| `channelsEnabled`                 | (Managed settings only) Allow [channels](/en/channels) for Team and Enterprise users. Unset or `false` blocks channel message delivery regardless of what users pass to `--channels`                                                                                                                                                                                                                                                                                                                                                                     | `true`                                                                                                                         |
| `cleanupPeriodDays`               | Sessions inactive for longer than this period are deleted at startup (default: 30 days, minimum 1). Setting to `0` is rejected with a validation error. Also controls the age cutoff for automatic removal of [orphaned subagent worktrees](/en/common-workflows#worktree-cleanup) at startup. To disable transcript writes entirely in non-interactive mode (`-p`), use the `--no-session-persistence` flag or the `persistSession: false` SDK option; there is no interactive-mode equivalent.                                                         | `20`                                                                                                                           |
| `companyAnnouncements`            | Announcement to display to users at startup. If multiple announcements are provided, they will be cycled through at random.                                                                                                                                                                                                                                                                                                                                                                                                                              | `["Welcome to Acme Corp! Review our code guidelines at docs.acme.com"]`                                                        |
| `defaultShell`                    | Default shell for input-box `!` commands. Accepts `"bash"` (default) or `"powershell"`. Setting `"powershell"` routes interactive `!` commands through PowerShell on Windows. Requires `CLAUDE_CODE_USE_POWERSHELL_TOOL=1`. See [PowerShell tool](/en/tools-reference#powershell-tool)                                                                                                                                                                                                                                                                   | `"powershell"`                                                                                                                 |
| `deniedMcpServers`                | When set in managed-settings.json, denylist of MCP servers that are explicitly blocked. Applies to all scopes including managed servers. Denylist takes precedence over allowlist. See [Managed MCP configuration](/en/mcp#managed-mcp-configuration)                                                                                                                                                                                                                                                                                                    | `[{ "serverName": "filesystem" }]`                                                                                             |
| `disableAllHooks`                 | Disable all [hooks](/en/hooks) and any custom [status line](/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `true`                                                                                                                         |
| `disableAutoMode`                 | Set to `"disable"` to prevent [auto mode](/en/permission-modes#eliminate-prompts-with-auto-mode) from being activated. Removes `auto` from the `Shift+Tab` cycle and rejects `--permission-mode auto` at startup. Most useful in [managed settings](/en/permissions#managed-settings) where users cannot override it                                                                                                                                                                                                                                     | `"disable"`                                                                                                                    |
| `disableDeepLinkRegistration`     | Set to `"disable"` to prevent Claude Code from registering the `claude-cli://` protocol handler with the operating system on startup. Deep links let external tools open a Claude Code session with a pre-filled prompt via `claude-cli://open?q=...`. The `q` parameter supports multi-line prompts using URL-encoded newlines (`%0A`). Useful in environments where protocol handler registration is restricted or managed separately                                                                                                                  | `"disable"`                                                                                                                    |
| `disabledMcpjsonServers`          | List of specific MCP servers from `.mcp.json` files to reject                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `["filesystem"]`                                                                                                               |
| `disableSkillShellExecution`      | Disable inline shell execution for `` !`...` `` and ` ```! ` blocks in [skills](/en/skills) and custom commands from user, project, plugin, or additional-directory sources. Commands are replaced with `[shell command execution disabled by policy]` instead of being run. Bundled and managed skills are not affected. Most useful in [managed settings](/en/permissions#managed-settings) where users cannot override it                                                                                                                             | `true`                                                                                                                         |
| `effortLevel`                     | Persist the [effort level](/en/model-config#adjust-effort-level) across sessions. Accepts `"low"`, `"medium"`, or `"high"`. Written automatically when you run `/effort low`, `/effort medium`, or `/effort high`. Supported on Opus 4.6 and Sonnet 4.6                                                                                                                                                                                                                                                                                                  | `"medium"`                                                                                                                     |
| `enableAllProjectMcpServers`      | Automatically approve all MCP servers defined in project `.mcp.json` files                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `true`                                                                                                                         |
| `enabledMcpjsonServers`           | List of specific MCP servers from `.mcp.json` files to approve                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `["memory", "github"]`                                                                                                         |
| `env`                             | Environment variables that will be applied to every session                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `{"FOO": "bar"}`                                                                                                               |
| `fastModePerSessionOptIn`         | When `true`, fast mode does not persist across sessions. Each session starts with fast mode off, requiring users to enable it with `/fast`. The user's fast mode preference is still saved. See [Require per-session opt-in](/en/fast-mode#require-per-session-opt-in)                                                                                                                                                                                                                                                                                   | `true`                                                                                                                         |
| `feedbackSurveyRate`              | Probability (0–1) that the [session quality survey](/en/data-usage#session-quality-surveys) appears when eligible. Set to `0` to suppress entirely. Useful when using Bedrock, Vertex, or Foundry where the default sample rate does not apply                                                                                                                                                                                                                                                                                                           | `0.05`                                                                                                                         |
| `fileSuggestion`                  | Configure a custom script for `@` file autocomplete. See [File suggestion settings](#file-suggestion-settings)                                                                                                                                                                                                                                                                                                                                                                                                                                           | `{"type": "command", "command": "~/.claude/file-suggestion.sh"}`                                                               |
| `forceLoginMethod`                | Use `claudeai` to restrict login to Claude.ai accounts, `console` to restrict login to Claude Console (API usage billing) accounts                                                                                                                                                                                                                                                                                                                                                                                                                       | `claudeai`                                                                                                                     |
| `forceLoginOrgUUID`               | Require login to belong to a specific organization. Accepts a single UUID string, which also pre-selects that organization during login, or an array of UUIDs where any listed organization is accepted without pre-selection. When set in managed settings, login fails if the authenticated account does not belong to a listed organization; an empty array fails closed and blocks login with a misconfiguration message                                                                                                                             | `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` or `["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"]` |
| `forceRemoteSettingsRefresh`      | (Managed settings only) Block CLI startup until remote managed settings are freshly fetched from the server. If the fetch fails, the CLI exits rather than continuing with cached or no settings. When not set, startup continues without waiting for remote settings. See [fail-closed enforcement](/en/server-managed-settings#enforce-fail-closed-startup)                                                                                                                                                                                            | `true`                                                                                                                         |
| `hooks`                           | Configure custom commands to run at lifecycle events. See [hooks documentation](/en/hooks) for format                                                                                                                                                                                                                                                                                                                                                                                                                                                    | See [hooks](/en/hooks)                                                                                                         |
| `httpHookAllowedEnvVars`          | Allowlist of environment variable names HTTP hooks may interpolate into headers. When set, each hook's effective `allowedEnvVars` is the intersection with this list. Undefined = no restriction. Arrays merge across settings sources. See [Hook configuration](#hook-configuration)                                                                                                                                                                                                                                                                    | `["MY_TOKEN", "HOOK_SECRET"]`                                                                                                  |
| `includeCoAuthoredBy`             | **Deprecated**: Use `attribution` instead. Whether to include the `co-authored-by Claude` byline in git commits and pull requests (default: `true`)                                                                                                                                                                                                                                                                                                                                                                                                      | `false`                                                                                                                        |
| `includeGitInstructions`          | Include built-in commit and PR workflow instructions and the git status snapshot in Claude's system prompt (default: `true`). Set to `false` to remove both, for example when using your own git workflow skills. The `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS` environment variable takes precedence over this setting when set                                                                                                                                                                                                                            | `false`                                                                                                                        |
| `language`                        | Configure Claude's preferred response language (e.g., `"japanese"`, `"spanish"`, `"french"`). Claude will respond in this language by default. Also sets the [voice dictation](/en/voice-dictation#change-the-dictation-language) language                                                                                                                                                                                                                                                                                                               | `"japanese"`                                                                                                                   |
| `model`                           | Override the default model to use for Claude Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `"claude-sonnet-4-6"`                                                                                                          |
| `modelOverrides`                  | Map Anthropic model IDs to provider-specific model IDs such as Bedrock inference profile ARNs. Each model picker entry uses its mapped value when calling the provider API. See [Override model IDs per version](/en/model-config#override-model-ids-per-version)                                                                                                                                                                                                                                                                                        | `{"claude-opus-4-6": "arn:aws:bedrock:..."}`                                                                                   |
| `otelHeadersHelper`               | Script to generate dynamic OpenTelemetry headers. Runs at startup and periodically (see [Dynamic headers](/en/monitoring-usage#dynamic-headers))                                                                                                                                                                                                                                                                                                                                                                                                         | `/bin/generate_otel_headers.sh`                                                                                                |
| `outputStyle`                     | Configure an output style to adjust the system prompt. See [output styles documentation](/en/output-styles)                                                                                                                                                                                                                                                                                                                                                                                                                                              | `"Explanatory"`                                                                                                                |
| `permissions`                     | See table below for structure of permissions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                                                |
| `plansDirectory`                  | Customize where plan files are stored. Path is relative to project root. Default: `~/.claude/plans`                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `"./plans"`                                                                                                                    |
| `pluginTrustMessage`              | (Managed settings only) Custom message appended to the plugin trust warning shown before installation. Use this to add organization-specific context, for example to confirm that plugins from your internal marketplace are vetted.                                                                                                                                                                                                                                                                                                                     | `"All plugins from our marketplace are approved by IT"`                                                                        |
| `prefersReducedMotion`            | Reduce or disable UI animations (spinners, shimmer, flash effects) for accessibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `true`                                                                                                                         |
| `respectGitignore`                | Control whether the `@` file picker respects `.gitignore` patterns. When `true` (default), files matching `.gitignore` patterns are excluded from suggestions                                                                                                                                                                                                                                                                                                                                                                                            | `false`                                                                                                                        |
| `showClearContextOnPlanAccept`    | Show the "clear context" option on the plan accept screen. Defaults to `false`. Set to `true` to restore the option                                                                                                                                                                                                                                                                                                                                                                                                                                      | `true`                                                                                                                         |
| `showThinkingSummaries`           | Show [extended thinking](/en/common-workflows#use-extended-thinking-thinking-mode) summaries in interactive sessions. When unset or `false` (default in interactive mode), thinking blocks are redacted by the API and shown as a collapsed stub. Redaction only changes what you see, not what the model generates: to reduce thinking spend, [lower the budget or disable thinking](/en/common-workflows#use-extended-thinking-thinking-mode) instead. Non-interactive mode (`-p`) and SDK callers always receive summaries regardless of this setting | `true`                                                                                                                         |
| `spinnerTipsEnabled`              | Show tips in the spinner while Claude is working. Set to `false` to disable tips (default: `true`)                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `false`                                                                                                                        |
| `spinnerTipsOverride`             | Override spinner tips with custom strings. `tips`: array of tip strings. `excludeDefault`: if `true`, only show custom tips; if `false` or absent, custom tips are merged with built-in tips                                                                                                                                                                                                                                                                                                                                                             | `{ "excludeDefault": true, "tips": ["Use our internal tool X"] }`                                                              |
| `spinnerVerbs`                    | Customize the action verbs shown in the spinner and turn duration messages. Set `mode` to `"replace"` to use only your verbs, or `"append"` to add them to the defaults                                                                                                                                                                                                                                                                                                                                                                                  | `{"mode": "append", "verbs": ["Pondering", "Crafting"]}`                                                                       |
| `statusLine`                      | Configure a custom status line to display context. See [`statusLine` documentation](/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `{"type": "command", "command": "~/.claude/statusline.sh"}`                                                                    |
| `strictKnownMarketplaces`         | (Managed settings only) Allowlist of plugin marketplaces users can add. Undefined = no restrictions, empty array = lockdown. Applies to marketplace additions only. See [Managed marketplace restrictions](/en/plugin-marketplaces#managed-marketplace-restrictions)                                                                                                                                                                                                                                                                                     | `[{ "source": "github", "repo": "acme-corp/plugins" }]`                                                                        |
| `useAutoModeDuringPlan`           | Whether plan mode uses auto mode semantics when auto mode is available. Default: `true`. Not read from shared project settings. Appears in `/config` as "Use auto mode during plan"                                                                                                                                                                                                                                                                                                                                                                      | `false`                                                                                                                        |
| `voiceEnabled`                    | Enable push-to-talk [voice dictation](/en/voice-dictation). Written automatically when you run `/voice`. Requires a Claude.ai account                                                                                                                                                                                                                                                                                                                                                                                                                    | `true`                                                                                                                         |

### Global config settings

These settings are stored in `~/.claude.json` rather than `settings.json`. Adding them to `settings.json` will trigger a schema validation error.

| Key                          | Description                                                                                                                                                                                                                                                                                                          | Example        |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| `autoConnectIde`             | Automatically connect to a running IDE when Claude Code starts from an external terminal. Default: `false`. Appears in `/config` as **Auto-connect to IDE (external terminal)** when running outside a VS Code or JetBrains terminal                                                                                 | `true`         |
| `autoInstallIdeExtension`    | Automatically install the Claude Code IDE extension when running from a VS Code terminal. Default: `true`. Appears in `/config` as **Auto-install IDE extension** when running inside a VS Code or JetBrains terminal. You can also set the [`CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL`](/en/env-vars) environment variable | `false`        |
| `editorMode`                 | Key binding mode for the input prompt: `"normal"` or `"vim"`. Default: `"normal"`. Appears in `/config` as **Editor mode**                                                                                                                                                                                           | `"vim"`        |
| `showTurnDuration`           | Show turn duration messages after responses, e.g. "Cooked for 1m 6s". Default: `true`. Appears in `/config` as **Show turn duration**                                                                                                                                                                                | `false`        |
| `terminalProgressBarEnabled` | Show the terminal progress bar in supported terminals: ConEmu, Ghostty 1.2.0+, and iTerm2 3.6.6+. Default: `true`. Appears in `/config` as **Terminal progress bar**                                                                                                                                                 | `false`        |
| `teammateMode`               | How [agent team](/en/agent-teams) teammates display: `auto` (picks split panes in tmux or iTerm2, in-process otherwise), `in-process`, or `tmux`. See [choose a display mode](/en/agent-teams#choose-a-display-mode)                                                                                                 | `"in-process"` |

### Worktree settings

Configure how `--worktree` creates and manages git worktrees. Use these settings to reduce disk usage and startup time in large monorepos.

| Key                           | Description                                                                                                                                                  | Example                               |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------ |
| `worktree.symlinkDirectories` | Directories to symlink from the main repository into each worktree to avoid duplicating large directories on disk. No directories are symlinked by default   | `["node_modules", ".cache"]`          |
| `worktree.sparsePaths`        | Directories to check out in each worktree via git sparse-checkout (cone mode). Only the listed paths are written to disk, which is faster in large monorepos | `["packages/my-app", "shared/utils"]` |

To copy gitignored files like `.env` into new worktrees, use a [`.worktreeinclude` file](/en/common-workflows#copy-gitignored-files-to-worktrees) in your project root instead of a setting.

### Permission settings

| Keys                                | Description                                                                                                                                                                                                                                                                            | Example                                                                |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| `allow`                             | Array of permission rules to allow tool use. See [Permission rule syntax](#permission-rule-syntax) below for pattern matching details                                                                                                                                                  | `[ "Bash(git diff *)" ]`                                               |
| `ask`                               | Array of permission rules to ask for confirmation upon tool use. See [Permission rule syntax](#permission-rule-syntax) below                                                                                                                                                           | `[ "Bash(git push *)" ]`                                               |
| `deny`                              | Array of permission rules to deny tool use. Use this to exclude sensitive files from Claude Code access. See [Permission rule syntax](#permission-rule-syntax) and [Bash permission limitations](/en/permissions#tool-specific-permission-rules)                                       | `[ "WebFetch", "Bash(curl *)", "Read(./.env)", "Read(./secrets/**)" ]` |
| `additionalDirectories`             | Additional [working directories](/en/permissions#working-directories) for file access. Most `.claude/` configuration is [not discovered](/en/permissions#additional-directories-grant-file-access-not-configuration) from these directories                                            | `[ "../docs/" ]`                                                       |
| `defaultMode`                       | Default [permission mode](/en/permission-modes) when opening Claude Code. Valid values: `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions`. The `--permission-mode` CLI flag overrides this setting for a single session                                         | `"acceptEdits"`                                                        |
| `disableBypassPermissionsMode`      | Set to `"disable"` to prevent `bypassPermissions` mode from being activated. This disables the `--dangerously-skip-permissions` command-line flag. Typically placed in [managed settings](/en/permissions#managed-settings) to enforce organizational policy, but works from any scope | `"disable"`                                                            |
| `skipDangerousModePermissionPrompt` | Skip the confirmation prompt shown before entering bypass permissions mode via `--dangerously-skip-permissions` or `defaultMode: "bypassPermissions"`. Ignored when set in project settings (`.claude/settings.json`) to prevent untrusted repositories from auto-bypassing the prompt | `true`                                                                 |

### Permission rule syntax

Permission rules follow the format `Tool` or `Tool(specifier)`. Rules are evaluated in order: deny rules first, then ask, then allow. The first matching rule wins.

Quick examples:

| Rule                           | Effect                                   |
| :----------------------------- | :--------------------------------------- |
| `Bash`                         | Matches all Bash commands                |
| `Bash(npm run *)`              | Matches commands starting with `npm run` |
| `Read(./.env)`                 | Matches reading the `.env` file          |
| `WebFetch(domain:example.com)` | Matches fetch requests to example.com    |

For the complete rule syntax reference, including wildcard behavior, tool-specific patterns for Read, Edit, WebFetch, MCP, and Agent rules, and security limitations of Bash patterns, see [Permission rule syntax](/en/permissions#permission-rule-syntax).

### Sandbox settings

Configure advanced sandboxing behavior. Sandboxing isolates bash commands from your filesystem and network. See [Sandboxing](/en/sandboxing) for details.

| Keys                                   | Description                                                                                                                                                                                                                                                                                                                                     | Example                         |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------ |
| `enabled`                              | Enable bash sandboxing (macOS, Linux, and WSL2). Default: false                                                                                                                                                                                                                                                                                 | `true`                          |
| `failIfUnavailable`                    | Exit with an error at startup if `sandbox.enabled` is true but the sandbox cannot start (missing dependencies, unsupported platform, or platform restrictions). When false (default), a warning is shown and commands run unsandboxed. Intended for managed settings deployments that require sandboxing as a hard gate                         | `true`                          |
| `autoAllowBashIfSandboxed`             | Auto-approve bash commands when sandboxed. Default: true                                                                                                                                                                                                                                                                                        | `true`                          |
| `excludedCommands`                     | Commands that should run outside of the sandbox                                                                                                                                                                                                                                                                                                 | `["docker *"]`                  |
| `allowUnsandboxedCommands`             | Allow commands to run outside the sandbox via the `dangerouslyDisableSandbox` parameter. When set to `false`, the `dangerouslyDisableSandbox` escape hatch is completely disabled and all commands must run sandboxed (or be in `excludedCommands`). Useful for enterprise policies that require strict sandboxing. Default: true               | `false`                         |
| `filesystem.allowWrite`                | Additional paths where sandboxed commands can write. Arrays are merged across all settings scopes: user, project, and managed paths are combined, not replaced. Also merged with paths from `Edit(...)` allow permission rules. See [path prefixes](#sandbox-path-prefixes) below.                                                              | `["/tmp/build", "~/.kube"]`     |
| `filesystem.denyWrite`                 | Paths where sandboxed commands cannot write. Arrays are merged across all settings scopes. Also merged with paths from `Edit(...)` deny permission rules.                                                                                                                                                                                       | `["/etc", "/usr/local/bin"]`    |
| `filesystem.denyRead`                  | Paths where sandboxed commands cannot read. Arrays are merged across all settings scopes. Also merged with paths from `Read(...)` deny permission rules.                                                                                                                                                                                        | `["~/.aws/credentials"]`        |
| `filesystem.allowRead`                 | Paths to re-allow reading within `denyRead` regions. Takes precedence over `denyRead`. Arrays are merged across all settings scopes. Use this to create workspace-only read access patterns.                                                                                                                                                    | `["."]`                         |
| `filesystem.allowManagedReadPathsOnly` | (Managed settings only) Only `filesystem.allowRead` paths from managed settings are respected. `denyRead` still merges from all sources. Default: false                                                                                                                                                                                         | `true`                          |
| `network.allowUnixSockets`             | Unix socket paths accessible in sandbox (for SSH agents, etc.)                                                                                                                                                                                                                                                                                  | `["~/.ssh/agent-socket"]`       |
| `network.allowAllUnixSockets`          | Allow all Unix socket connections in sandbox. Default: false                                                                                                                                                                                                                                                                                    | `true`                          |
| `network.allowLocalBinding`            | Allow binding to localhost ports (macOS only). Default: false                                                                                                                                                                                                                                                                                   | `true`                          |
| `network.allowedDomains`               | Array of domains to allow for outbound network traffic. Supports wildcards (e.g., `*.example.com`).                                                                                                                                                                                                                                             | `["github.com", "*.npmjs.org"]` |
| `network.allowManagedDomainsOnly`      | (Managed settings only) Only `allowedDomains` and `WebFetch(domain:...)` allow rules from managed settings are respected. Domains from user, project, and local settings are ignored. Non-allowed domains are blocked automatically without prompting the user. Denied domains are still respected from all sources. Default: false             | `true`                          |
| `network.httpProxyPort`                | HTTP proxy port used if you wish to bring your own proxy. If not specified, Claude will run its own proxy.                                                                                                                                                                                                                                      | `8080`                          |
| `network.socksProxyPort`               | SOCKS5 proxy port used if you wish to bring your own proxy. If not specified, Claude will run its own proxy.                                                                                                                                                                                                                                    | `8081`                          |
| `enableWeakerNestedSandbox`            | Enable weaker sandbox for unprivileged Docker environments (Linux and WSL2 only). **Reduces security.** Default: false                                                                                                                                                                                                                          | `true`                          |
| `enableWeakerNetworkIsolation`         | (macOS only) Allow access to the system TLS trust service (`com.apple.trustd.agent`) in the sandbox. Required for Go-based tools like `gh`, `gcloud`, and `terraform` to verify TLS certificates when using `httpProxyPort` with a MITM proxy and custom CA. **Reduces security** by opening a potential data exfiltration path. Default: false | `true`                          |

#### Sandbox path prefixes

Paths in `filesystem.allowWrite`, `filesystem.denyWrite`, `filesystem.denyRead`, and `filesystem.allowRead` support these prefixes:

| Prefix            | Meaning                                                                                | Example                                                                   |
| :---------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| `/`               | Absolute path from filesystem root                                                     | `/tmp/build` stays `/tmp/build`                                           |
| `~/`              | Relative to home directory                                                             | `~/.kube` becomes `$HOME/.kube`                                           |
| `./` or no prefix | Relative to the project root for project settings, or to `~/.claude` for user settings | `./output` in `.claude/settings.json` resolves to `<project-root>/output` |

The older `//path` prefix for absolute paths still works. If you previously used single-slash `/path` expecting project-relative resolution, switch to `./path`. This syntax differs from [Read and Edit permission rules](/en/permissions#read-and-edit), which use `//path` for absolute and `/path` for project-relative. Sandbox filesystem paths use standard conventions: `/tmp/build` is an absolute path.

**Configuration example:**

```json  theme={null}
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "excludedCommands": ["docker *"],
    "filesystem": {
      "allowWrite": ["/tmp/build", "~/.kube"],
      "denyRead": ["~/.aws/credentials"]
    },
    "network": {
      "allowedDomains": ["github.com", "*.npmjs.org", "registry.yarnpkg.com"],
      "allowUnixSockets": [
        "/var/run/docker.sock"
      ],
      "allowLocalBinding": true
    }
  }
}
```

**Filesystem and network restrictions** can be configured in two ways that are merged together:

* **`sandbox.filesystem` settings** (shown above): Control paths at the OS-level sandbox boundary. These restrictions apply to all subprocess commands (e.g., `kubectl`, `terraform`, `npm`), not just Claude's file tools.
* **Permission rules**: Use `Edit` allow/deny rules to control Claude's file tool access, `Read` deny rules to block reads, and `WebFetch` allow/deny rules to control network domains. Paths from these rules are also merged into the sandbox configuration.

### Attribution settings

Claude Code adds attribution to git commits and pull requests. These are configured separately:

* Commits use [git trailers](https://git-scm.com/docs/git-interpret-trailers) (like `Co-Authored-By`) by default,  which can be customized or disabled
* Pull request descriptions are plain text

| Keys     | Description                                                                                |
| :------- | :----------------------------------------------------------------------------------------- |
| `commit` | Attribution for git commits, including any trailers. Empty string hides commit attribution |
| `pr`     | Attribution for pull request descriptions. Empty string hides pull request attribution     |

**Default commit attribution:**

```text  theme={null}
🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Default pull request attribution:**

```text  theme={null}
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**Example:**

```json  theme={null}
{
  "attribution": {
    "commit": "Generated with AI\n\nCo-Authored-By: AI <ai@example.com>",
    "pr": ""
  }
}
```

<Note>
  The `attribution` setting takes precedence over the deprecated `includeCoAuthoredBy` setting. To hide all attribution, set `commit` and `pr` to empty strings.
</Note>

### File suggestion settings

Configure a custom command for `@` file path autocomplete. The built-in file suggestion uses fast filesystem traversal, but large monorepos may benefit from project-specific indexing such as a pre-built file index or custom tooling.

```json  theme={null}
{
  "fileSuggestion": {
    "type": "command",
    "command": "~/.claude/file-suggestion.sh"
  }
}
```

The command runs with the same environment variables as [hooks](/en/hooks), including `CLAUDE_PROJECT_DIR`. It receives JSON via stdin with a `query` field:

```json  theme={null}
{"query": "src/comp"}
```

Output newline-separated file paths to stdout (currently limited to 15):

```text  theme={null}
src/components/Button.tsx
src/components/Modal.tsx
src/components/Form.tsx
```

**Example:**

```bash  theme={null}
#!/bin/bash
query=$(cat | jq -r '.query')
your-repo-file-index --query "$query" | head -20
```

### Hook configuration

These settings control which hooks are allowed to run and what HTTP hooks can access. The `allowManagedHooksOnly` setting can only be configured in [managed settings](#settings-files). The URL and env var allowlists can be set at any settings level and merge across sources.

**Behavior when `allowManagedHooksOnly` is `true`:**

* Managed hooks and SDK hooks are loaded
* User hooks, project hooks, and plugin hooks are blocked

**Restrict HTTP hook URLs:**

Limit which URLs HTTP hooks can target. Supports `*` as a wildcard for matching. When the array is defined, HTTP hooks targeting non-matching URLs are silently blocked.

```json  theme={null}
{
  "allowedHttpHookUrls": ["https://hooks.example.com/*", "http://localhost:*"]
}
```

**Restrict HTTP hook environment variables:**

Limit which environment variable names HTTP hooks can interpolate into header values. Each hook's effective `allowedEnvVars` is the intersection of its own list and this setting.

```json  theme={null}
{
  "httpHookAllowedEnvVars": ["MY_TOKEN", "HOOK_SECRET"]
}
```

### Settings precedence

Settings apply in order of precedence. From highest to lowest:

1. **Managed settings** ([server-managed](/en/server-managed-settings), [MDM/OS-level policies](#configuration-scopes), or [managed settings](/en/settings#settings-files))
   * Policies deployed by IT through server delivery, MDM configuration profiles, registry policies, or managed settings files
   * Cannot be overridden by any other level, including command line arguments
   * Within the managed tier, precedence is: server-managed > MDM/OS-level policies > file-based (`managed-settings.d/*.json` + `managed-settings.json`) > HKCU registry (Windows only). Only one managed source is used; sources do not merge across tiers. Within the file-based tier, drop-in files and the base file are merged together.

2. **Command line arguments**
   * Temporary overrides for a specific session

3. **Local project settings** (`.claude/settings.local.json`)
   * Personal project-specific settings

4. **Shared project settings** (`.claude/settings.json`)
   * Team-shared project settings in source control

5. **User settings** (`~/.claude/settings.json`)
   * Personal global settings

This hierarchy ensures that organizational policies are always enforced while still allowing teams and individuals to customize their experience. The same precedence applies whether you run Claude Code from the CLI, the [VS Code extension](/en/vs-code), or a [JetBrains IDE](/en/jetbrains).

For example, if your user settings allow `Bash(npm run *)` but a project's shared settings deny it, the project setting takes precedence and the command is blocked.

<Note>
  **Array settings merge across scopes.** When the same array-valued setting (such as `sandbox.filesystem.allowWrite` or `permissions.allow`) appears in multiple scopes, the arrays are **concatenated and deduplicated**, not replaced. This means lower-priority scopes can add entries without overriding those set by higher-priority scopes, and vice versa. For example, if managed settings set `allowWrite` to `["/opt/company-tools"]` and a user adds `["~/.kube"]`, both paths are included in the final configuration.
</Note>

### Verify active settings

Run `/status` inside Claude Code to see which settings sources are active and where they come from. The output shows each configuration layer (managed, user, project) along with its origin, such as `Enterprise managed settings (remote)`, `Enterprise managed settings (plist)`, `Enterprise managed settings (HKLM)`, or `Enterprise managed settings (file)`. If a settings file contains errors, `/status` reports the issue so you can fix it.

### Key points about the configuration system

* **Memory files (`CLAUDE.md`)**: Contain instructions and context that Claude loads at startup
* **Settings files (JSON)**: Configure permissions, environment variables, and tool behavior
* **Skills**: Custom prompts that can be invoked with `/skill-name` or loaded by Claude automatically
* **MCP servers**: Extend Claude Code with additional tools and integrations
* **Precedence**: Higher-level configurations (Managed) override lower-level ones (User/Project)
* **Inheritance**: Settings are merged, with more specific settings adding to or overriding broader ones

### System prompt

Claude Code's internal system prompt is not published. To add custom instructions, use `CLAUDE.md` files or the `--append-system-prompt` flag.

### Excluding sensitive files

To prevent Claude Code from accessing files containing sensitive information like API keys, secrets, and environment files, use the `permissions.deny` setting in your `.claude/settings.json` file:

```json  theme={null}
{
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)",
      "Read(./config/credentials.json)",
      "Read(./build)"
    ]
  }
}
```

This replaces the deprecated `ignorePatterns` configuration. Files matching these patterns are excluded from file discovery and search results, and read operations on these files are denied.

## Subagent configuration

Claude Code supports custom AI subagents that can be configured at both user and project levels. These subagents are stored as Markdown files with YAML frontmatter:

* **User subagents**: `~/.claude/agents/` - Available across all your projects
* **Project subagents**: `.claude/agents/` - Specific to your project and can be shared with your team

Subagent files define specialized AI assistants with custom prompts and tool permissions. Learn more about creating and using subagents in the [subagents documentation](/en/sub-agents).

## Plugin configuration

Claude Code supports a plugin system that lets you extend functionality with skills, agents, hooks, and MCP servers. Plugins are distributed through marketplaces and can be configured at both user and repository levels.

### Plugin settings

Plugin-related settings in `settings.json`:

```json  theme={null}
{
  "enabledPlugins": {
    "formatter@acme-tools": true,
    "deployer@acme-tools": true,
    "analyzer@security-plugins": false
  },
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": "github",
      "repo": "acme-corp/claude-plugins"
    }
  }
}
```

#### `enabledPlugins`

Controls which plugins are enabled. Format: `"plugin-name@marketplace-name": true/false`

**Scopes**:

* **User settings** (`~/.claude/settings.json`): Personal plugin preferences
* **Project settings** (`.claude/settings.json`): Project-specific plugins shared with team
* **Local settings** (`.claude/settings.local.json`): Per-machine overrides (not committed)
* **Managed settings** (`managed-settings.json`): Organization-wide policy overrides that block installation at all scopes and hide the plugin from the marketplace

**Example**:

```json  theme={null}
{
  "enabledPlugins": {
    "code-formatter@team-tools": true,
    "deployment-tools@team-tools": true,
    "experimental-features@personal": false
  }
}
```

#### `extraKnownMarketplaces`

Defines additional marketplaces that should be made available for the repository. Typically used in repository-level settings to ensure team members have access to required plugin sources.

**When a repository includes `extraKnownMarketplaces`**:

1. Team members are prompted to install the marketplace when they trust the folder
2. Team members are then prompted to install plugins from that marketplace
3. Users can skip unwanted marketplaces or plugins (stored in user settings)
4. Installation respects trust boundaries and requires explicit consent

**Example**:

```json  theme={null}
{
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": {
        "source": "github",
        "repo": "acme-corp/claude-plugins"
      }
    },
    "security-plugins": {
      "source": {
        "source": "git",
        "url": "https://git.example.com/security/plugins.git"
      }
    }
  }
}
```

**Marketplace source types**:

* `github`: GitHub repository (uses `repo`)
* `git`: Any git URL (uses `url`)
* `directory`: Local filesystem path (uses `path`, for development only)
* `hostPattern`: regex pattern to match marketplace hosts (uses `hostPattern`)
* `settings`: inline marketplace declared directly in settings.json without a separate hosted repository (uses `name` and `plugins`)

Use `source: 'settings'` to declare a small set of plugins inline without setting up a hosted marketplace repository. Plugins listed here must reference external sources such as GitHub or npm. You still need to enable each plugin separately in `enabledPlugins`.

```json  theme={null}
{
  "extraKnownMarketplaces": {
    "team-tools": {
      "source": {
        "source": "settings",
        "name": "team-tools",
        "plugins": [
          {
            "name": "code-formatter",
            "source": {
              "source": "github",
              "repo": "acme-corp/code-formatter"
            }
          }
        ]
      }
    }
  }
}
```

#### `strictKnownMarketplaces`

**Managed settings only**: Controls which plugin marketplaces users are allowed to add. This setting can only be configured in [managed settings](/en/settings#settings-files) and provides administrators with strict control over marketplace sources.

**Managed settings file locations**:

* **macOS**: `/Library/Application Support/ClaudeCode/managed-settings.json`
* **Linux and WSL**: `/etc/claude-code/managed-settings.json`
* **Windows**: `C:\Program Files\ClaudeCode\managed-settings.json`

**Key characteristics**:

* Only available in managed settings (`managed-settings.json`)
* Cannot be overridden by user or project settings (highest precedence)
* Enforced BEFORE network/filesystem operations (blocked sources never execute)
* Uses exact matching for source specifications (including `ref`, `path` for git sources), except `hostPattern`, which uses regex matching

**Allowlist behavior**:

* `undefined` (default): No restrictions - users can add any marketplace
* Empty array `[]`: Complete lockdown - users cannot add any new marketplaces
* List of sources: Users can only add marketplaces that match exactly

**All supported source types**:

The allowlist supports multiple marketplace source types. Most sources use exact matching, while `hostPattern` uses regex matching against the marketplace host.

1. **GitHub repositories**:

```json  theme={null}
{ "source": "github", "repo": "acme-corp/approved-plugins" }
{ "source": "github", "repo": "acme-corp/security-tools", "ref": "v2.0" }
{ "source": "github", "repo": "acme-corp/plugins", "ref": "main", "path": "marketplace" }
```

Fields: `repo` (required), `ref` (optional: branch/tag/SHA), `path` (optional: subdirectory)

2. **Git repositories**:

```json  theme={null}
{ "source": "git", "url": "https://gitlab.example.com/tools/plugins.git" }
{ "source": "git", "url": "https://bitbucket.org/acme-corp/plugins.git", "ref": "production" }
{ "source": "git", "url": "ssh://git@git.example.com/plugins.git", "ref": "v3.1", "path": "approved" }
```

Fields: `url` (required), `ref` (optional: branch/tag/SHA), `path` (optional: subdirectory)

3. **URL-based marketplaces**:

```json  theme={null}
{ "source": "url", "url": "https://plugins.example.com/marketplace.json" }
{ "source": "url", "url": "https://cdn.example.com/marketplace.json", "headers": { "Authorization": "Bearer ${TOKEN}" } }
```

Fields: `url` (required), `headers` (optional: HTTP headers for authenticated access)

<Note>
  URL-based marketplaces only download the `marketplace.json` file. They do not download plugin files from the server. Plugins in URL-based marketplaces must use external sources (GitHub, npm, or git URLs) rather than relative paths. For plugins with relative paths, use a Git-based marketplace instead. See [Troubleshooting](/en/plugin-marketplaces#plugins-with-relative-paths-fail-in-url-based-marketplaces) for details.
</Note>

4. **NPM packages**:

```json  theme={null}
{ "source": "npm", "package": "@acme-corp/claude-plugins" }
{ "source": "npm", "package": "@acme-corp/approved-marketplace" }
```

Fields: `package` (required, supports scoped packages)

5. **File paths**:

```json  theme={null}
{ "source": "file", "path": "/usr/local/share/claude/acme-marketplace.json" }
{ "source": "file", "path": "/opt/acme-corp/plugins/marketplace.json" }
```

Fields: `path` (required: absolute path to marketplace.json file)

6. **Directory paths**:

```json  theme={null}
{ "source": "directory", "path": "/usr/local/share/claude/acme-plugins" }
{ "source": "directory", "path": "/opt/acme-corp/approved-marketplaces" }
```

Fields: `path` (required: absolute path to directory containing `.claude-plugin/marketplace.json`)

7. **Host pattern matching**:

```json  theme={null}
{ "source": "hostPattern", "hostPattern": "^github\\.example\\.com$" }
{ "source": "hostPattern", "hostPattern": "^gitlab\\.internal\\.example\\.com$" }
```

Fields: `hostPattern` (required: regex pattern to match against the marketplace host)

Use host pattern matching when you want to allow all marketplaces from a specific host without enumerating each repository individually. This is useful for organizations with internal GitHub Enterprise or GitLab servers where developers create their own marketplaces.

Host extraction by source type:

* `github`: always matches against `github.com`
* `git`: extracts hostname from the URL (supports both HTTPS and SSH formats)
* `url`: extracts hostname from the URL
* `npm`, `file`, `directory`: not supported for host pattern matching

**Configuration examples**:

Example: allow specific marketplaces only:

```json  theme={null}
{
  "strictKnownMarketplaces": [
    {
      "source": "github",
      "repo": "acme-corp/approved-plugins"
    },
    {
      "source": "github",
      "repo": "acme-corp/security-tools",
      "ref": "v2.0"
    },
    {
      "source": "url",
      "url": "https://plugins.example.com/marketplace.json"
    },
    {
      "source": "npm",
      "package": "@acme-corp/compliance-plugins"
    }
  ]
}
```

Example - Disable all marketplace additions:

```json  theme={null}
{
  "strictKnownMarketplaces": []
}
```

Example: allow all marketplaces from an internal git server:

```json  theme={null}
{
  "strictKnownMarketplaces": [
    {
      "source": "hostPattern",
      "hostPattern": "^github\\.example\\.com$"
    }
  ]
}
```

**Exact matching requirements**:

Marketplace sources must match **exactly** for a user's addition to be allowed. For git-based sources (`github` and `git`), this includes all optional fields:

* The `repo` or `url` must match exactly
* The `ref` field must match exactly (or both be undefined)
* The `path` field must match exactly (or both be undefined)

Examples of sources that **do NOT match**:

```json  theme={null}
// These are DIFFERENT sources:
{ "source": "github", "repo": "acme-corp/plugins" }
{ "source": "github", "repo": "acme-corp/plugins", "ref": "main" }

// These are also DIFFERENT:
{ "source": "github", "repo": "acme-corp/plugins", "path": "marketplace" }
{ "source": "github", "repo": "acme-corp/plugins" }
```

**Comparison with `extraKnownMarketplaces`**:

| Aspect                | `strictKnownMarketplaces`            | `extraKnownMarketplaces`             |
| --------------------- | ------------------------------------ | ------------------------------------ |
| **Purpose**           | Organizational policy enforcement    | Team convenience                     |
| **Settings file**     | `managed-settings.json` only         | Any settings file                    |
| **Behavior**          | Blocks non-allowlisted additions     | Auto-installs missing marketplaces   |
| **When enforced**     | Before network/filesystem operations | After user trust prompt              |
| **Can be overridden** | No (highest precedence)              | Yes (by higher precedence settings)  |
| **Source format**     | Direct source object                 | Named marketplace with nested source |
| **Use case**          | Compliance, security restrictions    | Onboarding, standardization          |

**Format difference**:

`strictKnownMarketplaces` uses direct source objects:

```json  theme={null}
{
  "strictKnownMarketplaces": [
    { "source": "github", "repo": "acme-corp/plugins" }
  ]
}
```

`extraKnownMarketplaces` requires named marketplaces:

```json  theme={null}
{
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": { "source": "github", "repo": "acme-corp/plugins" }
    }
  }
}
```

**Using both together**:

`strictKnownMarketplaces` is a policy gate: it controls what users may add but does not register any marketplaces. To both restrict and pre-register a marketplace for all users, set both in `managed-settings.json`:

```json  theme={null}
{
  "strictKnownMarketplaces": [
    { "source": "github", "repo": "acme-corp/plugins" }
  ],
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": { "source": "github", "repo": "acme-corp/plugins" }
    }
  }
}
```

With only `strictKnownMarketplaces` set, users can still add the allowed marketplace manually via `/plugin marketplace add`, but it is not available automatically.

**Important notes**:

* Restrictions are checked BEFORE any network requests or filesystem operations
* When blocked, users see clear error messages indicating the source is blocked by managed policy
* The restriction applies only to adding NEW marketplaces; previously installed marketplaces remain accessible
* Managed settings have the highest precedence and cannot be overridden

See [Managed marketplace restrictions](/en/plugin-marketplaces#managed-marketplace-restrictions) for user-facing documentation.

### Managing plugins

Use the `/plugin` command to manage plugins interactively:

* Browse available plugins from marketplaces
* Install/uninstall plugins
* Enable/disable plugins
* View plugin details (commands, agents, hooks provided)
* Add/remove marketplaces

Learn more about the plugin system in the [plugins documentation](/en/plugins).

## Environment variables

Environment variables let you control Claude Code behavior without editing settings files. Any variable can also be configured in [`settings.json`](#available-settings) under the `env` key to apply it to every session or roll it out to your team.

See the [environment variables reference](/en/env-vars) for the full list.

## Tools available to Claude

Claude Code has access to a set of tools for reading, editing, searching, running commands, and orchestrating subagents. Tool names are the exact strings you use in permission rules and hook matchers.

See the [tools reference](/en/tools-reference) for the full list and Bash tool behavior details.

## See also

* [Permissions](/en/permissions): permission system, rule syntax, tool-specific patterns, and managed policies
* [Authentication](/en/authentication): set up user access to Claude Code
* [Troubleshooting](/en/troubleshooting): solutions for common configuration issues
