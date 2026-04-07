# Configure permissions

Control what Claude Code can access and do with fine-grained permission rules, modes, and managed policies.

Claude Code supports fine-grained permissions so that you can specify exactly what the agent is allowed to do and what it cannot. Permission settings can be checked into version control and distributed to all developers in your organization, as well as customized by individual developers.

## Permission system

Claude Code uses a tiered permission system to balance power and safety:

| Tool type         | Example          | Approval required | "Yes, don't ask again" behavior               |
| :---------------- | :--------------- | :---------------- | :-------------------------------------------- |
| Read-only         | File reads, Grep | No                | N/A                                           |
| Bash commands     | Shell execution  | Yes               | Permanently per project directory and command |
| File modification | Edit/write files | Yes               | Until session end                             |

## Manage permissions

You can view and manage Claude Code's tool permissions with `/permissions`. This UI lists all permission rules and the settings.json file they are sourced from.

* **Allow** rules let Claude Code use the specified tool without manual approval.
* **Ask** rules prompt for confirmation whenever Claude Code tries to use the specified tool.
* **Deny** rules prevent Claude Code from using the specified tool.

Rules are evaluated in order: **deny -> ask -> allow**. The first matching rule wins, so deny rules always take precedence.

## Permission modes

Claude Code supports several permission modes that control how tools are approved. See [Permission modes](/en/permission-modes) for when to use each one. Set the `defaultMode` in your [settings files](/en/settings#settings-files):

| Mode                | Description                                                                                                                      |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| `default`           | Standard behavior: prompts for permission on first use of each tool                                                              |
| `acceptEdits`       | Automatically accepts file edit permissions for the session, except writes to protected directories                              |
| `plan`              | Plan Mode: Claude can analyze but not modify files or execute commands                                                           |
| `auto`              | Auto-approves tool calls with background safety checks that verify actions align with your request. Currently a research preview |
| `dontAsk`           | Auto-denies tools unless pre-approved via `/permissions` or `permissions.allow` rules                                            |
| `bypassPermissions` | Skips permission prompts except for writes to protected directories (see warning below)                                          |

`bypassPermissions` mode skips permission prompts. Writes to `.git`, `.claude`, `.vscode`, `.idea`, and `.husky` directories still prompt for confirmation to prevent accidental corruption of repository state, editor configuration, and git hooks. Writes to `.claude/commands`, `.claude/agents`, and `.claude/skills` are exempt and do not prompt, because Claude routinely writes there when creating skills, subagents, and commands. Only use this mode in isolated environments like containers or VMs where Claude Code cannot cause damage. Administrators can prevent this mode by setting `permissions.disableBypassPermissionsMode` to `"disable"` in [managed settings](#managed-settings).

To prevent `bypassPermissions` or `auto` mode from being used, set `permissions.disableBypassPermissionsMode` or `permissions.disableAutoMode` to `"disable"` in any [settings file](/en/settings#settings-files). These are most useful in [managed settings](#managed-settings) where they cannot be overridden.

## Permission rule syntax

Permission rules follow the format `Tool` or `Tool(specifier)`.

### Match all uses of a tool

To match all uses of a tool, use just the tool name without parentheses:

| Rule       | Effect                         |
| :--------- | :----------------------------- |
| `Bash`     | Matches all Bash commands      |
| `WebFetch` | Matches all web fetch requests |
| `Read`     | Matches all file reads         |

`Bash(*)` is equivalent to `Bash` and matches all Bash commands.

### Use specifiers for fine-grained control

Add a specifier in parentheses to match specific tool uses:

| Rule                           | Effect                                                   |
| :----------------------------- | :------------------------------------------------------- |
| `Bash(npm run build)`          | Matches the exact command `npm run build`                |
| `Read(./.env)`                 | Matches reading the `.env` file in the current directory |
| `WebFetch(domain:example.com)` | Matches fetch requests to example.com                    |

### Wildcard patterns

Bash rules support glob patterns with `*`. Wildcards can appear at any position in the command. This configuration allows npm and git commit commands while blocking git push:

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(git commit *)",
      "Bash(git * main)",
      "Bash(* --version)",
      "Bash(* --help *)"
    ],
    "deny": [
      "Bash(git push *)"
    ]
  }
}
```

The space before `*` matters: `Bash(ls *)` matches `ls -la` but not `lsof`, while `Bash(ls*)` matches both. The legacy `:*` suffix syntax is equivalent to ` *` but is deprecated.

## Tool-specific permission rules

### Bash

Bash permission rules support wildcard matching with `*`. Wildcards can appear at any position in the command, including at the beginning, middle, or end:

* `Bash(npm run build)` matches the exact Bash command `npm run build`
* `Bash(npm run test *)` matches Bash commands starting with `npm run test`
* `Bash(npm *)` matches any command starting with `npm `
* `Bash(* install)` matches any command ending with ` install`
* `Bash(git * main)` matches commands like `git checkout main`, `git merge main`

When `*` appears at the end with a space before it (like `Bash(ls *)`), it enforces a word boundary, requiring the prefix to be followed by a space or end-of-string. For example, `Bash(ls *)` matches `ls -la` but not `lsof`. In contrast, `Bash(ls*)` without a space matches both `ls -la` and `lsof` because there's no word boundary constraint.

Claude Code is aware of shell operators (like `&&`) so a prefix match rule like `Bash(safe-cmd *)` won't give it permission to run the command `safe-cmd && other-cmd`.

When you approve a compound command with "Yes, don't ask again", Claude Code saves a separate rule for each subcommand that requires approval, rather than a single rule for the full compound string. For example, approving `git status && npm test` saves a rule for `npm test`, so future `npm test` invocations are recognized regardless of what precedes the `&&`. Subcommands like `cd` into a subdirectory generate their own Read rule for that path. Up to 5 rules may be saved for a single compound command.

Bash permission patterns that try to constrain command arguments are fragile. For example, `Bash(curl http://github.com/ *)` intends to restrict curl to GitHub URLs, but won't match variations like options before URL, different protocol, redirects, variables, or extra spaces. For more reliable URL filtering, consider restricting Bash network tools using deny rules and using the WebFetch tool with `WebFetch(domain:github.com)` permission for allowed domains instead.

Note that using WebFetch alone does not prevent network access. If Bash is allowed, Claude can still use `curl`, `wget`, or other tools to reach any URL.

### Read and Edit

`Edit` rules apply to all built-in tools that edit files. Claude makes a best-effort attempt to apply `Read` rules to all built-in tools that read files like Grep and Glob.

Read and Edit deny rules apply to Claude's built-in file tools, not to Bash subprocesses. A `Read(./.env)` deny rule blocks the Read tool but does not prevent `cat .env` in Bash. For OS-level enforcement that blocks all processes from accessing a path, [enable the sandbox](/en/sandboxing).

Read and Edit rules both follow the [gitignore](https://git-scm.com/docs/gitignore) specification with four distinct pattern types:

| Pattern            | Meaning                                | Example                          | Matches                        |
| ------------------ | -------------------------------------- | -------------------------------- | ------------------------------ |
| `//path`           | **Absolute** path from filesystem root | `Read(//Users/alice/secrets/**)` | `/Users/alice/secrets/**`      |
| `~/path`           | Path from **home** directory           | `Read(~/Documents/*.pdf)`        | `/Users/alice/Documents/*.pdf` |
| `/path`            | Path **relative to project root**      | `Edit(/src/**/*.ts)`             | `<project root>/src/**/*.ts`   |
| `path` or `./path` | Path **relative to current directory** | `Read(*.env)`                    | `<cwd>/*.env`                  |

A pattern like `/Users/alice/file` is NOT an absolute path. It's relative to the project root. Use `//Users/alice/file` for absolute paths.

On Windows, paths are normalized to POSIX form before matching. `C:\Users\alice` becomes `/c/Users/alice`, so use `//c/**/.env` to match `.env` files anywhere on that drive. To match across all drives, use `//**/.env`.

Examples:

* `Edit(/docs/**)`: edits in `<project>/docs/` (NOT `/docs/` and NOT `<project>/.claude/docs/`)
* `Read(~/.zshrc)`: reads your home directory's `.zshrc`
* `Edit(//tmp/scratch.txt)`: edits the absolute path `/tmp/scratch.txt`
* `Read(src/**)`: reads from `<current-directory>/src/`

In gitignore patterns, `*` matches files in a single directory while `**` matches recursively across directories. To allow all file access, use just the tool name without parentheses: `Read`, `Edit`, or `Write`.

### WebFetch

* `WebFetch(domain:example.com)` matches fetch requests to example.com

### MCP

* `mcp__puppeteer` matches any tool provided by the `puppeteer` server (name configured in Claude Code)
* `mcp__puppeteer__*` wildcard syntax that also matches all tools from the `puppeteer` server
* `mcp__puppeteer__puppeteer_navigate` matches the `puppeteer_navigate` tool provided by the `puppeteer` server

### Agent (subagents)

Use `Agent(AgentName)` rules to control which [subagents](/en/sub-agents) Claude can use:

* `Agent(Explore)` matches the Explore subagent
* `Agent(Plan)` matches the Plan subagent
* `Agent(my-custom-agent)` matches a custom subagent named `my-custom-agent`

Add these rules to the `deny` array in your settings or use the `--disallowedTools` CLI flag to disable specific agents. To disable the Explore agent:

```json
{
  "permissions": {
    "deny": ["Agent(Explore)"]
  }
}
```

## Extend permissions with hooks

[Claude Code hooks](/en/hooks-guide) provide a way to register custom shell commands to perform permission evaluation at runtime. When Claude Code makes a tool call, PreToolUse hooks run before the permission prompt. The hook output can deny the tool call, force a prompt, or skip the prompt to let the call proceed.

Skipping the prompt does not bypass permission rules. Deny and ask rules are still evaluated after a hook returns `"allow"`, so a matching deny rule still blocks the call. This preserves the deny-first precedence described in [Manage permissions](#manage-permissions), including deny rules set in managed settings.

A blocking hook also takes precedence over allow rules. A hook that exits with code 2 stops the tool call before permission rules are evaluated, so the block applies even when an allow rule would otherwise let the call proceed. To run all Bash commands without prompts except for a few you want blocked, add `"Bash"` to your allow list and register a PreToolUse hook that rejects those specific commands. See [Block edits to protected files](/en/hooks-guide#block-edits-to-protected-files) for a hook script you can adapt.

## Working directories

By default, Claude has access to files in the directory where it was launched. You can extend this access:

* **During startup**: use `--add-dir <path>` CLI argument
* **During session**: use `/add-dir` command
* **Persistent configuration**: add to `additionalDirectories` in [settings files](/en/settings#settings-files)

Files in additional directories follow the same permission rules as the original working directory: they become readable without prompts, and file editing permissions follow the current permission mode.

### Additional directories grant file access, not configuration

Adding a directory extends where Claude can read and edit files. It does not make that directory a full configuration root: most `.claude/` configuration is not discovered from additional directories, though a few types are loaded as exceptions.

The following configuration types are loaded from `--add-dir` directories:

| Configuration                                      | Loaded from `--add-dir`                                           |
| :------------------------------------------------- | :---------------------------------------------------------------- |
| [Skills](/en/skills) in `.claude/skills/`          | Yes, with live reload                                             |
| Plugin settings in `.claude/settings.json`         | `enabledPlugins` and `extraKnownMarketplaces` only                |
| [CLAUDE.md](/en/memory) files and `.claude/rules/` | Only when `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1` is set |

Everything else, including subagents, commands, output styles, hooks, and other settings, is discovered only from the current working directory and its parents, your user directory at `~/.claude/`, and managed settings. To share that configuration across projects, use one of these approaches:

* **User-level configuration**: place files in `~/.claude/agents/`, `~/.claude/output-styles/`, or `~/.claude/settings.json` to make them available in every project
* **Plugins**: package and distribute configuration as a [plugin](/en/plugins) that teams can install
* **Launch from the config directory**: run Claude Code from the directory containing the `.claude/` configuration you want

## How permissions interact with sandboxing

Permissions and [sandboxing](/en/sandboxing) are complementary security layers:

* **Permissions** control which tools Claude Code can use and which files or domains it can access. They apply to all tools (Bash, Read, Edit, WebFetch, MCP, and others).
* **Sandboxing** provides OS-level enforcement that restricts the Bash tool's filesystem and network access. It applies only to Bash commands and their child processes.

Use both for defense-in-depth:

* Permission deny rules block Claude from even attempting to access restricted resources
* Sandbox restrictions prevent Bash commands from reaching resources outside defined boundaries, even if a prompt injection bypasses Claude's decision-making
* Filesystem restrictions in the sandbox use Read and Edit deny rules, not separate sandbox configuration
* Network restrictions combine WebFetch permission rules with the sandbox's `allowedDomains` list

When sandboxing is enabled with `autoAllowBashIfSandboxed: true`, which is the default, sandboxed Bash commands run without prompting even if your permissions include `ask: Bash(*)`. The sandbox boundary substitutes for the per-command prompt. See [sandbox modes](/en/sandboxing#sandbox-modes) to change this behavior.

## Managed settings

For organizations that need centralized control over Claude Code configuration, administrators can deploy managed settings that cannot be overridden by user or project settings. These policy settings follow the same format as regular settings files and can be delivered through MDM/OS-level policies, managed settings files, or [server-managed settings](/en/server-managed-settings). See [settings files](/en/settings#settings-files) for delivery mechanisms and file locations.

### Managed-only settings

The following settings are only read from managed settings. Placing them in user or project settings files has no effect.

| Setting                                        | Description                                                                                                                                                                                                                 |
| :--------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `allowedChannelPlugins`                        | Allowlist of channel plugins that may push messages. Replaces the default Anthropic allowlist when set. Requires `channelsEnabled: true`.                                                                                    |
| `allowManagedHooksOnly`                        | When `true`, prevents loading of user, project, and plugin hooks. Only managed hooks and SDK hooks are allowed                                                                                                              |
| `allowManagedMcpServersOnly`                   | When `true`, only `allowedMcpServers` from managed settings are respected. `deniedMcpServers` still merges from all sources.                                                                                              |
| `allowManagedPermissionRulesOnly`              | When `true`, prevents user and project settings from defining `allow`, `ask`, or `deny` permission rules. Only rules in managed settings apply                                                                              |
| `blockedMarketplaces`                          | Blocklist of marketplace sources. Blocked sources are checked before downloading, so they never touch the filesystem.                                                                                                      |
| `channelsEnabled`                              | Allow channels for Team and Enterprise users. Unset or `false` blocks channel message delivery regardless of what users pass to `--channels`                                                                                 |
| `forceRemoteSettingsRefresh`                   | When `true`, blocks CLI startup until remote managed settings are freshly fetched and exits if the fetch fails.                                                                                                            |
| `pluginTrustMessage`                           | Custom message appended to the plugin trust warning shown before installation                                                                                                                                               |
| `sandbox.filesystem.allowManagedReadPathsOnly` | When `true`, only `filesystem.allowRead` paths from managed settings are respected. `denyRead` still merges from all sources                                                                                                |
| `sandbox.network.allowManagedDomainsOnly`      | When `true`, only `allowedDomains` and `WebFetch(domain:...)` allow rules from managed settings are respected. Non-allowed domains are blocked automatically without prompting the user. Denied domains still merge from all sources |
| `strictKnownMarketplaces`                      | Controls which plugin marketplaces users can add.                                                                                                                                                                          |

`disableBypassPermissionsMode` is typically placed in managed settings to enforce organizational policy, but it works from any scope. A user can set it in their own settings to lock themselves out of bypass mode.

Access to [Remote Control](/en/remote-control) and [web sessions](/en/claude-code-on-the-web) is not controlled by a managed settings key. On Team and Enterprise plans, an admin enables or disables these features in [Claude Code admin settings](https://claude.ai/admin-settings/claude-code).

## Review auto mode denials

When [auto mode](/en/permission-modes#eliminate-prompts-with-auto-mode) denies a tool call, a notification appears and the denied action is recorded in `/permissions` under the Recently denied tab. Press `r` on a denied action to mark it for retry: when you exit the dialog, Claude Code sends a message telling the model it may retry that tool call and resumes the conversation.

To react to denials programmatically, use the [`PermissionDenied` hook](/en/hooks#permissiondenied).

## Configure the auto mode classifier

[Auto mode](/en/permission-modes#eliminate-prompts-with-auto-mode) uses a classifier model to decide whether each action is safe to run without prompting. Out of the box it trusts only the working directory and, if present, the current repo's remotes. Actions like pushing to your company's source control org or writing to a team cloud bucket will be blocked as potential data exfiltration. The `autoMode` settings block lets you tell the classifier which infrastructure your organization trusts.

The classifier reads `autoMode` from user settings, `.claude/settings.local.json`, and managed settings. It does not read from shared project settings in `.claude/settings.json`, because a checked-in repo could otherwise inject its own allow rules.

| Scope                      | File                          | Use for                                             |
| :------------------------- | :---------------------------- | :-------------------------------------------------- |
| One developer              | `~/.claude/settings.json`     | Personal trusted infrastructure                     |
| One project, one developer | `.claude/settings.local.json` | Per-project trusted buckets or services, gitignored |
| Organization-wide          | Managed settings              | Trusted infrastructure enforced for all developers  |

Entries from each scope are combined. A developer can extend `environment`, `allow`, and `soft_deny` with personal entries but cannot remove entries that managed settings provide. Because allow rules act as exceptions to block rules inside the classifier, a developer-added `allow` entry can override an organization `soft_deny` entry: the combination is additive, not a hard policy boundary. If you need a rule that developers cannot work around, use `permissions.deny` in managed settings instead, which blocks actions before the classifier is consulted.

### Define trusted infrastructure

For most organizations, `autoMode.environment` is the only field you need to set. It tells the classifier which repos, buckets, and domains are trusted, without touching the built-in block and allow rules. The classifier uses `environment` to decide what "external" means: any destination not listed is a potential exfiltration target.

```json
{
  "autoMode": {
    "environment": [
      "Source control: github.example.com/acme-corp and all repos under it",
      "Trusted cloud buckets: s3://acme-build-artifacts, gs://acme-ml-datasets",
      "Trusted internal domains: *.corp.example.com, api.internal.example.com",
      "Key internal services: Jenkins at ci.example.com, Artifactory at artifacts.example.com"
    ]
  }
}
```

Entries are prose, not regex or tool patterns. The classifier reads them as natural-language rules. Write them the way you would describe your infrastructure to a new engineer. A thorough environment section covers:

* **Organization**: your company name and what Claude Code is primarily used for
* **Source control**: every GitHub, GitLab, or Bitbucket org your developers push to
* **Cloud providers and trusted buckets**: bucket names or prefixes that Claude should be able to read from and write to
* **Trusted internal domains**: hostnames for APIs, dashboards, and services inside your network
* **Key internal services**: CI, artifact registries, internal package indexes, incident tooling
* **Additional context**: regulated-industry constraints, multi-tenant infrastructure, or compliance requirements

A useful starting template: fill in the bracketed fields and remove any lines that don't apply:

```json
{
  "autoMode": {
    "environment": [
      "Organization: {COMPANY_NAME}. Primary use: {PRIMARY_USE_CASE}",
      "Source control: {SOURCE_CONTROL}",
      "Cloud provider(s): {CLOUD_PROVIDERS}",
      "Trusted cloud buckets: {TRUSTED_BUCKETS}",
      "Trusted internal domains: {TRUSTED_DOMAINS}",
      "Key internal services: {SERVICES}",
      "Additional context: {EXTRA}"
    ]
  }
}
```

The more specific context you give, the better the classifier can distinguish routine internal operations from exfiltration attempts.

You don't need to fill everything in at once. A reasonable rollout: start with the defaults and add your source control org and key internal services, which resolves the most common false positives like pushing to your own repos. Add trusted domains and cloud buckets next. Fill the rest as blocks come up.

### Override the block and allow rules

Two additional fields let you replace the classifier's built-in rule lists: `autoMode.soft_deny` controls what gets blocked, and `autoMode.allow` controls which exceptions apply. Each is an array of prose descriptions, read as natural-language rules.

Inside the classifier, the precedence is: `soft_deny` rules block first, then `allow` rules override as exceptions, then explicit user intent overrides both. If the user's message directly and specifically describes the exact action Claude is about to take, the classifier allows it even if a `soft_deny` rule matches. General requests don't count: asking Claude to "clean up the repo" does not authorize force-pushing, but asking Claude to "force-push this branch" does.

To loosen: remove rules from `soft_deny` when the defaults block something your pipeline already guards against with PR review, CI, or staging environments, or add to `allow` when the classifier repeatedly flags a routine pattern the default exceptions don't cover. To tighten: add to `soft_deny` for risks specific to your environment that the defaults miss, or remove from `allow` to hold a default exception to the block rules. In all cases, run `claude auto-mode defaults` to get the full default lists, then copy and edit: never start from an empty list.

```json
{
  "autoMode": {
    "environment": [
      "Source control: github.example.com/acme-corp and all repos under it"
    ],
    "allow": [
      "Deploying to the staging namespace is allowed: staging is isolated from production and resets nightly",
      "Writing to s3://acme-scratch/ is allowed: ephemeral bucket with a 7-day lifecycle policy"
    ],
    "soft_deny": [
      "Never run database migrations outside the migrations CLI, even against dev databases",
      "Never modify files under infra/terraform/prod/: production infrastructure changes go through the review workflow",
      "...copy full default soft_deny list here first, then add your rules..."
    ]
  }
}
```

Setting `allow` or `soft_deny` replaces the entire default list for that section. If you set `soft_deny` with a single entry, every built-in block rule is discarded: force push, data exfiltration, `curl | bash`, production deploys, and all other default block rules become allowed. To customize safely, run `claude auto-mode defaults` to print the built-in rules, copy them into your settings file, then review each rule against your own pipeline and risk tolerance. Only remove rules for risks your infrastructure already mitigates.

The three sections are evaluated independently, so setting `environment` alone leaves the default `allow` and `soft_deny` lists intact.

### Inspect the defaults and your effective config

Because setting `allow` or `soft_deny` replaces the defaults, start any customization by copying the full default lists. Three CLI subcommands help you inspect and validate:

```bash
claude auto-mode defaults  # the built-in environment, allow, and soft_deny rules
claude auto-mode config    # what the classifier actually uses: your settings where set, defaults otherwise
claude auto-mode critique  # get AI feedback on your custom allow and soft_deny rules
```

Save the output of `claude auto-mode defaults` to a file, edit the lists to match your policy, and paste the result into your settings file. After saving, run `claude auto-mode config` to confirm the effective rules are what you expect. If you've written custom rules, `claude auto-mode critique` reviews them and flags entries that are ambiguous, redundant, or likely to cause false positives.

## Settings precedence

Permission rules follow the same [settings precedence](/en/settings#settings-precedence) as all other Claude Code settings:

1. **Managed settings**: cannot be overridden by any other level, including command line arguments
2. **Command line arguments**: temporary session overrides
3. **Local project settings** (`.claude/settings.local.json`)
4. **Shared project settings** (`.claude/settings.json`)
5. **User settings** (`~/.claude/settings.json`)

If a tool is denied at any level, no other level can allow it. For example, a managed settings deny cannot be overridden by `--allowedTools`, and `--disallowedTools` can add restrictions beyond what managed settings define.

If a permission is allowed in user settings but denied in project settings, the project setting takes precedence and the permission is blocked.

## Example configurations

This repository includes starter settings configurations for common deployment scenarios. Use these as starting points and adjust them to fit your needs.

## See also

* [Settings](/en/settings): complete configuration reference including the permission settings table
* [Sandboxing](/en/sandboxing): OS-level filesystem and network isolation for Bash commands
* [Authentication](/en/authentication): set up user access to Claude Code
* [Security](/en/security): security safeguards and best practices
* [Hooks](/en/hooks-guide): automate workflows and extend permission evaluation
