> ## Documentation Index
> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Use Claude Code Desktop

> Get more out of Claude Code Desktop: computer use, Dispatch sessions from your phone, parallel sessions with Git isolation, visual diff review, app previews, PR monitoring, connectors, and enterprise configuration.

The Code tab within the Claude Desktop app lets you use Claude Code through a graphical interface instead of the terminal.

Desktop adds these capabilities on top of the standard Claude Code experience:

* [Visual diff review](#review-changes-with-diff-view) with inline comments
* [Live app preview](#preview-your-app) with dev servers
* [Computer use](#let-claude-use-your-computer) to open apps and control your screen on macOS and Windows
* [GitHub PR monitoring](#monitor-pull-request-status) with auto-fix and auto-merge
* [Parallel sessions](#work-in-parallel-with-sessions) with automatic Git worktree isolation
* [Dispatch](#sessions-from-dispatch) integration: send a task from your phone, get a session here
* [Scheduled tasks](/en/desktop-scheduled-tasks) that run Claude on a recurring schedule
* [Connectors](#connect-external-tools) for GitHub, Slack, Linear, and more
* Local, [SSH](#ssh-sessions), and [cloud](#run-long-running-tasks-remotely) environments

<Tip>
  New to Desktop? Start with [Get started](/en/desktop-quickstart) to install the app and make your first edit.
</Tip>

This page covers [working with code](#work-with-code), [computer use](#let-claude-use-your-computer), [managing sessions](#manage-sessions), [extending Claude Code](#extend-claude-code), and [configuration](#environment-configuration). It also includes a [CLI comparison](#coming-from-the-cli) and [troubleshooting](#troubleshooting).

## Start a session

Before you send your first message, configure four things in the prompt area:

* **Environment**: choose where Claude runs. Select **Local** for your machine, **Remote** for Anthropic-hosted cloud sessions, or an [**SSH connection**](#ssh-sessions) for a remote machine you manage. See [environment configuration](#environment-configuration).
* **Project folder**: select the folder or repository Claude works in. For remote sessions, you can add [multiple repositories](#run-long-running-tasks-remotely).
* **Model**: pick a [model](/en/model-config#available-models) from the dropdown next to the send button. The model is locked once the session starts.
* **Permission mode**: choose how much autonomy Claude has from the [mode selector](#choose-a-permission-mode). You can change this during the session.

Type your task and press **Enter** to start. Each session tracks its own context and changes independently.

## Work with code

Give Claude the right context, control how much it does on its own, and review what it changed.

### Use the prompt box

Type what you want Claude to do and press **Enter** to send. Claude reads your project files, makes changes, and runs commands based on your [permission mode](#choose-a-permission-mode). You can interrupt Claude at any point: click the stop button or type your correction and press **Enter**. Claude stops what it's doing and adjusts based on your input.

The **+** button next to the prompt box gives you access to file attachments, [skills](#use-skills), [connectors](#connect-external-tools), and [plugins](#install-plugins).

### Add files and context to prompts

The prompt box supports two ways to bring in external context:

* **@mention files**: type `@` followed by a filename to add a file to the conversation context. Claude can then read and reference that file. @mention is not available in remote sessions.
* **Attach files**: attach images, PDFs, and other files to your prompt using the attachment button, or drag and drop files directly into the prompt. This is useful for sharing screenshots of bugs, design mockups, or reference documents.

### Choose a permission mode

Permission modes control how much autonomy Claude has during a session: whether it asks before editing files, running commands, or both. You can switch modes at any time using the mode selector next to the send button. Start with Ask permissions to see exactly what Claude does, then move to Auto accept edits or Plan mode as you get comfortable.

| Mode                   | Settings key        | Behavior                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ask permissions**    | `default`           | Claude asks before editing files or running commands. You see a diff and can accept or reject each change. Recommended for new users.                                                                                                                                                                                 |
| **Auto accept edits**  | `acceptEdits`       | Claude auto-accepts file edits but still asks before running terminal commands. Use this when you trust file changes and want faster iteration.                                                                                                                                                                       |
| **Plan mode**          | `plan`              | Claude analyzes your code and creates a plan without modifying files or running commands. Good for complex tasks where you want to review the approach first.                                                                                                                                                         |
| **Auto**               | `auto`              | Claude executes all actions with background safety checks that verify alignment with your request. Reduces permission prompts while maintaining oversight. Currently a research preview. Available on Team, Enterprise, and API plans. Requires Claude Sonnet 4.6 or Opus 4.6. Enable in your Settings → Claude Code. |
| **Bypass permissions** | `bypassPermissions` | Claude runs without any permission prompts, equivalent to `--dangerously-skip-permissions` in the CLI. Enable in your Settings → Claude Code under "Allow bypass permissions mode". Only use this in sandboxed containers or VMs. Enterprise admins can disable this option.                                          |

The `dontAsk` permission mode is available only in the [CLI](/en/permission-modes#allow-only-pre-approved-tools-with-dontask-mode).

<Tip title="Best practice">
  Start complex tasks in Plan mode so Claude maps out an approach before making changes. Once you approve the plan, switch to Auto accept edits or Ask permissions to execute it. See [explore first, then plan, then code](/en/best-practices#explore-first-then-plan-then-code) for more on this workflow.
</Tip>

Remote sessions support Auto accept edits and Plan mode. Ask permissions is not available because remote sessions auto-accept file edits by default, and Bypass permissions is not available because the remote environment is already sandboxed.

Enterprise admins can restrict which permission modes are available. See [enterprise configuration](#enterprise-configuration) for details.

### Preview your app

Claude can start a dev server and open an embedded browser to verify its changes. This works for frontend web apps as well as backend servers: Claude can test API endpoints, view server logs, and iterate on issues it finds. In most cases, Claude starts the server automatically after editing project files. You can also ask Claude to preview at any time. By default, Claude [auto-verifies](#auto-verify-changes) changes after every edit.

From the preview panel, you can:

* Interact with your running app directly in the embedded browser
* Watch Claude verify its own changes automatically: it takes screenshots, inspects the DOM, clicks elements, fills forms, and fixes issues it finds
* Start or stop servers from the **Preview** dropdown in the session toolbar
* Persist cookies and local storage across server restarts by selecting **Persist sessions** in the dropdown, so you don't have to re-login during development
* Edit the server configuration or stop all servers at once

Claude creates the initial server configuration based on your project. If your app uses a custom dev command, edit `.claude/launch.json` to match your setup. See [Configure preview servers](#configure-preview-servers) for the full reference.

To clear saved session data, toggle **Persist preview sessions** off in Settings → Claude Code. To disable preview entirely, toggle off **Preview** in Settings → Claude Code.

### Review changes with diff view

After Claude makes changes to your code, the diff view lets you review modifications file by file before creating a pull request.

When Claude changes files, a diff stats indicator appears showing the number of lines added and removed, such as `+12 -1`. Click this indicator to open the diff viewer, which displays a file list on the left and the changes for each file on the right.

To comment on specific lines, click any line in the diff to open a comment box. Type your feedback and press **Enter** to add the comment. After adding comments to multiple lines, submit all comments at once:

* **macOS**: press **Cmd+Enter**
* **Windows**: press **Ctrl+Enter**

Claude reads your comments and makes the requested changes, which appear as a new diff you can review.

### Review your code

In the diff view, click **Review code** in the top-right toolbar to ask Claude to evaluate the changes before you commit. Claude examines the current diffs and leaves comments directly in the diff view. You can respond to any comment or ask Claude to revise.

The review focuses on high-signal issues: compile errors, definite logic errors, security vulnerabilities, and obvious bugs. It does not flag style, formatting, pre-existing issues, or anything a linter would catch.

### Monitor pull request status

After you open a pull request, a CI status bar appears in the session. Claude Code uses the GitHub CLI to poll check results and surface failures.

* **Auto-fix**: when enabled, Claude automatically attempts to fix failing CI checks by reading the failure output and iterating.
* **Auto-merge**: when enabled, Claude merges the PR once all checks pass. The merge method is squash. Auto-merge must be [enabled in your GitHub repository settings](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-auto-merge-for-pull-requests-in-your-repository) for this to work.

Use the **Auto-fix** and **Auto-merge** toggles in the CI status bar to enable either option. Claude Code also sends a desktop notification when CI finishes.

<Note>
  PR monitoring requires the [GitHub CLI (`gh`)](https://cli.github.com/) to be installed and authenticated on your machine. If `gh` is not installed, Desktop prompts you to install it the first time you try to create a PR.
</Note>

## Let Claude use your computer

Computer use lets Claude open your apps, control your screen, and work directly on your machine the way you would. Ask Claude to test a native app in a mobile simulator, interact with a desktop tool that has no CLI, or automate something that only works through a GUI.

<Note>
  Computer use is a research preview on macOS and Windows that requires a Pro or Max plan. It is not available on Team or Enterprise plans. The Claude Desktop app must be running.
</Note>

Computer use is off by default. [Enable it in Settings](#enable-computer-use) before Claude can control your screen. On macOS, you also need to grant Accessibility and Screen Recording permissions.

<Warning>
  Unlike the [sandboxed Bash tool](/en/sandboxing), computer use runs on your actual desktop with access to whatever you approve. Claude checks each action and flags potential prompt injection from on-screen content, but the trust boundary is different. See the [computer use safety guide](https://support.claude.com/en/articles/14128542) for best practices.
</Warning>

### When computer use applies

Claude has several ways to interact with an app or service, and computer use is the broadest and slowest. It tries the most precise tool first:

* If you have a [connector](#connect-external-tools) for a service, Claude uses the connector.
* If the task is a shell command, Claude uses Bash.
* If the task is browser work and you have [Claude in Chrome](/en/chrome) set up, Claude uses that.
* If none of those apply, Claude uses computer use.

The [per-app access tiers](#app-permissions) reinforce this: browsers are capped at view-only, and terminals and IDEs at click-only, steering Claude toward the dedicated tool even when computer use is active. Screen control is reserved for things nothing else can reach, like native apps, hardware control panels, mobile simulators, or proprietary tools without an API.

### Enable computer use

Computer use is off by default. If you ask Claude to do something that needs it while it's off, Claude tells you it could do the task if you enable computer use in Settings.

<Steps>
  <Step title="Update the desktop app">
    Make sure you have the latest version of Claude Desktop. Download or update at [claude.com/download](https://claude.com/download), then restart the app.
  </Step>

  <Step title="Turn on the toggle">
    In the desktop app, go to **Settings > General** (under **Desktop app**). Find the **Computer use** toggle and turn it on. On Windows, the toggle takes effect immediately and setup is complete. On macOS, continue to the next step.

    If you don't see the toggle, confirm you're on macOS or Windows with a Pro or Max plan, then update and restart the app.
  </Step>

  <Step title="Grant macOS permissions">
    On macOS, grant two system permissions before the toggle takes effect:

    * **Accessibility**: lets Claude click, type, and scroll
    * **Screen Recording**: lets Claude see what's on your screen

    The Settings page shows the current status of each permission. If either is denied, click the badge to open the relevant System Settings pane.
  </Step>
</Steps>

### App permissions

The first time Claude needs to use an app, a prompt appears in your session. Click **Allow for this session** or **Deny**. Approvals last for the current session, or 30 minutes in [Dispatch-spawned sessions](#sessions-from-dispatch).

The prompt also shows what level of control Claude gets for that app. These tiers are fixed by app category and can't be changed:

| Tier         | What Claude can do                                       | Applies to                  |
| :----------- | :------------------------------------------------------- | :-------------------------- |
| View only    | See the app in screenshots                               | Browsers, trading platforms |
| Click only   | Click and scroll, but not type or use keyboard shortcuts | Terminals, IDEs             |
| Full control | Click, type, drag, and use keyboard shortcuts            | Everything else             |

Apps with broad reach, like terminals, Finder or File Explorer, and System Settings or Settings, show an extra warning in the prompt so you know what approving them grants.

You can configure two settings in **Settings > General** (under **Desktop app**):

* **Denied apps**: add apps here to reject them without prompting. Claude may still affect a denied app indirectly through actions in an allowed app, but it can't interact with the denied app directly.
* **Unhide apps when Claude finishes**: while Claude is working, your other windows are hidden so it interacts with only the approved app. When Claude finishes, hidden windows are restored unless you turn this setting off.

## Manage sessions

Each session is an independent conversation with its own context and changes. You can run multiple sessions in parallel, send work to the cloud, or let Dispatch start sessions for you from your phone.

### Work in parallel with sessions

Click **+ New session** in the sidebar to work on multiple tasks in parallel. For Git repositories, each session gets its own isolated copy of your project using [Git worktrees](/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees), so changes in one session don't affect other sessions until you commit them.

Worktrees are stored in `<project-root>/.claude/worktrees/` by default. You can change this to a custom directory in Settings → Claude Code under "Worktree location". You can also set a branch prefix that gets prepended to every worktree branch name, which is useful for keeping Claude-created branches organized. To remove a worktree when you're done, hover over the session in the sidebar and click the archive icon.

To include gitignored files like `.env` in new worktrees, create a [`.worktreeinclude` file](/en/common-workflows#copy-gitignored-files-to-worktrees) in your project root.

<Note>
  Session isolation requires [Git](https://git-scm.com/downloads). Most Macs include Git by default. Run `git --version` in Terminal to check. On Windows, Git is required for the Code tab to work: [download Git for Windows](https://git-scm.com/downloads/win), install it, and restart the app. If you run into Git errors, try a Cowork session to help troubleshoot your setup.
</Note>

Use the filter icon at the top of the sidebar to filter sessions by status (Active, Archived) and environment (Local, Cloud). To rename a session or check context usage, click the session title in the toolbar at the top of the active session. When context fills up, Claude automatically summarizes the conversation and continues working. You can also type `/compact` to trigger summarization earlier and free up context space. See [the context window](/en/how-claude-code-works#the-context-window) for details on how compaction works.

### Run long-running tasks remotely

For large refactors, test suites, migrations, or other long-running tasks, select **Remote** instead of **Local** when starting a session. Remote sessions run on Anthropic's cloud infrastructure and continue even if you close the app or shut down your computer. Check back anytime to see progress or steer Claude in a different direction. You can also monitor remote sessions from [claude.ai/code](https://claude.ai/code) or the Claude iOS app.

Remote sessions also support multiple repositories. After selecting a cloud environment, click the **+** button next to the repo pill to add additional repositories to the session. Each repo gets its own branch selector. This is useful for tasks that span multiple codebases, such as updating a shared library and its consumers.

See [Claude Code on the web](/en/claude-code-on-the-web) for more on how remote sessions work.

### Continue in another surface

The **Continue in** menu, accessible from the VS Code icon in the bottom right of the session toolbar, lets you move your session to another surface:

* **Claude Code on the Web**: sends your local session to continue running remotely. Desktop pushes your branch, generates a summary of the conversation, and creates a new remote session with the full context. You can then choose to archive the local session or keep it. This requires a clean working tree, and is not available for SSH sessions.
* **Your IDE**: opens your project in a supported IDE at the current working directory.

### Sessions from Dispatch

[Dispatch](https://support.claude.com/en/articles/13947068) is a persistent conversation with Claude that lives in the [Cowork](https://claude.com/product/cowork#dispatch-and-computer-use) tab. You message Dispatch a task, and it decides how to handle it.

A task can end up as a Code session in two ways: you ask for one directly, such as "open a Claude Code session and fix the login bug", or Dispatch decides the task is development work and spawns one on its own. Tasks that typically route to Code include fixing bugs, updating dependencies, running tests, or opening pull requests. Research, document editing, and spreadsheet work stay in Cowork.

Either way, the Code session appears in the Code tab's sidebar with a **Dispatch** badge. You get a push notification on your phone when it finishes or needs your approval.

If you have [computer use](#let-claude-use-your-computer) enabled, Dispatch-spawned Code sessions can use it too. App approvals in those sessions expire after 30 minutes and re-prompt, rather than lasting the full session like regular Code sessions.

For setup, pairing, and Dispatch settings, see the [Dispatch help article](https://support.claude.com/en/articles/13947068). Dispatch requires a Pro or Max plan and is not available on Team or Enterprise plans.

Dispatch is one of several ways to work with Claude when you're away from your terminal. See [Platforms and integrations](/en/platforms#work-when-you-are-away-from-your-terminal) to compare it with Remote Control, Channels, Slack, and scheduled tasks.

## Extend Claude Code

Connect external services, add reusable workflows, customize Claude's behavior, and configure preview servers.

### Connect external tools

For local and [SSH](#ssh-sessions) sessions, click the **+** button next to the prompt box and select **Connectors** to add integrations like Google Calendar, Slack, GitHub, Linear, Notion, and more. You can add connectors before or during a session. The **+** button is not available in remote sessions, but [scheduled tasks](/en/web-scheduled-tasks) configure connectors at task creation time.

To manage or disconnect connectors, go to Settings → Connectors in the desktop app, or select **Manage connectors** from the Connectors menu in the prompt box.

Once connected, Claude can read your calendar, send messages, create issues, and interact with your tools directly. You can ask Claude what connectors are configured in your session.

Connectors are [MCP servers](/en/mcp) with a graphical setup flow. Use them for quick integration with supported services. For integrations not listed in Connectors, add MCP servers manually via [settings files](/en/mcp#installing-mcp-servers). You can also [create custom connectors](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp).

### Use skills

[Skills](/en/skills) extend what Claude can do. Claude loads them automatically when relevant, or you can invoke one directly: type `/` in the prompt box or click the **+** button and select **Slash commands** to browse what's available. This includes [built-in commands](/en/commands), your [custom skills](/en/skills#create-your-first-skill), project skills from your codebase, and skills from any [installed plugins](/en/plugins). Select one and it appears highlighted in the input field. Type your task after it and send as usual.

### Install plugins

[Plugins](/en/plugins) are reusable packages that add skills, agents, hooks, MCP servers, and LSP configurations to Claude Code. You can install plugins from the desktop app without using the terminal.

For local and [SSH](#ssh-sessions) sessions, click the **+** button next to the prompt box and select **Plugins** to see your installed plugins and their commands. To add a plugin, select **Add plugin** from the submenu to open the plugin browser, which shows available plugins from your configured [marketplaces](/en/plugin-marketplaces) including the official Anthropic marketplace. Select **Manage plugins** to enable, disable, or uninstall plugins.

Plugins can be scoped to your user account, a specific project, or local-only. Plugins are not available for remote sessions. For the full plugin reference including creating your own plugins, see [plugins](/en/plugins).

### Configure preview servers

Claude automatically detects your dev server setup and stores the configuration in `.claude/launch.json` at the root of the folder you selected when starting the session. Preview uses this folder as its working directory, so if you selected a parent folder, subfolders with their own dev servers won't be detected automatically. To work with a subfolder's server, either start a session in that folder directly or add a configuration manually.

To customize how your server starts, for example to use `yarn dev` instead of `npm run dev` or to change the port, edit the file manually or click **Edit configuration** in the Preview dropdown to open it in your code editor. The file supports JSON with comments.

```json  theme={null}
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "my-app",
      "runtimeExecutable": "npm",
      "runtimeArgs": ["run", "dev"],
      "port": 3000
    }
  ]
}
```

You can define multiple configurations to run different servers from the same project, such as a frontend and an API. See the [examples](#examples) below.

#### Auto-verify changes

When `autoVerify` is enabled, Claude automatically verifies code changes after editing files. It takes screenshots, checks for errors, and confirms changes work before completing its response.

Auto-verify is on by default. Disable it per-project by adding `"autoVerify": false` to `.claude/launch.json`, or toggle it from the **Preview** dropdown menu.

```json  theme={null}
{
  "version": "0.0.1",
  "autoVerify": false,
  "configurations": [...]
}
```

When disabled, preview tools are still available and you can ask Claude to verify at any time. Auto-verify makes it automatic after every edit.

#### Configuration fields

Each entry in the `configurations` array accepts the following fields:

| Field               | Type      | Description                                                                                                                                                                                                                    |
| ------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`              | string    | A unique identifier for this server                                                                                                                                                                                            |
| `runtimeExecutable` | string    | The command to run, such as `npm`, `yarn`, or `node`                                                                                                                                                                           |
| `runtimeArgs`       | string\[] | Arguments passed to `runtimeExecutable`, such as `["run", "dev"]`                                                                                                                                                              |
| `port`              | number    | The port your server listens on. Defaults to 3000                                                                                                                                                                              |
| `cwd`               | string    | Working directory relative to your project root. Defaults to the project root. Use `${workspaceFolder}` to reference the project root explicitly                                                                               |
| `env`               | object    | Additional environment variables as key-value pairs, such as `{ "NODE_ENV": "development" }`. Don't put secrets here since this file is committed to your repo. Secrets set in your shell profile are inherited automatically. |
| `autoPort`          | boolean   | How to handle port conflicts. See below                                                                                                                                                                                        |
| `program`           | string    | A script to run with `node`. See [when to use `program` vs `runtimeExecutable`](#when-to-use-program-vs-runtimeexecutable)                                                                                                     |
| `args`              | string\[] | Arguments passed to `program`. Only used when `program` is set                                                                                                                                                                 |

##### When to use `program` vs `runtimeExecutable`

Use `runtimeExecutable` with `runtimeArgs` to start a dev server through a package manager. For example, `"runtimeExecutable": "npm"` with `"runtimeArgs": ["run", "dev"]` runs `npm run dev`.

Use `program` when you have a standalone script you want to run with `node` directly. For example, `"program": "server.js"` runs `node server.js`. Pass additional flags with `args`.

#### Port conflicts

The `autoPort` field controls what happens when your preferred port is already in use:

* **`true`**: Claude finds and uses a free port automatically. Suitable for most dev servers.
* **`false`**: Claude fails with an error. Use this when your server must use a specific port, such as for OAuth callbacks or CORS allowlists.
* **Not set (default)**: Claude asks whether the server needs that exact port, then saves your answer.

When Claude picks a different port, it passes the assigned port to your server via the `PORT` environment variable.

#### Examples

These configurations show common setups for different project types:

<Tabs>
  <Tab title="Next.js">
    This configuration runs a Next.js app using Yarn on port 3000:

    ```json  theme={null}
    {
      "version": "0.0.1",
      "configurations": [
        {
          "name": "web",
          "runtimeExecutable": "yarn",
          "runtimeArgs": ["dev"],
          "port": 3000
        }
      ]
    }
    ```
  </Tab>

  <Tab title="Multiple servers">
    For a monorepo with a frontend and an API server, define multiple configurations. The frontend uses `autoPort: true` so it picks a free port if 3000 is taken, while the API server requires port 8080 exactly:

    ```json  theme={null}
    {
      "version": "0.0.1",
      "configurations": [
        {
          "name": "frontend",
          "runtimeExecutable": "npm",
          "runtimeArgs": ["run", "dev"],
          "cwd": "apps/web",
          "port": 3000,
          "autoPort": true
        },
        {
          "name": "api",
          "runtimeExecutable": "npm",
          "runtimeArgs": ["run", "start"],
          "cwd": "server",
          "port": 8080,
          "env": { "NODE_ENV": "development" },
          "autoPort": false
        }
      ]
    }
    ```
  </Tab>

  <Tab title="Node.js script">
    To run a Node.js script directly instead of using a package manager command, use the `program` field:

    ```json  theme={null}
    {
      "version": "0.0.1",
      "configurations": [
        {
          "name": "server",
          "program": "server.js",
          "args": ["--verbose"],
          "port": 4000
        }
      ]
    }
    ```
  </Tab>
</Tabs>

## Environment configuration

The environment you pick when [starting a session](#start-a-session) determines where Claude executes and how you connect:

* **Local**: runs on your machine with direct access to your files
* **Remote**: runs on Anthropic's cloud infrastructure. Sessions continue even if you close the app.
* **SSH**: runs on a remote machine you connect to over SSH, such as your own servers, cloud VMs, or dev containers

### Local sessions

Local sessions inherit environment variables from your shell. If you need additional variables, set them in your shell profile, such as `~/.zshrc` or `~/.bashrc`, and restart the desktop app. See [environment variables](/en/env-vars) for the full list of supported variables.

[Extended thinking](/en/common-workflows#use-extended-thinking-thinking-mode) is enabled by default, which improves performance on complex reasoning tasks but uses additional tokens. To disable thinking entirely, set `MAX_THINKING_TOKENS=0` in your shell profile. On Opus, `MAX_THINKING_TOKENS` is ignored except for `0` because adaptive reasoning controls thinking depth instead.

### Remote sessions

Remote sessions continue in the background even if you close the app. Usage counts toward your [subscription plan limits](/en/costs) with no separate compute charges.

You can create custom cloud environments with different network access levels and environment variables. Select the environment dropdown when starting a remote session and choose **Add environment**. See [cloud environments](/en/claude-code-on-the-web#cloud-environment) for details on configuring network access and environment variables.

### SSH sessions

SSH sessions let you run Claude Code on a remote machine while using the desktop app as your interface. This is useful for working with codebases that live on cloud VMs, dev containers, or servers with specific hardware or dependencies.

To add an SSH connection, click the environment dropdown before starting a session and select **+ Add SSH connection**. The dialog asks for:

* **Name**: a friendly label for this connection
* **SSH Host**: `user@hostname` or a host defined in `~/.ssh/config`
* **SSH Port**: defaults to 22 if left empty, or uses the port from your SSH config
* **Identity File**: path to your private key, such as `~/.ssh/id_rsa`. Leave empty to use the default key or your SSH config.

Once added, the connection appears in the environment dropdown. Select it to start a session on that machine. Claude runs on the remote machine with access to its files and tools.

Claude Code must be installed on the remote machine. Once connected, SSH sessions support permission modes, connectors, plugins, and MCP servers.

## Enterprise configuration

Organizations on Team or Enterprise plans can manage desktop app behavior through admin console controls, managed settings files, and device management policies.

### Admin console controls

These settings are configured through the [admin settings console](https://claude.ai/admin-settings/claude-code):

* **Code in the desktop**: control whether users in your organization can access Claude Code in the desktop app
* **Code in the web**: enable or disable [web sessions](/en/claude-code-on-the-web) for your organization
* **Remote Control**: enable or disable [Remote Control](/en/remote-control) for your organization
* **Disable Bypass permissions mode**: prevent users in your organization from enabling bypass permissions mode

### Managed settings

Managed settings override project and user settings and apply when Desktop spawns CLI sessions. You can set these keys in your organization's [managed settings](/en/settings#settings-precedence) file or push them remotely through the admin console.

| Key                                        | Description                                                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `permissions.disableBypassPermissionsMode` | set to `"disable"` to prevent users from enabling Bypass permissions mode.                                                                                                                    |
| `disableAutoMode`                          | set to `"disable"` to prevent users from enabling [Auto](/en/permission-modes#eliminate-prompts-with-auto-mode) mode. Removes Auto from the mode selector. Also accepted under `permissions`. |
| `autoMode`                                 | customize what the auto mode classifier trusts and blocks across your organization. See [Configure the auto mode classifier](/en/permissions#configure-the-auto-mode-classifier).             |

`permissions.disableBypassPermissionsMode` and `disableAutoMode` also work in user and project settings, but placing them in managed settings prevents users from overriding them. `autoMode` is read from user settings, `.claude/settings.local.json`, and managed settings, but not from the checked-in `.claude/settings.json`: a cloned repo cannot inject its own classifier rules. For the complete list of managed-only settings including `allowManagedPermissionRulesOnly` and `allowManagedHooksOnly`, see [managed-only settings](/en/permissions#managed-only-settings).

Remote managed settings uploaded through the admin console currently apply to CLI and IDE sessions only. For Desktop-specific restrictions, use the admin console controls above.

### Device management policies

IT teams can manage the desktop app through MDM on macOS or group policy on Windows. Available policies include enabling or disabling the Claude Code feature, controlling auto-updates, and setting a custom deployment URL.

* **macOS**: configure via `com.anthropic.Claude` preference domain using tools like Jamf or Kandji
* **Windows**: configure via registry at `SOFTWARE\Policies\Claude`

### Authentication and SSO

Enterprise organizations can require SSO for all users. See [authentication](/en/authentication) for plan-level details and [Setting up SSO](https://support.claude.com/en/articles/13132885-setting-up-single-sign-on-sso) for SAML and OIDC configuration.

### Data handling

Claude Code processes your code locally in local sessions or on Anthropic's cloud infrastructure in remote sessions. Conversations and code context are sent to Anthropic's API for processing. See [data handling](/en/data-usage) for details on data retention, privacy, and compliance.

### Deployment

Desktop can be distributed through enterprise deployment tools:

* **macOS**: distribute via MDM such as Jamf or Kandji using the `.dmg` installer
* **Windows**: deploy via MSIX package or `.exe` installer. See [Deploy Claude Desktop for Windows](https://support.claude.com/en/articles/12622703-deploy-claude-desktop-for-windows) for enterprise deployment options including silent installation

For network configuration such as proxy settings, firewall allowlisting, and LLM gateways, see [network configuration](/en/network-config).

For the full enterprise configuration reference, see the [enterprise configuration guide](https://support.claude.com/en/articles/12622667-enterprise-configuration).

## Coming from the CLI?

If you already use the Claude Code CLI, Desktop runs the same underlying engine with a graphical interface. You can run both simultaneously on the same machine, even on the same project. Each maintains separate session history, but they share configuration and project memory via CLAUDE.md files.

To move a CLI session into Desktop, run `/desktop` in the terminal. Claude saves your session and opens it in the desktop app, then exits the CLI. This command is available on macOS and Windows only.

<Tip>
  When to use Desktop vs CLI: use Desktop when you want visual diff review, file attachments, or session management in a sidebar. Use the CLI when you need scripting, automation, third-party providers, or prefer a terminal workflow.
</Tip>

### CLI flag equivalents

This table shows the desktop app equivalent for common CLI flags. Flags not listed have no desktop equivalent because they are designed for scripting or automation.

| CLI                                   | Desktop equivalent                                                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--model sonnet`                      | Model dropdown next to the send button, before starting a session                                                                        |
| `--resume`, `--continue`              | Click a session in the sidebar                                                                                                           |
| `--permission-mode`                   | Mode selector next to the send button                                                                                                    |
| `--dangerously-skip-permissions`      | Bypass permissions mode. Enable in Settings → Claude Code → "Allow bypass permissions mode". Enterprise admins can disable this setting. |
| `--add-dir`                           | Add multiple repos with the **+** button in remote sessions                                                                              |
| `--allowedTools`, `--disallowedTools` | Not available in Desktop                                                                                                                 |
| `--verbose`                           | Not available. Check system logs: Console.app on macOS, Event Viewer → Windows Logs → Application on Windows                             |
| `--print`, `--output-format`          | Not available. Desktop is interactive only.                                                                                              |
| `ANTHROPIC_MODEL` env var             | Model dropdown next to the send button                                                                                                   |
| `MAX_THINKING_TOKENS` env var         | Set in shell profile; applies to local sessions. See [environment configuration](#environment-configuration).                            |

### Shared configuration

Desktop and CLI read the same configuration files, so your setup carries over:

* **[CLAUDE.md](/en/memory)** and `CLAUDE.local.md` files in your project are used by both
* **[MCP servers](/en/mcp)** configured in `~/.claude.json` or `.mcp.json` work in both
* **[Hooks](/en/hooks)** and **[skills](/en/skills)** defined in settings apply to both
* **[Settings](/en/settings)** in `~/.claude.json` and `~/.claude/settings.json` are shared. Permission rules, allowed tools, and other settings in `settings.json` apply to Desktop sessions.
* **Models**: Sonnet, Opus, and Haiku are available in both. In Desktop, select the model from the dropdown next to the send button before starting a session. You cannot change the model during an active session.

<Note>
  **MCP servers: desktop chat app vs Claude Code**: MCP servers configured for the Claude Desktop chat app in `claude_desktop_config.json` are separate from Claude Code and will not appear in the Code tab. To use MCP servers in Claude Code, configure them in `~/.claude.json` or your project's `.mcp.json` file. See [MCP configuration](/en/mcp#installing-mcp-servers) for details.
</Note>

### Feature comparison

This table compares core capabilities between the CLI and Desktop. For a full list of CLI flags, see the [CLI reference](/en/cli-reference).

| Feature                                               | CLI                                                       | Desktop                                                                                     |
| ----------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Permission modes                                      | All modes including `dontAsk`                             | Ask permissions, Auto accept edits, Plan mode, Auto, and Bypass permissions via Settings    |
| `--dangerously-skip-permissions`                      | CLI flag                                                  | Bypass permissions mode. Enable in Settings → Claude Code → "Allow bypass permissions mode" |
| [Third-party providers](/en/third-party-integrations) | Bedrock, Vertex, Foundry                                  | Not available. Desktop connects to Anthropic's API directly.                                |
| [MCP servers](/en/mcp)                                | Configure in settings files                               | Connectors UI for local and SSH sessions, or settings files                                 |
| [Plugins](/en/plugins)                                | `/plugin` command                                         | Plugin manager UI                                                                           |
| @mention files                                        | Text-based                                                | With autocomplete; local and SSH sessions only                                              |
| File attachments                                      | Not available                                             | Images, PDFs                                                                                |
| Session isolation                                     | [`--worktree`](/en/cli-reference) flag                    | Automatic worktrees                                                                         |
| Multiple sessions                                     | Separate terminals                                        | Sidebar tabs                                                                                |
| Recurring tasks                                       | Cron jobs, CI pipelines                                   | [Scheduled tasks](/en/desktop-scheduled-tasks)                                              |
| Computer use                                          | [Enable via `/mcp`](/en/computer-use) on macOS            | [App and screen control](#let-claude-use-your-computer) on macOS and Windows                |
| Dispatch integration                                  | Not available                                             | [Dispatch sessions](#sessions-from-dispatch) in the sidebar                                 |
| Scripting and automation                              | [`--print`](/en/cli-reference), [Agent SDK](/en/headless) | Not available                                                                               |

### What's not available in Desktop

The following features are only available in the CLI or VS Code extension:

* **Third-party providers**: Desktop connects to Anthropic's API directly. Use the [CLI](/en/quickstart) with Bedrock, Vertex, or Foundry instead.
* **Linux**: the desktop app is available on macOS and Windows only.
* **Inline code suggestions**: Desktop does not provide autocomplete-style suggestions. It works through conversational prompts and explicit code changes.
* **Agent teams**: multi-agent orchestration is available via the [CLI](/en/agent-teams) and [Agent SDK](/en/headless), not in Desktop.

## Troubleshooting

### Check your version

To see which version of the desktop app you're running:

* **macOS**: click **Claude** in the menu bar, then **About Claude**
* **Windows**: click **Help**, then **About**

Click the version number to copy it to your clipboard.

### 403 or authentication errors in the Code tab

If you see `Error 403: Forbidden` or other authentication failures when using the Code tab:

1. Sign out and back in from the app menu. This is the most common fix.
2. Verify you have an active paid subscription: Pro, Max, Team, or Enterprise.
3. If the CLI works but Desktop does not, quit the desktop app completely, not just close the window, then reopen and sign in again.
4. Check your internet connection and proxy settings.

### Blank or stuck screen on launch

If the app opens but shows a blank or unresponsive screen:

1. Restart the app.
2. Check for pending updates. The app auto-updates on launch.
3. On Windows, check Event Viewer for crash logs under **Windows Logs → Application**.

### "Failed to load session"

If you see `Failed to load session`, the selected folder may no longer exist, a Git repository may require Git LFS that isn't installed, or file permissions may prevent access. Try selecting a different folder or restarting the app.

### Session not finding installed tools

If Claude can't find tools like `npm`, `node`, or other CLI commands, verify the tools work in your regular terminal, check that your shell profile properly sets up PATH, and restart the desktop app to reload environment variables.

### Git and Git LFS errors

On Windows, Git is required for the Code tab to start local sessions. If you see "Git is required," install [Git for Windows](https://git-scm.com/downloads/win) and restart the app.

If you see "Git LFS is required by this repository but is not installed," install Git LFS from [git-lfs.com](https://git-lfs.com/), run `git lfs install`, and restart the app.

### MCP servers not working on Windows

If MCP server toggles don't respond or servers fail to connect on Windows, check that the server is properly configured in your settings, restart the app, verify the server process is running in Task Manager, and review server logs for connection errors.

### App won't quit

* **macOS**: press Cmd+Q. If the app doesn't respond, use Force Quit with Cmd+Option+Esc, select Claude, and click Force Quit.
* **Windows**: use Task Manager with Ctrl+Shift+Esc to end the Claude process.

### Windows-specific issues

* **PATH not updated after install**: open a new terminal window. PATH updates only apply to new terminal sessions.
* **Concurrent installation error**: if you see an error about another installation in progress but there isn't one, try running the installer as Administrator.
* **ARM64**: Windows ARM64 devices are fully supported.

### Cowork tab unavailable on Intel Macs

The Cowork tab requires Apple Silicon (M1 or later) on macOS. On Windows, Cowork is available on all supported hardware. The Chat and Code tabs work normally on Intel Macs.

### "Branch doesn't exist yet" when opening in CLI

Remote sessions can create branches that don't exist on your local machine. Click the branch name in the session toolbar to copy it, then fetch it locally:

```bash  theme={null}
git fetch origin <branch-name>
git checkout <branch-name>
```

### Still stuck?

* Search or file a bug on [GitHub Issues](https://github.com/anthropics/claude-code/issues)
* Visit the [Claude support center](https://support.claude.com/)

When filing a bug, include your desktop app version, your operating system, the exact error message, and relevant logs. On macOS, check Console.app. On Windows, check Event Viewer → Windows Logs → Application.
