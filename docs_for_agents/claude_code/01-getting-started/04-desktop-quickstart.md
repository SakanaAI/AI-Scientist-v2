# Get started with the desktop app

Install Claude Code on desktop and start your first coding session

The desktop app gives you Claude Code with a graphical interface: visual diff review, live app preview, GitHub PR monitoring with auto-merge, parallel sessions with Git worktree isolation, scheduled tasks, and the ability to run tasks remotely. No terminal required.

Download Claude for your platform:

- **macOS**: Universal build for Intel and Apple Silicon
- **Windows**: For x64 processors

For Windows ARM64, download the [ARM64 installer](https://claude.ai/api/desktop/win32/arm64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs). Linux is not currently supported.

> Claude Code requires a [Pro, Max, Team, or Enterprise subscription](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=desktop_quickstart_pricing).

This page walks through installing the app and starting your first session. If you're already set up, see [Use Claude Code Desktop](/en/desktop) for the full reference.

The desktop app has three tabs:

* **Chat**: General conversation with no file access, similar to claude.ai.
* **Cowork**: An autonomous background agent that works on tasks in a cloud VM with its own environment. It can run independently while you do other work.
* **Code**: An interactive coding assistant with direct access to your local files. You review and approve each change in real time.

Chat and Cowork are covered in the [Claude Desktop support articles](https://support.claude.com/en/collections/16163169-claude-desktop). This page focuses on the **Code** tab.

## Install

**Step 1: Install and sign in**

Download Claude for your platform and run the installer:

* [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs): universal build for Intel and Apple Silicon
* [Windows x64](https://claude.ai/api/desktop/win32/x64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs): for x64 processors
* [Windows ARM64](https://claude.ai/api/desktop/win32/arm64/setup/latest/redirect?utm_source=claude_code&utm_medium=docs): for ARM processors

Launch Claude from your Applications folder (macOS) or Start menu (Windows). Sign in with your Anthropic account.

**Step 2: Open the Code tab**

Click the **Code** tab at the top center. If clicking Code prompts you to upgrade, you need to [subscribe to a paid plan](https://claude.com/pricing?utm_source=claude_code&utm_medium=docs&utm_content=desktop_quickstart_upgrade) first. If it prompts you to sign in online, complete the sign-in and restart the app. If you see a 403 error, see [authentication troubleshooting](/en/desktop#403-or-authentication-errors-in-the-code-tab).

The desktop app includes Claude Code. You don't need to install Node.js or the CLI separately. To use `claude` from the terminal, install the CLI separately. See [Get started with the CLI](/en/quickstart).

## Start your first session

With the Code tab open, choose a project and give Claude something to do.

**Step 1: Choose an environment and folder**

Select **Local** to run Claude on your machine using your files directly. Click **Select folder** and choose your project directory.

> Start with a small project you know well. It's the fastest way to see what Claude Code can do. On Windows, [Git](https://git-scm.com/downloads/win) must be installed for local sessions to work. Most Macs include Git by default.

You can also select:

* **Remote**: Run sessions on Anthropic's cloud infrastructure that continue even if you close the app. Remote sessions use the same infrastructure as [Claude Code on the web](/en/claude-code-on-the-web).
* **SSH**: Connect to a remote machine over SSH (your own servers, cloud VMs, or dev containers). Claude Code must be installed on the remote machine.

**Step 2: Choose a model**

Select a model from the dropdown next to the send button. See [models](/en/model-config#available-models) for a comparison of Opus, Sonnet, and Haiku. You cannot change the model after the session starts.

**Step 3: Tell Claude what to do**

Type what you want Claude to do:

* `Find a TODO comment and fix it`
* `Add tests for the main function`
* `Create a CLAUDE.md with instructions for this codebase`

A [session](/en/desktop#work-in-parallel-with-sessions) is a conversation with Claude about your code. Each session tracks its own context and changes, so you can work on multiple tasks without them interfering with each other.

**Step 4: Review and accept changes**

By default, the Code tab starts in [Ask permissions mode](/en/desktop#choose-a-permission-mode), where Claude proposes changes and waits for your approval before applying them. You'll see:

1. A [diff view](/en/desktop#review-changes-with-diff-view) showing exactly what will change in each file
2. Accept/Reject buttons to approve or decline each change
3. Real-time updates as Claude works through your request

If you reject a change, Claude will ask how you'd like to proceed differently. Your files aren't modified until you accept.

## Now what?

You've made your first edit. For the full reference on everything Desktop can do, see [Use Claude Code Desktop](/en/desktop). Here are some things to try next.

**Interrupt and steer.** You can interrupt Claude at any point. If it's going down the wrong path, click the stop button or type your correction and press **Enter**. Claude stops what it's doing and adjusts based on your input. You don't have to wait for it to finish or start over.

**Give Claude more context.** Type `@filename` in the prompt box to pull a specific file into the conversation, attach images and PDFs using the attachment button, or drag and drop files directly into the prompt. The more context Claude has, the better the results. See [Add files and context](/en/desktop#add-files-and-context-to-prompts).

**Use skills for repeatable tasks.** Type `/` or click **+** → **Slash commands** to browse [built-in commands](/en/commands), [custom skills](/en/skills), and plugin skills. Skills are reusable prompts you can invoke whenever you need them, like code review checklists or deployment steps.

**Review changes before committing.** After Claude edits files, a `+12 -1` indicator appears. Click it to open the [diff view](/en/desktop#review-changes-with-diff-view), review modifications file by file, and comment on specific lines. Claude reads your comments and revises. Click **Review code** to have Claude evaluate the diffs itself and leave inline suggestions.

**Adjust how much control you have.** Your [permission mode](/en/desktop#choose-a-permission-mode) controls the balance. Ask permissions (default) requires approval before every edit. Auto accept edits auto-accepts file edits for faster iteration. Plan mode lets Claude map out an approach without touching any files, which is useful before a large refactor.

**Add plugins for more capabilities.** Click the **+** button next to the prompt box and select **Plugins** to browse and install [plugins](/en/desktop#install-plugins) that add skills, agents, MCP servers, and more.

**Preview your app.** Click the **Preview** dropdown to run your dev server directly in the desktop. Claude can view the running app, test endpoints, inspect logs, and iterate on what it sees. See [Preview your app](/en/desktop#preview-your-app).

**Track your pull request.** After opening a PR, Claude Code monitors CI check results and can automatically fix failures or merge the PR once all checks pass. See [Monitor pull request status](/en/desktop#monitor-pull-request-status).

**Put Claude on a schedule.** Set up [scheduled tasks](/en/desktop-scheduled-tasks) to run Claude automatically on a recurring basis: a daily code review every morning, a weekly dependency audit, or a briefing that pulls from your connected tools.

**Scale up when you're ready.** Open [parallel sessions](/en/desktop#work-in-parallel-with-sessions) from the sidebar to work on multiple tasks at once, each in its own Git worktree. Send [long-running work to the cloud](/en/desktop#run-long-running-tasks-remotely) so it continues even if you close the app, or [continue a session on the web or in your IDE](/en/desktop#continue-in-another-surface) if a task takes longer than expected. [Connect external tools](/en/desktop#extend-claude-code) like GitHub, Slack, and Linear to bring your workflow together.

## Coming from the CLI?

Desktop runs the same engine as the CLI with a graphical interface. You can run both simultaneously on the same project, and they share configuration (CLAUDE.md files, MCP servers, hooks, skills, and settings). For a full comparison of features, flag equivalents, and what's not available in Desktop, see [CLI comparison](/en/desktop#coming-from-the-cli).

## What's next

* [Use Claude Code Desktop](/en/desktop): permission modes, parallel sessions, diff view, connectors, and enterprise configuration
* [Troubleshooting](/en/desktop#troubleshooting): solutions to common errors and setup issues
* [Best practices](/en/best-practices): tips for writing effective prompts and getting the most out of Claude Code
* [Common workflows](/en/common-workflows): tutorials for debugging, refactoring, testing, and more
