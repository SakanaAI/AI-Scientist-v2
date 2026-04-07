# Platforms and integrations

> Choose where to run Claude Code and what to connect it to. Compare the CLI, Desktop, VS Code, JetBrains, web, and integrations like Chrome, Slack, and CI/CD.

Claude Code runs the same underlying engine everywhere, but each surface is tuned for a different way of working. This page helps you pick the right platform for your workflow and connect the tools you already use.

## Where to run Claude Code

Choose a platform based on how you like to work and where your project lives.

| Platform                          | Best for                                                                                           | What you get                                                                                                                                         |
| :-------------------------------- | :------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| [CLI](/en/quickstart)             | Terminal workflows, scripting, remote servers                                                      | Full feature set, [Agent SDK](/en/headless), [computer use](/en/computer-use) on macOS (Pro and Max), third-party providers                          |
| [Desktop](/en/desktop)            | Visual review, parallel sessions, managed setup                                                    | Diff viewer, app preview, [computer use](/en/desktop#let-claude-use-your-computer) and [Dispatch](/en/desktop#sessions-from-dispatch) on Pro and Max |
| [VS Code](/en/vs-code)            | Working inside VS Code without switching to a terminal                                             | Inline diffs, integrated terminal, file context                                                                                                      |
| [JetBrains](/en/jetbrains)        | Working inside IntelliJ, PyCharm, WebStorm, or other JetBrains IDEs                                | Diff viewer, selection sharing, terminal session                                                                                                     |
| [Web](/en/claude-code-on-the-web) | Long-running tasks that don't need much steering, or work that should continue when you're offline | Anthropic-managed cloud, continues after you disconnect                                                                                              |

The CLI is the most complete surface for terminal-native work: scripting, third-party providers, and the Agent SDK are CLI-only. Desktop and the IDE extensions trade some CLI-only features for visual review and tighter editor integration. The web runs in Anthropic's cloud, so tasks keep going after you disconnect.

You can mix surfaces on the same project. Configuration, project memory, and MCP servers are shared across the local surfaces.

## Connect your tools

Integrations let Claude work with services outside your codebase.

| Integration                          | What it does                                       | Use it for                                                       |
| :----------------------------------- | :------------------------------------------------- | :--------------------------------------------------------------- |
| [Chrome](/en/chrome)                 | Controls your browser with your logged-in sessions | Testing web apps, filling forms, automating sites without an API |
| [GitHub Actions](/en/github-actions) | Runs Claude in your CI pipeline                    | Automated PR reviews, issue triage, scheduled maintenance        |
| [GitLab CI/CD](/en/gitlab-ci-cd)     | Same as GitHub Actions for GitLab                  | CI-driven automation on GitLab                                   |
| [Code Review](/en/code-review)       | Reviews every PR automatically                     | Catching bugs before human review                                |
| [Slack](/en/slack)                   | Responds to `@Claude` mentions in your channels    | Turning bug reports into pull requests from team chat            |

For integrations not listed here, [MCP servers](/en/mcp) and [connectors](/en/desktop#connect-external-tools) let you connect almost anything: Linear, Notion, Google Drive, or your own internal APIs.

## Work when you are away from your terminal

Claude Code offers several ways to work when you're not at your terminal. They differ in what triggers the work, where Claude runs, and how much you need to set up.

|                                                | Trigger                                                                                        | Claude runs on                                                                                          | Setup                                                                                                                                | Best for                                                      |
| :--------------------------------------------- | :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| [Dispatch](/en/desktop#sessions-from-dispatch) | Message a task from the Claude mobile app                                                      | Your machine (Desktop)                                                                                  | [Pair the mobile app with Desktop](https://support.claude.com/en/articles/13947068)                                                  | Delegating work while you're away, minimal setup              |
| [Remote Control](/en/remote-control)           | Drive a running session from [claude.ai/code](https://claude.ai/code) or the Claude mobile app | Your machine (CLI or VS Code)                                                                           | Run `claude remote-control`                                                                                                          | Steering in-progress work from another device                 |
| [Channels](/en/channels)                       | Push events from a chat app like Telegram or Discord, or your own server                       | Your machine (CLI)                                                                                      | [Install a channel plugin](/en/channels#quickstart) or [build your own](/en/channels-reference)                                      | Reacting to external events like CI failures or chat messages |
| [Slack](/en/slack)                             | Mention `@Claude` in a team channel                                                            | Anthropic cloud                                                                                         | [Install the Slack app](/en/slack#setting-up-claude-code-in-slack) with [Claude Code on the web](/en/claude-code-on-the-web) enabled | PRs and reviews from team chat                                |
| [Scheduled tasks](/en/scheduled-tasks)         | Set a schedule                                                                                 | [CLI](/en/scheduled-tasks), [Desktop](/en/desktop-scheduled-tasks), or [cloud](/en/web-scheduled-tasks) | Pick a frequency                                                                                                                     | Recurring automation like daily reviews                       |

If you're not sure where to start, [install the CLI](/en/quickstart) and run it in a project directory. If you'd rather not use a terminal, [Desktop](/en/desktop-quickstart) gives you the same engine with a graphical interface.

## Related resources

### Platforms

* [CLI quickstart](/en/quickstart): install and run your first command in the terminal
* [Desktop](/en/desktop): visual diff review, parallel sessions, computer use, and Dispatch
* [VS Code](/en/vs-code): the Claude Code extension inside your editor
* [JetBrains](/en/jetbrains): the extension for IntelliJ, PyCharm, and other JetBrains IDEs
* [Claude Code on the web](/en/claude-code-on-the-web): cloud sessions that keep running when you disconnect

### Integrations

* [Chrome](/en/chrome): automate browser tasks with your logged-in sessions
* [Computer use](/en/computer-use): let Claude open apps and control your screen on macOS
* [GitHub Actions](/en/github-actions): run Claude in your CI pipeline
* [GitLab CI/CD](/en/gitlab-ci-cd): the same for GitLab
* [Code Review](/en/code-review): automatic review on every pull request
* [Slack](/en/slack): send tasks from team chat, get PRs back

### Remote access

* [Dispatch](/en/desktop#sessions-from-dispatch): message a task from your phone and it can spawn a Desktop session
* [Remote Control](/en/remote-control): drive a running session from your phone or browser
* [Channels](/en/channels): push events from chat apps or your own servers into a session
* [Scheduled tasks](/en/scheduled-tasks): run prompts on a recurring schedule
