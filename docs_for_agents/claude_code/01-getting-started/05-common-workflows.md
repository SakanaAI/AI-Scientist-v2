# Common workflows

Step-by-step guides for exploring codebases, fixing bugs, refactoring, testing, and other everyday tasks with Claude Code.

This page covers practical workflows for everyday development: exploring unfamiliar code, debugging, refactoring, writing tests, creating PRs, and managing sessions. Each section includes example prompts you can adapt to your own projects. For higher-level patterns and tips, see [Best practices](/en/best-practices).

## Understand new codebases

### Get a quick codebase overview

Suppose you've just joined a new project and need to understand its structure quickly.

**Step 1: Navigate to the project root directory**

```bash
cd /path/to/project
```

**Step 2: Start Claude Code**

```bash
claude
```

**Step 3: Ask for a high-level overview**

```text
give me an overview of this codebase
```

**Step 4: Dive deeper into specific components**

```text
explain the main architecture patterns used here
```

```text
what are the key data models?
```

```text
how is authentication handled?
```

Tips:

* Start with broad questions, then narrow down to specific areas
* Ask about coding conventions and patterns used in the project
* Request a glossary of project-specific terms

### Find relevant code

Suppose you need to locate code related to a specific feature or functionality.

**Step 1: Ask Claude to find relevant files**

```text
find the files that handle user authentication
```

**Step 2: Get context on how components interact**

```text
how do these authentication files work together?
```

**Step 3: Understand the execution flow**

```text
trace the login process from front-end to database
```

Tips:

* Be specific about what you're looking for
* Use domain language from the project
* Install a [code intelligence plugin](/en/discover-plugins#code-intelligence) for your language to give Claude precise "go to definition" and "find references" navigation

## Fix bugs efficiently

Suppose you've encountered an error message and need to find and fix its source.

**Step 1: Share the error with Claude**

```text
I'm seeing an error when I run npm test
```

**Step 2: Ask for fix recommendations**

```text
suggest a few ways to fix the @ts-ignore in user.ts
```

**Step 3: Apply the fix**

```text
update user.ts to add the null check you suggested
```

Tips:

* Tell Claude the command to reproduce the issue and get a stack trace
* Mention any steps to reproduce the error
* Let Claude know if the error is intermittent or consistent

## Refactor code

Suppose you need to update old code to use modern patterns and practices.

**Step 1: Identify legacy code for refactoring**

```text
find deprecated API usage in our codebase
```

**Step 2: Get refactoring recommendations**

```text
suggest how to refactor utils.js to use modern JavaScript features
```

**Step 3: Apply the changes safely**

```text
refactor utils.js to use ES2024 features while maintaining the same behavior
```

**Step 4: Verify the refactoring**

```text
run tests for the refactored code
```

Tips:

* Ask Claude to explain the benefits of the modern approach
* Request that changes maintain backward compatibility when needed
* Do refactoring in small, testable increments

## Use specialized subagents

Suppose you want to use specialized AI subagents to handle specific tasks more effectively.

**Step 1: View available subagents**

```text
/agents
```

This shows all available subagents and lets you create new ones.

**Step 2: Use subagents automatically**

Claude Code automatically delegates appropriate tasks to specialized subagents:

```text
review my recent code changes for security issues
```

```text
run all tests and fix any failures
```

**Step 3: Explicitly request specific subagents**

```text
use the code-reviewer subagent to check the auth module
```

```text
have the debugger subagent investigate why users can't log in
```

**Step 4: Create custom subagents for your workflow**

```text
/agents
```

Then select "Create New subagent" and follow the prompts to define:

* A unique identifier that describes the subagent's purpose (for example, `code-reviewer`, `api-designer`).
* When Claude should use this agent
* Which tools it can access
* A system prompt describing the agent's role and behavior

Tips:

* Create project-specific subagents in `.claude/agents/` for team sharing
* Use descriptive `description` fields to enable automatic delegation
* Limit tool access to what each subagent actually needs
* Check the [subagents documentation](/en/sub-agents) for detailed examples

## Use Plan Mode for safe code analysis

Plan Mode instructs Claude to create a plan by analyzing the codebase with read-only operations, perfect for exploring codebases, planning complex changes, or reviewing code safely. In Plan Mode, Claude uses [`AskUserQuestion`](/en/tools-reference) to gather requirements and clarify your goals before proposing a plan.

### When to use Plan Mode

* **Multi-step implementation**: When your feature requires making edits to many files
* **Code exploration**: When you want to research the codebase thoroughly before changing anything
* **Interactive development**: When you want to iterate on the direction with Claude

### How to use Plan Mode

**Turn on Plan Mode during a session**

You can switch into Plan Mode during a session using **Shift+Tab** to cycle through permission modes.

If you are in Normal Mode, **Shift+Tab** first switches into Auto-Accept Mode, indicated by `⏵⏵ accept edits on` at the bottom of the terminal. A subsequent **Shift+Tab** will switch into Plan Mode, indicated by `⏸ plan mode on`.

**Start a new session in Plan Mode**

To start a new session in Plan Mode, use the `--permission-mode plan` flag:

```bash
claude --permission-mode plan
```

**Run "headless" queries in Plan Mode**

You can also run a query in Plan Mode directly with `-p` (that is, in ["headless mode"](/en/headless)):

```bash
claude --permission-mode plan -p "Analyze the authentication system and suggest improvements"
```

### Example: Planning a complex refactor

```bash
claude --permission-mode plan
```

```text
I need to refactor our authentication system to use OAuth2. Create a detailed migration plan.
```

Claude analyzes the current implementation and create a comprehensive plan. Refine with follow-ups:

```text
What about backward compatibility?
```

```text
How should we handle database migration?
```

> Press `Ctrl+G` to open the plan in your default text editor, where you can edit it directly before Claude proceeds.

When you accept a plan, Claude automatically names the session from the plan content. The name appears on the prompt bar and in the session picker. If you've already set a name with `--name` or `/rename`, accepting a plan won't overwrite it.

### Configure Plan Mode as default

```json
// .claude/settings.json
{
  "permissions": {
    "defaultMode": "plan"
  }
}
```

See [settings documentation](/en/settings#available-settings) for more configuration options.

## Work with tests

Suppose you need to add tests for uncovered code.

**Step 1: Identify untested code**

```text
find functions in NotificationsService.swift that are not covered by tests
```

**Step 2: Generate test scaffolding**

```text
add tests for the notification service
```

**Step 3: Add meaningful test cases**

```text
add test cases for edge conditions in the notification service
```

**Step 4: Run and verify tests**

```text
run the new tests and fix any failures
```

Claude can generate tests that follow your project's existing patterns and conventions. When asking for tests, be specific about what behavior you want to verify. Claude examines your existing test files to match the style, frameworks, and assertion patterns already in use.

For comprehensive coverage, ask Claude to identify edge cases you might have missed. Claude can analyze your code paths and suggest tests for error conditions, boundary values, and unexpected inputs that are easy to overlook.

## Create pull requests

You can create pull requests by asking Claude directly ("create a pr for my changes"), or guide Claude through it step-by-step:

**Step 1: Summarize your changes**

```text
summarize the changes I've made to the authentication module
```

**Step 2: Generate a pull request**

```text
create a pr
```

**Step 3: Review and refine**

```text
enhance the PR description with more context about the security improvements
```

When you create a PR using `gh pr create`, the session is automatically linked to that PR. You can resume it later with `claude --from-pr <number>`.

> Review Claude's generated PR before submitting and ask Claude to highlight potential risks or considerations.

## Handle documentation

Suppose you need to add or update documentation for your code.

**Step 1: Identify undocumented code**

```text
find functions without proper JSDoc comments in the auth module
```

**Step 2: Generate documentation**

```text
add JSDoc comments to the undocumented functions in auth.js
```

**Step 3: Review and enhance**

```text
improve the generated documentation with more context and examples
```

**Step 4: Verify documentation**

```text
check if the documentation follows our project standards
```

Tips:

* Specify the documentation style you want (JSDoc, docstrings, etc.)
* Ask for examples in the documentation
* Request documentation for public APIs, interfaces, and complex logic

## Work with images

Suppose you need to work with images in your codebase, and you want Claude's help analyzing image content.

**Step 1: Add an image to the conversation**

You can use any of these methods:

1. Drag and drop an image into the Claude Code window
2. Copy an image and paste it into the CLI with ctrl+v (Do not use cmd+v)
3. Provide an image path to Claude. E.g., "Analyze this image: /path/to/your/image.png"

**Step 2: Ask Claude to analyze the image**

```text
What does this image show?
```

```text
Describe the UI elements in this screenshot
```

```text
Are there any problematic elements in this diagram?
```

**Step 3: Use images for context**

```text
Here's a screenshot of the error. What's causing it?
```

```text
This is our current database schema. How should we modify it for the new feature?
```

**Step 4: Get code suggestions from visual content**

```text
Generate CSS to match this design mockup
```

```text
What HTML structure would recreate this component?
```

Tips:

* Use images when text descriptions would be unclear or cumbersome
* Include screenshots of errors, UI designs, or diagrams for better context
* You can work with multiple images in a conversation
* Image analysis works with diagrams, screenshots, mockups, and more
* When Claude references images (for example, `[Image #1]`), `Cmd+Click` (Mac) or `Ctrl+Click` (Windows/Linux) the link to open the image in your default viewer

## Reference files and directories

Use @ to quickly include files or directories without waiting for Claude to read them.

**Step 1: Reference a single file**

```text
Explain the logic in @src/utils/auth.js
```

This includes the full content of the file in the conversation.

**Step 2: Reference a directory**

```text
What's the structure of @src/components?
```

This provides a directory listing with file information.

**Step 3: Reference MCP resources**

```text
Show me the data from @github:repos/owner/repo/issues
```

This fetches data from connected MCP servers using the format @server:resource. See [MCP resources](/en/mcp#use-mcp-resources) for details.

Tips:

* File paths can be relative or absolute
* @ file references add `CLAUDE.md` in the file's directory and parent directories to context
* Directory references show file listings, not contents
* You can reference multiple files in a single message (for example, "@file1.js and @file2.js")

## Use extended thinking (thinking mode)

[Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) is enabled by default, giving Claude space to reason through complex problems step-by-step before responding. This reasoning is visible in verbose mode, which you can toggle on with `Ctrl+O`.

Additionally, Opus 4.6 and Sonnet 4.6 support adaptive reasoning: instead of a fixed thinking token budget, the model dynamically allocates thinking based on your [effort level](/en/model-config#adjust-effort-level) setting. Extended thinking and adaptive reasoning work together to give you control over how deeply Claude reasons before responding.

Extended thinking is particularly valuable for complex architectural decisions, challenging bugs, multi-step implementation planning, and evaluating tradeoffs between different approaches.

> Phrases like "think", "think hard", and "think more" are interpreted as regular prompt instructions and don't allocate thinking tokens.

### Configure thinking mode

Thinking is enabled by default, but you can adjust or disable it.

| Scope                    | How to configure                                                                     | Details                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Effort level**         | Run `/effort`, adjust in `/model`, or set [`CLAUDE_CODE_EFFORT_LEVEL`](/en/env-vars) | Control thinking depth for Opus 4.6 and Sonnet 4.6. See [Adjust effort level](/en/model-config#adjust-effort-level)                                                                       |
| **`ultrathink` keyword** | Include "ultrathink" anywhere in your prompt                                         | Sets effort to high for that turn on Opus 4.6 and Sonnet 4.6. Useful for one-off tasks requiring deep reasoning without permanently changing your effort setting                          |
| **Toggle shortcut**      | Press `Option+T` (macOS) or `Alt+T` (Windows/Linux)                                  | Toggle thinking on/off for the current session (all models). May require [terminal configuration](/en/terminal-config) to enable Option key shortcuts                                     |
| **Global default**       | Use `/config` to toggle thinking mode                                                | Sets your default across all projects (all models).<br />Saved as `alwaysThinkingEnabled` in `~/.claude/settings.json`                                                                    |
| **Limit token budget**   | Set [`MAX_THINKING_TOKENS`](/en/env-vars) environment variable                       | Limit the thinking budget to a specific number of tokens. On Opus 4.6 and Sonnet 4.6, only `0` applies unless adaptive reasoning is disabled. Example: `export MAX_THINKING_TOKENS=10000` |

To view Claude's thinking process, press `Ctrl+O` to toggle verbose mode and see the internal reasoning displayed as gray italic text.

### How extended thinking works

Extended thinking controls how much internal reasoning Claude performs before responding. More thinking provides more space to explore solutions, analyze edge cases, and self-correct mistakes.

**With Opus 4.6 and Sonnet 4.6**, thinking uses adaptive reasoning: the model dynamically allocates thinking tokens based on the [effort level](/en/model-config#adjust-effort-level) you select. This is the recommended way to tune the tradeoff between speed and reasoning depth.

**With older models**, thinking uses a fixed token budget drawn from your output allocation. The budget varies by model; see [`MAX_THINKING_TOKENS`](/en/env-vars) for per-model ceilings. You can limit the budget with that environment variable, or disable thinking entirely via `/config` or the `Option+T`/`Alt+T` toggle.

On Opus 4.6 and Sonnet 4.6, [adaptive reasoning](/en/model-config#adjust-effort-level) controls thinking depth, so `MAX_THINKING_TOKENS` only applies when set to `0` to disable thinking, or when `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` reverts these models to the fixed budget. See [environment variables](/en/env-vars).

> You're charged for all thinking tokens used even when thinking summaries are redacted. In interactive mode, thinking appears as a collapsed stub by default. Set `showThinkingSummaries: true` in `settings.json` to show full summaries.

## Resume previous conversations

When starting Claude Code, you can resume a previous session:

* `claude --continue` continues the most recent conversation in the current directory
* `claude --resume` opens a conversation picker or resumes by name
* `claude --from-pr 123` resumes sessions linked to a specific pull request

From inside an active session, use `/resume` to switch to a different conversation.

Sessions are stored per project directory. The `/resume` picker shows interactive sessions from the same git repository, including worktrees. Sessions created by `claude -p` or SDK invocations do not appear in the picker, but you can still resume one by passing its session ID directly to `claude --resume <session-id>`.

### Name your sessions

Give sessions descriptive names to find them later. This is a best practice when working on multiple tasks or features.

**Step 1: Name the session**

Name a session at startup with `-n`:

```bash
claude -n auth-refactor
```

Or use `/rename` during a session, which also shows the name on the prompt bar:

```text
/rename auth-refactor
```

You can also rename any session from the picker: run `/resume`, navigate to a session, and press `R`.

**Step 2: Resume by name later**

From the command line:

```bash
claude --resume auth-refactor
```

Or from inside an active session:

```text
/resume auth-refactor
```

### Use the session picker

The `/resume` command (or `claude --resume` without arguments) opens an interactive session picker with these features:

**Keyboard shortcuts in the picker:**

| Shortcut  | Action                                            |
| :-------- | :------------------------------------------------ |
| `↑` / `↓` | Navigate between sessions                         |
| `→` / `←` | Expand or collapse grouped sessions               |
| `Enter`   | Select and resume the highlighted session         |
| `P`       | Preview the session content                       |
| `R`       | Rename the highlighted session                    |
| `/`       | Search to filter sessions                         |
| `A`       | Toggle between current directory and all projects |
| `B`       | Filter to sessions from your current git branch   |
| `Esc`     | Exit the picker or search mode                    |

**Session organization:**

The picker displays sessions with helpful metadata:

* Session name or initial prompt
* Time elapsed since last activity
* Message count
* Git branch (if applicable)

Forked sessions (created with `/branch`, `/rewind`, or `--fork-session`) are grouped together under their root session, making it easier to find related conversations.

Tips:

* **Name sessions early**: Use `/rename` when starting work on a distinct task: it's much easier to find "payment-integration" than "explain this function" later
* Use `--continue` for quick access to your most recent conversation in the current directory
* Use `--resume session-name` when you know which session you need
* Use `--resume` (without a name) when you need to browse and select
* For scripts, use `claude --continue --print "prompt"` to resume in non-interactive mode
* Press `P` in the picker to preview a session before resuming it
* The resumed conversation starts with the same model and configuration as the original

How it works:

1. **Conversation Storage**: All conversations are automatically saved locally with their full message history
2. **Message Deserialization**: When resuming, the entire message history is restored to maintain context
3. **Tool State**: Tool usage and results from the previous conversation are preserved
4. **Context Restoration**: The conversation resumes with all previous context intact

## Run parallel Claude Code sessions with Git worktrees

When working on multiple tasks at once, you need each Claude session to have its own copy of the codebase so changes don't collide. Git worktrees solve this by creating separate working directories that each have their own files and branch, while sharing the same repository history and remote connections. This means you can have Claude working on a feature in one worktree while fixing a bug in another, without either session interfering with the other.

Use the `--worktree` (`-w`) flag to create an isolated worktree and start Claude in it. The value you pass becomes the worktree directory name and branch name:

```bash
# Start Claude in a worktree named "feature-auth"
# Creates .claude/worktrees/feature-auth/ with a new branch
claude --worktree feature-auth

# Start another session in a separate worktree
claude --worktree bugfix-123
```

If you omit the name, Claude generates a random one automatically:

```bash
# Auto-generates a name like "bright-running-fox"
claude --worktree
```

Worktrees are created at `<repo>/.claude/worktrees/<name>` and branch from the default remote branch, which is where `origin/HEAD` points. The worktree branch is named `worktree-<name>`.

The base branch is not configurable through a Claude Code flag or setting. `origin/HEAD` is a reference stored in your local `.git` directory that Git set once when you cloned. If the repository's default branch later changes on GitHub or GitLab, your local `origin/HEAD` keeps pointing at the old one, and worktrees will branch from there. To re-sync your local reference with whatever the remote currently considers its default:

```bash
git remote set-head origin -a
```

This is a standard Git command that only updates your local `.git` directory. Nothing on the remote server changes. If you want worktrees to base off a specific branch rather than the remote's default, set it explicitly with `git remote set-head origin your-branch-name`.

For full control over how worktrees are created, including choosing a different base per invocation, configure a [WorktreeCreate hook](/en/hooks#worktreecreate). The hook replaces Claude Code's default `git worktree` logic entirely, so you can fetch and branch from whatever ref you need.

You can also ask Claude to "work in a worktree" or "start a worktree" during a session, and it will create one automatically.

### Subagent worktrees

Subagents can also use worktree isolation to work in parallel without conflicts. Ask Claude to "use worktrees for your agents" or configure it in a [custom subagent](/en/sub-agents#supported-frontmatter-fields) by adding `isolation: worktree` to the agent's frontmatter. Each subagent gets its own worktree that is automatically cleaned up when the subagent finishes without changes.

### Worktree cleanup

When you exit a worktree session, Claude handles cleanup based on whether you made changes:

* **No changes**: the worktree and its branch are removed automatically
* **Changes or commits exist**: Claude prompts you to keep or remove the worktree. Keeping preserves the directory and branch so you can return later. Removing deletes the worktree directory and its branch, discarding all uncommitted changes and commits

Subagent worktrees orphaned by a crash or an interrupted parallel run are removed automatically at startup once they are older than your [`cleanupPeriodDays`](/en/settings#available-settings) setting, provided they have no modifications to tracked files and no unpushed commits. Untracked files (new files never staged with `git add`) are not checked and do not prevent removal. Worktrees you create with `--worktree` are never removed by this sweep.

To clean up worktrees outside of a Claude session, use [manual worktree management](#manage-worktrees-manually).

> Add `.claude/worktrees/` to your `.gitignore` to prevent worktree contents from appearing as untracked files in your main repository.

### Copy gitignored files to worktrees

Git worktrees are fresh checkouts, so they don't include untracked files like `.env` or `.env.local` from your main repository. To automatically copy these files when Claude creates a worktree, add a `.worktreeinclude` file to your project root.

The file uses `.gitignore` syntax to list which files to copy. Only files that match a pattern and are also gitignored get copied, so tracked files are never duplicated.

```text
.env
.env.local
config/secrets.json
```

This applies to worktrees created with `--worktree`, subagent worktrees, and parallel sessions in the [desktop app](/en/desktop#work-in-parallel-with-sessions).

### Manage worktrees manually

For more control over worktree location and branch configuration, create worktrees with Git directly. This is useful when you need to check out a specific existing branch or place the worktree outside the repository.

```bash
# Create a worktree with a new branch
git worktree add ../project-feature-a -b feature-a

# Create a worktree with an existing branch
git worktree add ../project-bugfix bugfix-123

# Start Claude in the worktree
cd ../project-feature-a && claude

# Clean up when done
git worktree list
git worktree remove ../project-feature-a
```

Learn more in the [official Git worktree documentation](https://git-scm.com/docs/git-worktree).

> Remember to initialize your development environment in each new worktree according to your project's setup. Depending on your stack, this might include running dependency installation (`npm install`, `yarn`), setting up virtual environments, or following your project's standard setup process.

### Non-git version control

Worktree isolation works with git by default. For other version control systems like SVN, Perforce, or Mercurial, configure [WorktreeCreate and WorktreeRemove hooks](/en/hooks#worktreecreate) to provide custom worktree creation and cleanup logic. When configured, these hooks replace the default git behavior when you use `--worktree`, so [`.worktreeinclude`](#copy-gitignored-files-to-worktrees) is not processed. Copy any local configuration files inside your hook script instead.

For automated coordination of parallel sessions with shared tasks and messaging, see [agent teams](/en/agent-teams).

## Get notified when Claude needs your attention

When you kick off a long-running task and switch to another window, you can set up desktop notifications so you know when Claude finishes or needs your input. This uses the `Notification` [hook event](/en/hooks-guide#get-notified-when-claude-needs-input), which fires whenever Claude is waiting for permission, idle and ready for a new prompt, or completing authentication.

**Step 1: Add the hook to your settings**

Open `~/.claude/settings.json` and add a `Notification` hook that calls your platform's native notification command:

**macOS:**

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude Code needs your attention\" with title \"Claude Code\"'"
          }
        ]
      }
    ]
  }
}
```

**Linux:**

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Claude Code needs your attention'"
          }
        ]
      }
    ]
  }
}
```

**Windows:**

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "powershell.exe -Command \"[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); [System.Windows.Forms.MessageBox]::Show('Claude Code needs your attention', 'Claude Code')\""
          }
        ]
      }
    ]
  }
}
```

If your settings file already has a `hooks` key, merge the `Notification` entry into it rather than overwriting. You can also ask Claude to write the hook for you by describing what you want in the CLI.

**Step 2: Optionally narrow the matcher**

By default the hook fires on all notification types. To fire only for specific events, set the `matcher` field to one of these values:

| Matcher              | Fires when                                      |
| :------------------- | :---------------------------------------------- |
| `permission_prompt`  | Claude needs you to approve a tool use          |
| `idle_prompt`        | Claude is done and waiting for your next prompt |
| `auth_success`       | Authentication completes                        |
| `elicitation_dialog` | Claude is asking you a question                 |

**Step 3: Verify the hook**

Type `/hooks` and select `Notification` to confirm the hook appears. Selecting it shows the command that will run. To test it end-to-end, ask Claude to run a command that requires permission and switch away from the terminal, or ask Claude to trigger a notification directly.

For the complete event schema and notification types, see the [Notification reference](/en/hooks#notification).

## Use Claude as a unix-style utility

### Add Claude to your verification process

Suppose you want to use Claude Code as a linter or code reviewer.

**Add Claude to your build script:**

```json
// package.json
{
    ...
    "scripts": {
        ...
        "lint:claude": "claude -p 'you are a linter. please look at the changes vs. main and report any issues related to typos. report the filename and line number on one line, and a description of the issue on the second line. do not return any other text.'"
    }
}
```

Tips:

* Use Claude for automated code review in your CI/CD pipeline
* Customize the prompt to check for specific issues relevant to your project
* Consider creating multiple scripts for different types of verification

### Pipe in, pipe out

Suppose you want to pipe data into Claude, and get back data in a structured format.

**Pipe data through Claude:**

```bash
cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt
```

Tips:

* Use pipes to integrate Claude into existing shell scripts
* Combine with other Unix tools for powerful workflows
* Consider using `--output-format` for structured output

### Control output format

Suppose you need Claude's output in a specific format, especially when integrating Claude Code into scripts or other tools.

**Step 1: Use text format (default)**

```bash
cat data.txt | claude -p 'summarize this data' --output-format text > summary.txt
```

This outputs just Claude's plain text response (default behavior).

**Step 2: Use JSON format**

```bash
cat code.py | claude -p 'analyze this code for bugs' --output-format json > analysis.json
```

This outputs a JSON array of messages with metadata including cost and duration.

**Step 3: Use streaming JSON format**

```bash
cat log.txt | claude -p 'parse this log file for errors' --output-format stream-json
```

This outputs a series of JSON objects in real-time as Claude processes the request. Each message is a valid JSON object, but the entire output is not valid JSON if concatenated.

Tips:

* Use `--output-format text` for simple integrations where you just need Claude's response
* Use `--output-format json` when you need the full conversation log
* Use `--output-format stream-json` for real-time output of each conversation turn

## Run Claude on a schedule

Suppose you want Claude to handle a task automatically on a recurring basis, like reviewing open PRs every morning, auditing dependencies weekly, or checking for CI failures overnight.

Pick a scheduling option based on where you want the task to run:

| Option                                                 | Where it runs                     | Best for                                                                                                      |
| :----------------------------------------------------- | :-------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| [Cloud scheduled tasks](/en/web-scheduled-tasks)       | Anthropic-managed infrastructure  | Tasks that should run even when your computer is off. Configure at [claude.ai/code](https://claude.ai/code).  |
| [Desktop scheduled tasks](/en/desktop-scheduled-tasks) | Your machine, via the desktop app | Tasks that need direct access to local files, tools, or uncommitted changes.                                  |
| [GitHub Actions](/en/github-actions)                   | Your CI pipeline                  | Tasks tied to repo events like opened PRs, or cron schedules that should live alongside your workflow config. |
| [`/loop`](/en/scheduled-tasks)                         | The current CLI session           | Quick polling while a session is open. Tasks are cancelled when you exit.                                     |

> When writing prompts for scheduled tasks, be explicit about what success looks like and what to do with results. The task runs autonomously, so it can't ask clarifying questions. For example: "Review open PRs labeled `needs-review`, leave inline comments on any issues, and post a summary in the `#eng-reviews` Slack channel."

## Ask Claude about its capabilities

Claude has built-in access to its documentation and can answer questions about its own features and limitations.

### Example questions

```text
can Claude Code create pull requests?
```

```text
how does Claude Code handle permissions?
```

```text
what skills are available?
```

```text
how do I use MCP with Claude Code?
```

```text
how do I configure Claude Code for Amazon Bedrock?
```

```text
what are the limitations of Claude Code?
```

> Claude provides documentation-based answers to these questions. For hands-on demonstrations, run `/powerup` for interactive lessons with animated demos, or refer to the specific workflow sections above.

Tips:

* Claude always has access to the latest Claude Code documentation, regardless of the version you're using
* Ask specific questions to get detailed answers
* Claude can explain complex features like MCP integration, enterprise configurations, and advanced workflows

## Next steps

- **Best practices**: Patterns for getting the most out of Claude Code
- **How Claude Code works**: Understand the agentic loop and context management
- **Extend Claude Code**: Add skills, hooks, MCP, subagents, and plugins
- **Reference implementation**: Clone the development container reference implementation
