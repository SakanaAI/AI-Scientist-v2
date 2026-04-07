# Tools reference

> Complete reference for the tools Claude Code can use, including permission requirements.

Claude Code has access to a set of built-in tools that help it understand and modify your codebase. The tool names are the exact strings you use in [permission rules](/en/permissions#tool-specific-permission-rules), [subagent tool lists](/en/sub-agents), and [hook matchers](/en/hooks). To disable a tool entirely, add its name to the `deny` array in your [permission settings](/en/permissions#tool-specific-permission-rules).

To add custom tools, connect an [MCP server](/en/mcp). To extend Claude with reusable prompt-based workflows, write a [skill](/en/skills), which runs through the existing `Skill` tool rather than adding a new tool entry.

| Tool                   | Description                                                                                                                                                                                                                                  | Permission Required |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ |
| `Agent`                | Spawns a [subagent](/en/sub-agents) with its own context window to handle a task                                                                                                                                                             | No                  |
| `AskUserQuestion`      | Asks multiple-choice questions to gather requirements or clarify ambiguity                                                                                                                                                                   | No                  |
| `Bash`                 | Executes shell commands in your environment. See [Bash tool behavior](#bash-tool-behavior)                                                                                                                                                   | Yes                 |
| `CronCreate`           | Schedules a recurring or one-shot prompt within the current session (gone when Claude exits). See [scheduled tasks](/en/scheduled-tasks)                                                                                                     | No                  |
| `CronDelete`           | Cancels a scheduled task by ID                                                                                                                                                                                                               | No                  |
| `CronList`             | Lists all scheduled tasks in the session                                                                                                                                                                                                     | No                  |
| `Edit`                 | Makes targeted edits to specific files                                                                                                                                                                                                       | Yes                 |
| `EnterPlanMode`        | Switches to plan mode to design an approach before coding                                                                                                                                                                                    | No                  |
| `EnterWorktree`        | Creates an isolated [git worktree](/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees) and switches into it                                                                                                           | No                  |
| `ExitPlanMode`         | Presents a plan for approval and exits plan mode                                                                                                                                                                                             | Yes                 |
| `ExitWorktree`         | Exits a worktree session and returns to the original directory                                                                                                                                                                               | No                  |
| `Glob`                 | Finds files based on pattern matching                                                                                                                                                                                                        | No                  |
| `Grep`                 | Searches for patterns in file contents                                                                                                                                                                                                       | No                  |
| `ListMcpResourcesTool` | Lists resources exposed by connected [MCP servers](/en/mcp)                                                                                                                                                                                  | No                  |
| `LSP`                  | Code intelligence via language servers: jump to definitions, find references, report type errors and warnings. See [LSP tool behavior](#lsp-tool-behavior)                                                                                   | No                  |
| `NotebookEdit`         | Modifies Jupyter notebook cells                                                                                                                                                                                                              | Yes                 |
| `PowerShell`           | Executes PowerShell commands on Windows. Opt-in preview. See [PowerShell tool](#powershell-tool)                                                                                                                                             | Yes                 |
| `Read`                 | Reads the contents of files                                                                                                                                                                                                                  | No                  |
| `ReadMcpResourceTool`  | Reads a specific MCP resource by URI                                                                                                                                                                                                         | No                  |
| `SendMessage`          | Sends a message to an [agent team](/en/agent-teams) teammate, or [resumes a subagent](/en/sub-agents#resume-subagents) by its agent ID. Stopped subagents auto-resume in the background. Only available when `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set | No                  |
| `Skill`                | Executes a [skill](/en/skills#control-who-invokes-a-skill) within the main conversation                                                                                                                                                      | Yes                 |
| `TaskCreate`           | Creates a new task in the task list                                                                                                                                                                                                          | No                  |
| `TaskGet`              | Retrieves full details for a specific task                                                                                                                                                                                                   | No                  |
| `TaskList`             | Lists all tasks with their current status                                                                                                                                                                                                    | No                  |
| `TaskOutput`           | (Deprecated) Retrieves output from a background task. Prefer `Read` on the task's output file path                                                                                                                                           | No                  |
| `TaskStop`             | Kills a running background task by ID                                                                                                                                                                                                        | No                  |
| `TaskUpdate`           | Updates task status, dependencies, details, or deletes tasks                                                                                                                                                                                 | No                  |
| `TeamCreate`           | Creates an [agent team](/en/agent-teams) with multiple teammates. Only available when `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set                                                                                                        | No                  |
| `TeamDelete`           | Disbands an agent team and cleans up teammate processes. Only available when `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` is set                                                                                                                 | No                  |
| `TodoWrite`            | Manages the session task checklist. Available in non-interactive mode and the [Agent SDK](/en/headless); interactive sessions use TaskCreate, TaskGet, TaskList, and TaskUpdate instead                                                      | No                  |
| `ToolSearch`           | Searches for and loads deferred tools when [tool search](/en/mcp#scale-with-mcp-tool-search) is enabled                                                                                                                                      | No                  |
| `WebFetch`             | Fetches content from a specified URL                                                                                                                                                                                                         | Yes                 |
| `WebSearch`            | Performs web searches                                                                                                                                                                                                                        | Yes                 |
| `Write`                | Creates or overwrites files                                                                                                                                                                                                                  | Yes                 |

Permission rules can be configured using `/permissions` or in [permission settings](/en/settings#available-settings). Also see [Tool-specific permission rules](/en/permissions#tool-specific-permission-rules).

## Bash tool behavior

The Bash tool runs each command in a separate process with the following persistence behavior:

* Working directory persists across commands. Set `CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR=1` to reset to the project directory after each command.
* Environment variables do not persist. An `export` in one command will not be available in the next.

Activate your virtualenv or conda environment before launching Claude Code. To make environment variables persist across Bash commands, set [`CLAUDE_ENV_FILE`](/en/env-vars) to a shell script before launching Claude Code, or use a [SessionStart hook](/en/hooks#persist-environment-variables) to populate it dynamically.

## LSP tool behavior

The LSP tool gives Claude code intelligence from a running language server. After each file edit, it automatically reports type errors and warnings so Claude can fix issues without a separate build step. Claude can also call it directly to navigate code:

* Jump to a symbol's definition
* Find all references to a symbol
* Get type information at a position
* List symbols in a file or workspace
* Find implementations of an interface
* Trace call hierarchies

The tool is inactive until you install a [code intelligence plugin](/en/discover-plugins#code-intelligence) for your language. The plugin bundles the language server configuration, and you install the server binary separately.

## PowerShell tool

On Windows, Claude Code can run PowerShell commands natively instead of routing through Git Bash. This is an opt-in preview.

### Enable the PowerShell tool

Set `CLAUDE_CODE_USE_POWERSHELL_TOOL=1` in your environment or in `settings.json`:

```json  theme={null}
{
  "env": {
    "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1"
  }
}
```

Claude Code auto-detects `pwsh.exe` (PowerShell 7+) with a fallback to `powershell.exe` (PowerShell 5.1). The Bash tool remains registered alongside the PowerShell tool, so you may need to ask Claude to use PowerShell.

### Shell selection in settings, hooks, and skills

Three additional settings control where PowerShell is used:

* `"defaultShell": "powershell"` in [`settings.json`](/en/settings#available-settings): routes interactive `!` commands through PowerShell. Requires the PowerShell tool to be enabled.
* `"shell": "powershell"` on individual [command hooks](/en/hooks#command-hook-fields): runs that hook in PowerShell. Hooks spawn PowerShell directly, so this works regardless of `CLAUDE_CODE_USE_POWERSHELL_TOOL`.
* `shell: powershell` in [skill frontmatter](/en/skills#frontmatter-reference): runs `` !`command` `` blocks in PowerShell. Requires the PowerShell tool to be enabled.

### Preview limitations

The PowerShell tool has the following known limitations during the preview:

* Auto mode does not work with the PowerShell tool yet
* PowerShell profiles are not loaded
* Sandboxing is not supported
* Only supported on native Windows, not WSL
* Git Bash is still required to start Claude Code

## Check which tools are available

Your exact tool set depends on your provider, platform, and settings. To check what's loaded in a running session, ask Claude directly:

```text  theme={null}
What tools do you have access to?
```

Claude gives a conversational summary. For exact MCP tool names, run `/mcp`.

## See also

* [MCP servers](/en/mcp): add custom tools by connecting external servers
* [Permissions](/en/permissions): permission system, rule syntax, and tool-specific patterns
* [Subagents](/en/sub-agents): configure tool access for subagents
* [Hooks](/en/hooks-guide): run custom commands before or after tool execution
