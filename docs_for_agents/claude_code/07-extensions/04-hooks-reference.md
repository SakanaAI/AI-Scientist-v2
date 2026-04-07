# Hooks Reference - Complete Documentation

## Documentation Index
Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt
Use this file to discover all available pages before exploring further.

# Hooks reference

Reference for Claude Code hook events, configuration schema, JSON input/output formats, exit codes, async hooks, HTTP hooks, prompt hooks, and MCP tool hooks.

For a quickstart guide with examples, see [Automate workflows with hooks](/en/hooks-guide).

Hooks are user-defined shell commands, HTTP endpoints, or LLM prompts that execute automatically at specific points in Claude Code's lifecycle. Use this reference to look up event schemas, configuration options, JSON input/output formats, and advanced features like async hooks, HTTP hooks, and MCP tool hooks. If you're setting up hooks for the first time, start with the [guide](/en/hooks-guide) instead.

## Hook lifecycle

Hooks fire at specific points during a Claude Code session. When an event fires and a matcher matches, Claude Code passes JSON context about the event to your hook handler. For command hooks, input arrives on stdin. For HTTP hooks, it arrives as the POST request body. Your handler can then inspect the input, take action, and optionally return a decision.

The table below summarizes when each event fires:

| Event                | When it fires                                                                                                                                          |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionStart`       | When a session begins or resumes                                                                                                                       |
| `UserPromptSubmit`   | When you submit a prompt, before Claude processes it                                                                                                   |
| `PreToolUse`         | Before a tool call executes. Can block it                                                                                                              |
| `PermissionRequest`  | When a permission dialog appears                                                                                                                       |
| `PermissionDenied`   | When a tool call is denied by the auto mode classifier. Return `{retry: true}` to tell the model it may retry the denied tool call                     |
| `PostToolUse`        | After a tool call succeeds                                                                                                                             |
| `PostToolUseFailure` | After a tool call fails                                                                                                                                |
| `Notification`       | When Claude Code sends a notification                                                                                                                  |
| `SubagentStart`      | When a subagent is spawned                                                                                                                             |
| `SubagentStop`       | When a subagent finishes                                                                                                                               |
| `TaskCreated`        | When a task is being created via `TaskCreate`                                                                                                          |
| `TaskCompleted`      | When a task is being marked as completed                                                                                                               |
| `Stop`               | When Claude finishes responding                                                                                                                        |
| `StopFailure`        | When the turn ends due to an API error. Output and exit code are ignored                                                                               |
| `TeammateIdle`       | When an [agent team](/en/agent-teams) teammate is about to go idle                                                                                     |
| `InstructionsLoaded` | When a CLAUDE.md or `.claude/rules/*.md` file is loaded into context. Fires at session start and when files are lazily loaded during a session         |
| `ConfigChange`       | When a configuration file changes during a session                                                                                                     |
| `CwdChanged`         | When the working directory changes, for example when Claude executes a `cd` command. Useful for reactive environment management with tools like direnv |
| `FileChanged`        | When a watched file changes on disk. The `matcher` field specifies which filenames to watch                                                            |
| `WorktreeCreate`     | When a worktree is being created via `--worktree` or `isolation: "worktree"`. Replaces default git behavior                                            |
| `WorktreeRemove`     | When a worktree is being removed, either at session exit or when a subagent finishes                                                                   |
| `PreCompact`         | Before context compaction                                                                                                                              |
| `PostCompact`        | After context compaction completes                                                                                                                     |
| `Elicitation`        | When an MCP server requests user input during a tool call                                                                                              |
| `ElicitationResult`  | After a user responds to an MCP elicitation, before the response is sent back to the server                                                            |
| `SessionEnd`         | When a session terminates                                                                                                                              |

## Configuration

Hooks are configured in settings files as JSON. The basic structure is:

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "pattern",
        "hooks": [
          { "type": "command", "command": "..." }
        ]
      }
    ]
  }
}
```

## Hook types

### Command hooks

Execute shell commands:

```json
{
  "type": "command",
  "command": "my-script.sh"
}
```

### HTTP hooks

POST to an HTTP endpoint:

```json
{
  "type": "http",
  "url": "http://localhost:8080/hooks/event"
}
```

### Prompt hooks

Use an LLM to evaluate conditions:

```json
{
  "type": "prompt",
  "prompt": "Evaluate this condition"
}
```

### Agent hooks

Run a subagent for complex verification:

```json
{
  "type": "agent",
  "prompt": "Verify this condition"
}
```

## Input/Output Formats

Hook handlers receive JSON on stdin describing the event. The format depends on the event type. Common fields include:

- `session_id`: unique session identifier
- `cwd`: current working directory
- `hook_event_name`: name of the event that fired
- Event-specific fields

## Exit codes for command hooks

- **0**: action proceeds
- **2**: action is blocked (write reason to stderr)
- **Other**: action proceeds (stderr logged but not shown to Claude)

## See also

For complete technical specifications and detailed schemas for all hooks, see the [full Hooks reference documentation](https://code.claude.com/docs/en/hooks).
