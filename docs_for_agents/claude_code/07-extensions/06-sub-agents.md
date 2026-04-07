# Create custom subagents

> Create and use specialized AI subagents in Claude Code for task-specific workflows and improved context management.

Subagents are specialized AI assistants that handle specific types of tasks. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions. When Claude encounters a task that matches a subagent's description, it delegates to that subagent, which works independently and returns results.

Subagents help you:

* **Preserve context** by keeping exploration and implementation out of your main conversation
* **Enforce constraints** by limiting which tools a subagent can use
* **Reuse configurations** across projects with user-level subagents
* **Specialize behavior** with focused system prompts for specific domains
* **Control costs** by routing tasks to faster, cheaper models like Haiku

Claude uses each subagent's description to decide when to delegate tasks. When you create a subagent, write a clear description so Claude knows when to use it.

## Configure subagents

Subagents are defined as markdown files with YAML frontmatter:

```markdown
---
name: researcher
description: Researches questions by reading files and searching code
model: claude-3-5-haiku-20241022
maxTurns: 20
tools: Read Glob Grep Bash
---

You are a research assistant. Your job is to answer questions by:
1. Reading relevant files
2. Searching the codebase for examples
3. Summarizing your findings

Be thorough but concise. Return specific file references with line numbers.
```

### Frontmatter fields

| Field            | Description                                                   | Example                |
| :--------------- | :------------------------------------------------------------ | :--------------------- |
| `name`           | Subagent identifier (kebab-case)                              | `"researcher"`         |
| `description`    | When Claude should use this subagent                          | `"Researches questions"` |
| `model`          | Claude model to use (defaults to session model)               | `"claude-3-5-haiku"` |
| `maxTurns`       | Maximum number of tool-use turns before stopping (default: 10) | `20`                   |
| `tools`          | Space-separated list of allowed tools                         | `"Read Grep Bash"`     |
| `disallowedTools` | Space-separated list of forbidden tools                       | `"Edit Write"`         |
| `skills`         | Skills to preload into the subagent                           | `["style-guide"]`      |

## Where subagents live

Subagents are stored in `.claude/agents/` directories:

```
.claude/agents/
├── researcher/
│   └── researcher.md
├── reviewer/
│   └── reviewer.md
└── tester/
    └── tester.md
```

Each subagent is a separate markdown file.

## Invoke subagents

Claude uses subagents automatically when their description matches the task. You can also invoke them directly:

* **Automatic**: Claude decides to use `/subagent-name` based on the task description
* **Manual**: You invoke with `/subagent-name` followed by instructions

For example, to manually run the researcher subagent:

```
/researcher find all the database queries in this project
```

## Learn more

For complete details on subagent configuration, lifecycle, and best practices, see the [full subagents guide](/en/sub-agents).
