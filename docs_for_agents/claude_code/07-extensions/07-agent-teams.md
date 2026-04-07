# Orchestrate teams of Claude Code sessions

> Coordinate multiple Claude Code instances working together as a team, with shared tasks, inter-agent messaging, and centralized management.

**Warning**: Agent teams are experimental and disabled by default. Enable them by adding `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` to your settings.json or environment. Agent teams have known limitations around session resumption, task coordination, and shutdown behavior.

Agent teams let you coordinate multiple Claude Code instances working together. One session acts as the team lead, coordinating work, assigning tasks, and synthesizing results. Teammates work independently, each in its own context window, and communicate directly with each other.

Unlike subagents, which run within a single session and can only report back to the main agent, you can also interact with individual teammates directly without going through the lead.

**Note**: Agent teams require Claude Code v2.1.32 or later. Check your version with `claude --version`.

## When to use agent teams

Agent teams are most effective for tasks where parallel exploration adds real value. The strongest use cases are:

* **Research and review**: multiple teammates can investigate different aspects of a problem simultaneously, then share and challenge each other's findings
* **New modules or features**: teammates can each own a separate piece without stepping on each other
* **Debugging with competing hypotheses**: teammates test different theories in parallel and converge on the answer faster
* **Cross-layer coordination**: changes that span frontend, backend, and tests, each owned by a different teammate

Agent teams add coordination overhead and use significantly more tokens than a single session. They work best when teammates can operate independently.

## Enable agent teams

Agent teams are disabled by default. Enable them by setting the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` environment variable to `1`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

## Start your first agent team

Tell Claude to create an agent team and describe the task and the team structure you want in natural language:

```text
I'm designing a CLI tool that helps developers track TODO comments across their codebase. Create an agent team to explore this from different angles: one teammate on UX, one on technical architecture, one playing devil's advocate.
```

From there, Claude creates a team with a shared task list, spawns teammates for each perspective, has them explore the problem, synthesizes findings, and attempts to clean up the team when finished.

The lead's terminal lists all teammates and what they're working on. Use Shift+Down to cycle through teammates and message them directly.

## Control your agent team

Tell the lead what you want in natural language. It handles team coordination, task assignment, and delegation based on your instructions.

### Specify teammates and models

Claude decides the number of teammates to spawn based on your task, or you can specify exactly what you want:

```text
Create a team with 4 teammates to refactor these modules in parallel.
Use Sonnet for each teammate.
```

### Assign and claim tasks

The shared task list coordinates work across the team. The lead creates tasks and teammates work through them. Tasks have three states: pending, in progress, and completed.

* **Lead assigns**: tell the lead which task to give to which teammate
* **Self-claim**: after finishing a task, a teammate picks up the next unassigned, unblocked task on its own

### Talk to teammates directly

Each teammate is a full, independent Claude Code session. You can message any teammate directly to give additional instructions, ask follow-up questions, or redirect their approach.

* **In-process mode**: use Shift+Down to cycle through teammates, then type to send them a message.
* **Split-pane mode**: click into a teammate's pane to interact with their session directly.

### Shut down teammates

To gracefully end a teammate's session:

```text
Ask the researcher teammate to shut down
```

### Clean up the team

When you're done:

```text
Clean up the team
```

**Warning**: Always use the lead to clean up. Teammates should not run cleanup because their team context may not resolve correctly.

## Limitations

Agent teams are experimental. Current limitations include:

* **No session resumption with in-process teammates**: `/resume` and `/rewind` do not restore in-process teammates
* **Task status can lag**: teammates sometimes fail to mark tasks as completed, blocking dependent tasks
* **Shutdown can be slow**: teammates finish their current request before shutting down
* **One team per session**: a lead can only manage one team at a time
* **No nested teams**: teammates cannot spawn their own teams
* **Lead is fixed**: the session that creates the team is the lead for its lifetime
* **Split panes require tmux or iTerm2**: the default in-process mode works in any terminal

For more details, see the [full agent teams guide](/en/agent-teams).
