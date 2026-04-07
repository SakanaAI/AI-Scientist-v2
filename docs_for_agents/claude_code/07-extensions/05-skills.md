# Extend Claude with skills

> Create, manage, and share skills to extend Claude's capabilities in Claude Code. Includes custom commands and bundled skills.

Skills extend what Claude can do. Create a `SKILL.md` file with instructions, and Claude adds it to its toolkit. Claude uses skills when relevant, or you can invoke one directly with `/skill-name`.

Note: For built-in commands like `/help` and `/compact`, see the [built-in commands reference](/en/commands).

**Custom commands have been merged into skills.** A file at `.claude/commands/deploy.md` and a skill at `.claude/skills/deploy/SKILL.md` both create `/deploy` and work the same way. Your existing `.claude/commands/` files keep working. Skills add optional features: a directory for supporting files, frontmatter to control whether you or Claude invokes them, and the ability for Claude to load them automatically when relevant.

Claude Code skills follow the [Agent Skills](https://agentskills.io) open standard, which works across multiple AI tools. Claude Code extends the standard with additional features like invocation control, subagent execution, and dynamic context injection.

## Bundled skills

Bundled skills ship with Claude Code and are available in every session. Unlike built-in commands, which execute fixed logic directly, bundled skills are prompt-based: they give Claude a detailed playbook and let it orchestrate the work using its tools. This means bundled skills can spawn parallel agents, read files, and adapt to your codebase.

You invoke bundled skills the same way as any other skill: type `/` followed by the skill name.

| Skill                       | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/batch <instruction>`      | Orchestrate large-scale changes across a codebase in parallel. Researches the codebase, decomposes the work into 5 to 30 independent units, and presents a plan. Once approved, spawns one background agent per unit in an isolated git worktree. Each agent implements its unit, runs tests, and opens a pull request. Requires a git repository.                                                                                                                                          |
| `/claude-api`               | Load Claude API reference material for your project's language and Agent SDK reference. Covers tool use, streaming, batches, structured outputs, and common pitfalls. Also activates automatically when your code imports `anthropic` or `@anthropic-ai/sdk`.                                                                                                                                                                                           |
| `/debug [description]`      | Enable debug logging for the current session and troubleshoot issues by reading the session debug log. Debug logging is off by default unless you started with `claude --debug`, so running `/debug` mid-session starts capturing logs from that point forward.                                                                                                                                                                                |
| `/loop [interval] <prompt>` | Run a prompt repeatedly on an interval while the session stays open. Useful for polling a deployment, babysitting a PR, or periodically re-running another skill.                                                                                                                                                                                                                                                                                                           |
| `/simplify [focus]`         | Review your recently changed files for code reuse, quality, and efficiency issues, then fix them. Spawns three review agents in parallel, aggregates their findings, and applies fixes.                                                                                                                                                                                                                                                                                                                            |

## Getting started

### Create your first skill

This example creates a skill that teaches Claude to explain code using visual diagrams and analogies.

**Step 1: Create the skill directory**

Create a directory for the skill in your personal skills folder. Personal skills are available across all your projects.

```bash
mkdir -p ~/.claude/skills/explain-code
```

**Step 2: Write SKILL.md**

Every skill needs a `SKILL.md` file with YAML frontmatter and markdown content. The `name` field becomes the `/slash-command`, and the `description` helps Claude decide when to load it automatically.

Create `~/.claude/skills/explain-code/SKILL.md`:

```yaml
---
name: explain-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

**Step 3: Test the skill**

Test it two ways. Let Claude invoke it automatically by asking something that matches the description:

```
How does this code work?
```

Or invoke it directly with the skill name:

```
/explain-code src/auth/login.ts
```

Either way, Claude should include an analogy and ASCII diagram in its explanation.

## Where skills live

Where you store a skill determines who can use it:

| Location   | Path                                                | Applies to                     |
| :--------- | :-------------------------------------------------- | :----------------------------- |
| Enterprise | See [managed settings](/en/settings#settings-files) | All users in your organization |
| Personal   | `~/.claude/skills/<skill-name>/SKILL.md`            | All your projects              |
| Project    | `.claude/skills/<skill-name>/SKILL.md`              | This project only              |
| Plugin     | `<plugin>/skills/<skill-name>/SKILL.md`             | Where plugin is enabled        |

When skills share the same name across levels, higher-priority locations win: enterprise > personal > project. Plugin skills use a `plugin-name:skill-name` namespace, so they cannot conflict with other levels.

## Configure skills

Skills are configured through YAML frontmatter at the top of `SKILL.md` and the markdown content that follows.

### Frontmatter reference

| Field                      | Required    | Description                                                                                                                                                                                                                                                                               |
| :------------------------- | :---------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                     | No          | Display name for the skill. If omitted, uses the directory name. Lowercase letters, numbers, and hyphens only (max 64 characters).                                                                                                                                                        |
| `description`              | Recommended | What the skill does and when to use it. Claude uses this to decide when to apply the skill. If omitted, uses the first paragraph of markdown content.                                                                                                                                     |
| `disable-model-invocation` | No          | Set to `true` to prevent Claude from automatically loading this skill. Use for workflows you want to trigger manually with `/name`. Default: `false`.                                                                                                                                     |
| `user-invocable`           | No          | Set to `false` to hide from the `/` menu. Use for background knowledge users shouldn't invoke directly. Default: `true`.                                                                                                                                                                  |
| `allowed-tools`            | No          | Tools Claude can use without asking permission when this skill is active. Accepts a space-separated string or a YAML list.                                                                                                                                                                |
| `context`                  | No          | Set to `fork` to run in a forked subagent context.                                                                                                                                                                                                                                        |

## Pass arguments to skills

Both you and Claude can pass arguments when invoking a skill. Arguments are available via the `$ARGUMENTS` placeholder.

This skill fixes a GitHub issue by number:

```yaml
---
name: fix-issue
description: Fix a GitHub issue
disable-model-invocation: true
---

Fix GitHub issue $ARGUMENTS following our coding standards.

1. Read the issue description
2. Understand the requirements
3. Implement the fix
4. Write tests
5. Create a commit
```

When you run `/fix-issue 123`, Claude receives "Fix GitHub issue 123 following our coding standards..."

To access individual arguments by position, use `$ARGUMENTS[N]` or the shorter `$N`:

```yaml
---
name: migrate-component
description: Migrate a component from one framework to another
---

Migrate the $ARGUMENTS[0] component from $ARGUMENTS[1] to $ARGUMENTS[2].
Preserve all existing behavior and tests.
```

Running `/migrate-component SearchBar React Vue` replaces placeholders with actual values.

## Learn more

For complete details on skill development including advanced patterns, supporting files, and best practices, see the [full Skills guide](/en/skills).
