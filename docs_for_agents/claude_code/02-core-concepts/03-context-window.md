# Explore the context window

An interactive simulation of how Claude Code's context window fills during a session. See what loads automatically, what each file read costs, and when rules and hooks fire.

## Context Window Overview

Claude Code manages your conversation context with automatic loading and compaction. Understanding how your context window fills helps you work more efficiently.

## What loads at startup

When Claude Code starts a session, the following components load into the context in order:

### 1. System prompt (4200 tokens)
Core instructions for behavior, tool use, and response formatting. Always loaded first. You never see it.

### 2. Auto memory (MEMORY.md) (~680 tokens)
Claude's notes to itself from previous sessions: build commands it learned, patterns it noticed, mistakes to avoid. The first 200 lines or 25KB, whichever comes first, are loaded into the conversation context.

### 3. Environment info (~280 tokens)
Working directory, platform, shell, OS version, and whether this is a git repo. Git branch, status, and recent commits load as a separate block at the very end of the system prompt.

### 4. MCP tools (deferred) (~120 tokens)
MCP tool names listed so Claude knows what is available. By default, full schemas stay deferred and Claude loads specific ones on demand via tool search when a task needs them. Set `ENABLE_TOOL_SEARCH=auto` to load schemas upfront when they fit within 10% of the context window, or `ENABLE_TOOL_SEARCH=false` to load everything.

### 5. User CLAUDE.md and rules (~1200 tokens)
Your project-level instructions from CLAUDE.md, CLAUDE.local.md, and .claude/rules/ files. These are your persistent guidelines for the project.

### 6. Loaded skills (~800 tokens)
Full content of skills you've activated or that Claude is using. Skill descriptions load at startup, but full content only loads on use.

## How context fills during a session

As you work, context fills with:

- **Your prompts**: Each message you send
- **Claude's responses**: Reasoning and tool calls
- **Tool results**: Output from reads, searches, executions, etc.
- **Conversation history**: Previous turns in the session

## Context limits and compaction

Claude Code's context window has a maximum size of 200,000 tokens. As you approach this limit, Claude Code automatically:

1. **Clears old tool outputs first** - Detailed results from earlier tool calls are removed
2. **Summarizes conversation** - Early conversation turns are replaced with a concise summary
3. **Preserves instructions** - Your CLAUDE.md and persistent instructions are kept intact
4. **Keeps recent context** - Your current task and recent exchanges remain full detail

## Managing context proactively

### Run `/context` to see what's using space
The `/context` command shows a breakdown of what's consuming tokens in your current session.

### Use `/compact` for targeted compression
Instead of waiting for automatic compaction, use `/compact` to compress conversation from a specific point:

```
/compact focus on the API changes
```

This keeps earlier context in full detail while compressing later parts with your guidance.

### Configure compaction instructions in CLAUDE.md
Add a "Compact Instructions" section to guide what's preserved:

```markdown
# Compact instructions

When you are using compact, please focus on test output and code changes
```

## Context usage tips

### 1. Clear between unrelated tasks
Use `/clear` to start fresh when switching to completely different work. This prevents stale context from wasting tokens on subsequent messages.

### 2. Move detailed instructions to skills
If you have detailed workflow instructions that only apply to specific tasks, put them in [skills](/en/skills) instead of CLAUDE.md. Skills load on-demand only when invoked.

### 3. Use subagents for verbose operations
Delegate verbose operations like test runs or log analysis to [subagents](/en/sub-agents). Their context is separate, so verbose output doesn't bloat your main conversation.

### 4. Keep CLAUDE.md concise
Target under 200 lines. Longer CLAUDE.md files consume more context and reduce adherence. Use imports or `.claude/rules/` files to break up large instruction sets.

### 5. Disable unused MCP servers
Run `/mcp` to see configured servers and their context costs. Disable any you're not actively using.

## Context window composition example

A typical session context window breaks down approximately like this:

- System prompt: 4,200 tokens
- Auto memory: 680 tokens
- Environment info: 280 tokens
- MCP tools: 120 tokens
- CLAUDE.md and rules: 1,200 tokens
- Loaded skills: 800 tokens
- **Startup total: ~7,300 tokens**

Remaining ~192,700 tokens available for:
- Your conversation history
- File reads
- Command output
- Claude's responses

## See also

- [Memory](/en/memory): CLAUDE.md files and auto memory
- [How Claude Code works](/en/how-claude-code-works): The agentic loop and context management
- [Reduce token usage](/en/costs#reduce-token-usage): Strategies to lower costs
- [Settings](/en/settings): Configure context behavior
