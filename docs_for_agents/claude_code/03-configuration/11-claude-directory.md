> ## Documentation Index
> Fetch the complete documentation index at: https://code.claude.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Explore the .claude directory

> Where Claude Code reads CLAUDE.md, settings.json, hooks, skills, commands, subagents, rules, and auto memory. Explore the .claude directory in your project and ~/.claude in your home directory.

export const ClaudeExplorer = () => {
  const A = useMemo(() => ({href, children}) => <a href={href} style={{
    color: 'var(--ce-accent)',
    textDecoration: 'none',
    borderBottom: '1px dotted var(--ce-accent)'
  }}>{children}</a>, []);
  const C = useMemo(() => ({children}) => <code style={{
    fontFamily: 'var(--ce-mono)',
    fontSize: '0.92em',
    padding: '1px 4px',
    borderRadius: '3px',
    background: 'var(--ce-surface)',
    border: '0.5px solid var(--ce-border-subtle)'
  }}>{children}</code>, []);
  const commandsNote = useMemo(() => <>Commands and skills are now the same mechanism. For new workflows, use <A href="/en/skills">skills/</A> instead: same <C>/name</C> invocation, plus you can bundle supporting files.</>, []);
  const FILE_TREE = useMemo(() => ({
    project: {
      label: 'your-project/',
      children: [{
        id: 'claude-md',
        label: 'CLAUDE.md',
        type: 'file',
        icon: 'md',
        color: '#6A9BCC',
        badge: 'committed',
        oneLiner: 'Project instructions Claude reads every session',
        when: 'Loaded into context at the start of every session',
        description: 'Project-specific instructions that shape how Claude works in this repository. Put your conventions, common commands, and architectural context here so Claude operates with the same assumptions your team does.',
        tips: ['Target under 200 lines. Longer files still load in full but may reduce adherence', <>CLAUDE.md loads into every session. If something only matters for specific tasks, move it to a <A href="/en/skills">skill</A> or a path-scoped <A href="/en/memory#organize-rules-with-claude/rules/">rule</A> so it loads only when needed</>, 'List the commands you run most, like build, test, and format, so Claude knows them without you spelling them out each time', <>Run <C>/memory</C> to open and edit CLAUDE.md from within a session</>, <>Also works at <C>.claude/CLAUDE.md</C> if you prefer to keep the project root clean</>],
        exampleIntro: 'This example is for a TypeScript and React project. It lists the build and test commands, the framework conventions Claude should follow, and project-specific rules like export style and file layout.',
        example: `# Project conventions

## Commands
- Build: \`npm run build\`
- Test: \`npm test\`
- Lint: \`npm run lint\`

## Stack
- TypeScript with strict mode
- React 19, functional components only

## Rules
- Named exports, never default exports
- Tests live next to source: \`foo.ts\` -> \`foo.test.ts\`
- All API routes return \`{ data, error }\` shape`,
        docsLink: '/en/memory'
      }, {
        id: 'mcp-json',
        label: '.mcp.json',
        type: 'file',
        icon: 'json',
        color: '#9B7BC4',
        badge: 'committed',
        oneLiner: 'Project-scoped MCP servers, shared with your team',
        when: <>Servers connect when the session begins. Tool schemas are deferred by default and load on demand via <A href="/en/mcp#scale-with-mcp-tool-search">tool search</A></>,
        description: <>Configures Model Context Protocol (MCP) servers that give Claude access to external tools: databases, APIs, browsers, and more. This file holds the project-scoped servers your whole team uses. Personal servers you want to keep to yourself go in <C>~/.claude.json</C> instead.</>,
        tips: [<>Use environment variable references for secrets: <C>{'${GITHUB_TOKEN}'}</C></>, <>Lives at the project root, not inside <C>.claude/</C></>, <>For servers only you need, run <C>claude mcp add --scope user</C>. This writes to <C>~/.claude.json</C> instead of <C>.mcp.json</C></>],
        exampleIntro: <>This example configures the GitHub MCP server so Claude can read issues and open pull requests. The <C>{'${GITHUB_TOKEN}'}</C> reference is read from your shell environment when Claude Code starts the server, so the token never lands in the file.</>,
        example: `{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "\${GITHUB_TOKEN}"
      }
    }
  }
}`,
        docsLink: '/en/mcp'
      }, {
        id: 'worktreeinclude',
        label: '.worktreeinclude',
        type: 'file',
        icon: 'md',
        color: '#8FA876',
        badge: 'committed',
        oneLiner: 'Gitignored files to copy into new worktrees',
        when: <>Read when Claude creates a git worktree via <C>--worktree</C>, the <C>EnterWorktree</C> tool, or subagent <C>isolation: worktree</C></>,
        description: <>Lists gitignored files to copy from your main repository into each new worktree. Worktrees are fresh checkouts, so untracked files like <C>.env</C> are missing by default. Patterns here use <C>.gitignore</C> syntax. Only files that match a pattern and are also gitignored get copied, so tracked files are never duplicated.</>,
        tips: [<>Lives at the project root, not inside <C>.claude/</C></>, <>Git-only: if you configure a <A href="/en/hooks#worktreecreate">WorktreeCreate hook</A> for a different VCS, this file is not read. Copy files inside your hook script instead</>, <>Also applies to parallel sessions in the <A href="/en/desktop#work-in-parallel-with-sessions">desktop app</A></>],
        exampleIntro: 'This example copies your local environment files and a secrets config into every worktree Claude creates. Comments start with # and blank lines are ignored, same as .gitignore.',
        example: `# Local environment
.env
.env.local

# API credentials
config/secrets.json`,
        docsLink: '/en/common-workflows#copy-gitignored-files-to-worktrees'
      }, {
        id: 'dot-claude',
        label: '.claude/',
        type: 'folder',
        icon: 'folder',
        color: 'var(--ce-accent)',
        oneLiner: 'Project-level configuration, rules, and extensions',
        description: 'Everything Claude Code reads that is specific to this project. If you use git, commit most files here so your team shares them; a few, like settings.local.json, are automatically gitignored. Each file badge shows which.',
        children: [{
          id: 'settings-json',
          label: 'settings.json',
          type: 'file',
          icon: 'json',
          color: 'var(--ce-text-3)',
          badge: 'committed',
          oneLiner: 'Permissions, hooks, and configuration',
          when: <>Overrides global <C>~/.claude/settings.json</C>. Local settings, CLI flags, and managed settings override this</>,
          description: 'Settings that Claude Code applies directly. Permissions control which commands and tools Claude can use; hooks run your scripts at specific points in a session. Unlike CLAUDE.md, which Claude reads as guidance, these are enforced whether Claude follows them or not.',
          contains: [<><A href="/en/permissions">permissions</A>: allow, deny, or prompt before Claude uses specific tools or commands</>, <><A href="/en/hooks">hooks</A>: run your own scripts on events like before a tool call or after a file edit</>, <><A href="/en/statusline">statusLine</A>: customize the line shown at the bottom while Claude works</>, <><A href="/en/settings#available-settings">model</A>: pick a default model for this project</>, <><A href="/en/settings#environment-variables">env</A>: environment variables set in every session</>, <><A href="/en/output-styles">outputStyle</A>: select a custom system-prompt style from output-styles/</>],
          tips: [<>Bash permission patterns support wildcards: <C>Bash(npm test *)</C> matches any command starting with <C>npm test</C></>, <>Array settings like <C>permissions.allow</C> combine across all scopes; scalar settings like <C>model</C> use the most specific value</>],
          exampleIntro: <>This example allows <C>npm test</C> and <C>npm run</C> commands without prompting, blocks <C>rm -rf</C>, and runs Prettier on files after Claude edits or writes them.</>,
          example: `{
  "permissions": {
    "allow": [
      "Bash(npm test *)",
      "Bash(npm run *)"
    ],
    "deny": [
      "Bash(rm -rf *)"
    ]
  },
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
      }]
    }]
  }
}`,
          docsLink: '/en/settings'
        }, {
          id: 'settings-local-json',
          label: 'settings.local.json',
          type: 'file',
          icon: 'json',
          color: 'var(--ce-text-3)',
          badge: 'gitignored',
          oneLiner: 'Your personal settings overrides for this project',
          when: 'Highest of the user-editable settings files; CLI flags and managed settings still take precedence',
          description: 'Personal settings that take precedence over the project defaults. Same JSON format as settings.json, but not committed. Use this when you need different permissions or defaults than the team config.',
          tips: [<>Same schema as settings.json. Array settings like <C>permissions.allow</C> combine across scopes; scalar settings like <C>model</C> use the local value</>, <>Claude Code adds this file to <C>~/.config/git/ignore</C> the first time it writes one. If you use a custom <C>core.excludesFile</C>, add the pattern there too. To share the ignore rule with your team, also add it to the project <C>.gitignore</C></>],
          exampleIntro: 'This example adds Docker permissions on top of whatever the team settings.json allows.',
          example: `{
  "permissions": {
    "allow": [
      "Bash(docker *)"
    ]
  }
}`,
          docsLink: '/en/settings'
        }, {
          id: 'rules',
          label: 'rules/',
          type: 'folder',
          icon: 'folder',
          color: '#9B7BC4',
          oneLiner: 'Topic-scoped instructions, optionally gated by file paths',
          when: <>Rules without <C>paths:</C> load at session start. Rules with <C>paths:</C> load when a matching file enters context</>,
          description: [<>Project instructions split into topic files that can load conditionally based on file paths. A rule without <C>paths:</C> frontmatter loads at session start like CLAUDE.md; a rule with <C>paths:</C> loads only when Claude reads a matching file.</>, <>Like CLAUDE.md, rules are guidance Claude reads, not configuration Claude Code enforces. For guaranteed behavior use <A href="/en/hooks">hooks</A> or <A href="/en/permissions">permissions</A>.</>],
          tips: [<>Use <C>paths:</C> frontmatter with globs to scope rules to directories or file types</>, <>Subdirectories work: <C>.claude/rules/frontend/react.md</C> is discovered automatically</>, 'When CLAUDE.md approaches 200 lines, start splitting into rules'],
          docsLink: '/en/memory#organize-rules-with-claude/rules/',
          children: [{
            id: 'rule-testing',
            label: 'testing.md',
            type: 'file',
            icon: 'md',
            color: '#9B7BC4',
            badge: 'committed',
            oneLiner: 'Test conventions scoped to test files',
            when: <>Loaded when Claude reads a file matching the <C>paths:</C> globs below</>,
            description: <>An example rule that only loads when Claude is working on test files. The <C>paths:</C> globs in the frontmatter define which files trigger it; here, anything ending in .test.ts or .test.tsx. For other files, this rule is not loaded into context.</>,
            example: `---
paths:
  - "**/*.test.ts"
  - "**/*.test.tsx"
---

# Testing Rules

- Use descriptive test names: "should [expected] when [condition]"
- Mock external dependencies, not internal modules
- Clean up side effects in afterEach`
          }, {
            id: 'rule-api',
            label: 'api-design.md',
            type: 'file',
            icon: 'md',
            color: '#9B7BC4',
            badge: 'committed',
            oneLiner: 'API conventions scoped to backend code',
            when: <>Loaded when Claude reads a file matching the <C>paths:</C> glob below</>,
            description: <>A second example showing a rule scoped to backend code. The <C>paths:</C> glob matches files under src/api/, so these conventions load only when Claude is editing API routes.</>,
            example: `---
paths:
  - "src/api/**/*.ts"
---

# API Design Rules

- All endpoints must validate input with Zod schemas
- Return shape: { data: T } | { error: string }
- Rate limit all public endpoints`
          }]
        }, {
          id: 'skills',
          label: 'skills/',
          type: 'folder',
          icon: 'folder',
          color: '#D4A843',
          oneLiner: 'Reusable prompts you or Claude invoke by name',
          when: <>Invoked with <C>/skill-name</C> or when Claude matches the task to a skill</>,
          description: <>Each skill is a folder with a SKILL.md file plus any supporting files it needs. By default, both you and Claude can invoke a skill. Use frontmatter to control that: <C>disable-model-invocation: true</C> for user-only workflows like <C>/deploy</C>, or <C>user-invocable: false</C> to hide from the <C>/</C> menu while Claude can still invoke it.</>,
          tips: [<>Skills accept arguments: <C>/deploy staging</C> passes "staging" as <C>$ARGUMENTS</C>. Use <C>$0</C>, <C>$1</C>, and so on for positional access</>, <>The <C>description</C> frontmatter determines when Claude auto-invokes the skill</>, 'Bundle reference docs alongside SKILL.md. Claude knows the skill directory path and can read supporting files when you mention them'],
          docsLink: '/en/skills',
          children: [{
            id: 'skill-review',
            label: 'security-review/',
            type: 'folder',
            icon: 'folder',
            color: '#D4A843',
            oneLiner: 'A skill bundling SKILL.md with supporting files',
            children: [{
              id: 'skill-review-md',
              label: 'SKILL.md',
              type: 'file',
              icon: 'md',
              color: '#D4A843',
              badge: 'committed',
              oneLiner: 'Entrypoint: trigger, invocability, instructions',
              when: <>User types <C>/security-review &lt;target&gt;</C>; Claude cannot auto-invoke this skill</>,
              description: [<>This skill uses <C>disable-model-invocation: true</C> so only you can trigger it; Claude never invokes it on its own.</>, <>The <C>!`...`</C> line runs a shell command and injects its output into the prompt. <C>$ARGUMENTS</C> substitutes whatever you typed after the skill name. Claude sees the skill directory path, so mentioning a bundled file like checklist.md lets Claude read it.</>],
              example: `---
description: Reviews code changes for security vulnerabilities, authentication gaps, and injection risks
disable-model-invocation: true
argument-hint: <branch-or-path>
---

## Diff to review

!\`git diff $ARGUMENTS\`

Audit the changes above for:

1. Injection vulnerabilities (SQL, XSS, command)
2. Authentication and authorization gaps
3. Hardcoded secrets or credentials

Use checklist.md in this skill directory for the full review checklist.

Report findings with severity ratings and remediation steps.`
            }, {
              id: 'skill-checklist',
              label: 'checklist.md',
              type: 'file',
              icon: 'md',
              color: '#D4A843',
              badge: 'committed',
              oneLiner: 'Supporting file bundled with the skill',
              when: 'Claude reads it on demand while running the skill',
              description: <>Skills can bundle any supporting files: reference docs, templates, scripts. The skill directory path is prepended to SKILL.md, so Claude can read bundled files by name. For scripts in bash injection commands, use the <C>{'${CLAUDE_SKILL_DIR}'}</C> placeholder.</>,
              example: `# Security Review Checklist

## Input Validation
- [ ] All user input sanitized before DB queries
- [ ] File upload MIME types validated
- [ ] Path traversal prevented on file operations

## Authentication
- [ ] JWT tokens expire after 24 hours
- [ ] API keys stored in environment variables
- [ ] Passwords hashed with bcrypt or argon2`
            }]
          }]
        }, {
          id: 'commands',
          label: 'commands/',
          type: 'folder',
          icon: 'folder',
          color: '#788C5D',
          oneLiner: <>Single-file prompts invoked with <C>/name</C></>,
          note: commandsNote,
          when: <>User types <C>/command-name</C></>,
          description: <>A file at <C>commands/deploy.md</C> creates <C>/deploy</C> the same way a skill at <C>skills/deploy/SKILL.md</C> does, and both can be auto-invoked by Claude. Skills use a directory with SKILL.md, letting you bundle reference docs, templates, or scripts alongside the prompt.</>,
          tips: [<>Use <C>$ARGUMENTS</C> in the file to accept parameters: <C>/fix-issue 123</C></>, 'If a skill and command share a name, the skill takes precedence', 'New commands should usually be skills instead; commands remain supported'],
          docsLink: '/en/skills',
          children: [{
            id: 'cmd-example',
            label: 'fix-issue.md',
            type: 'file',
            icon: 'md',
            color: '#788C5D',
            badge: 'committed',
            oneLiner: <>Invoked as <C>/fix-issue &lt;number&gt;</C></>,
            note: commandsNote,
            description: [<>An example command for fixing a GitHub issue. Type <C>/fix-issue 123</C> and the <C>!`...`</C> line runs <C>gh issue view 123</C> in your shell, injecting the output into the prompt before Claude sees it.</>, <><C>$ARGUMENTS</C> substitutes whatever you typed after the command name. For positional access, use <C>$0</C> <C>$1</C> and so on.</>],
            example: `---
argument-hint: <issue-number>
---

!\`gh issue view $ARGUMENTS\`

Investigate and fix the issue above.

1. Trace the bug to its root cause
2. Implement the fix
3. Write or update tests
4. Summarize what you changed and why`
          }]
        }, {
          id: 'output-styles',
          label: 'output-styles/',
          type: 'folder',
          icon: 'folder',
          color: '#5AA7A7',
          oneLiner: 'Project-scoped output styles, if your team shares any',
          when: 'Applied at session start when selected via the outputStyle setting',
          description: <>Output styles are usually personal, so most live in <C>~/.claude/output-styles/</C>. Put one here if your team shares a style, like a review mode everyone uses. See <A href="#ce-global-output-styles">the Global tab</A> for the full explanation and example.</>,
          docsLink: '/en/output-styles',
          children: []
        }, {
          id: 'agents',
          label: 'agents/',
          type: 'folder',
          icon: 'folder',
          color: '#C46686',
          oneLiner: 'Specialized subagents with their own context window',
          when: 'Runs in its own context window when you or Claude invoke it',
          description: 'Each markdown file defines a subagent with its own system prompt, tool access, and optionally its own model. Subagents run in a fresh context window, keeping the main conversation clean. Useful for parallel work or isolated tasks.',
          tips: ['Each agent gets a fresh context window, separate from your main session', <>Restrict tool access per agent with the <C>tools:</C> frontmatter field</>, 'Type @ and pick an agent from the autocomplete to delegate directly'],
          docsLink: '/en/sub-agents',
          children: [{
            id: 'agent-reviewer',
            label: 'code-reviewer.md',
            type: 'file',
            icon: 'md',
            color: '#C46686',
            badge: 'committed',
            oneLiner: 'Subagent for isolated code review',
            when: 'Claude spawns it for review tasks, or you @-mention it from the autocomplete',
            description: <>An example subagent restricted to read-only tools. The <C>description</C> frontmatter tells Claude when to delegate to it automatically; <C>tools:</C> limits it to Read, Grep, and Glob so it can inspect code but never edit. The body becomes the subagent's system prompt.</>,
            example: `---
name: code-reviewer
description: Reviews code for correctness, security, and maintainability
tools: Read, Grep, Glob
---

You are a senior code reviewer. Review for:

1. Correctness: logic errors, edge cases, null handling
2. Security: injection, auth bypass, data exposure
3. Maintainability: naming, complexity, duplication

Every finding must include a concrete fix.`
          }]
        }, {
          id: 'agent-memory',
          label: 'agent-memory/',
          type: 'folder',
          icon: 'folder',
          color: '#C46686',
          badge: 'committed',
          autogen: true,
          oneLiner: 'Subagent persistent memory, separate from your main session auto memory',
          when: 'First 200 lines (capped at 25KB) of MEMORY.md loaded into the subagent system prompt when it runs',
          description: <>Subagents with <C>memory: project</C> in their frontmatter get a dedicated memory directory here. This is distinct from your <A href="/en/memory#auto-memory">main session auto memory</A> at <C>~/.claude/projects/</C>: each subagent reads and writes its own MEMORY.md, not yours.</>,
          tips: [<>Only created for subagents that set the <C>memory:</C> frontmatter field</>, <>This directory holds project-scoped subagent memory, meant to be shared with your team. To keep memory out of version control use <C>memory: local</C>, which writes to <C>.claude/agent-memory-local/</C> instead. For cross-project memory use <C>memory: user</C>, which writes to <C>~/.claude/agent-memory/</C></>, <>The main session auto memory is a different feature; see <C>~/.claude/projects/</C> in the Global tab</>],
          docsLink: '/en/sub-agents#enable-persistent-memory',
          children: [{
            id: 'agent-memory-sub',
            label: '<agent-name>/',
            type: 'folder',
            icon: 'folder',
            color: '#C46686',
            autogen: true,
            children: [{
              id: 'agent-memory-md',
              label: 'MEMORY.md',
              type: 'file',
              icon: 'md',
              color: '#C46686',
              badge: 'committed',
              autogen: true,
              oneLiner: 'The subagent writes and maintains this file automatically',
              when: 'Loaded into the subagent system prompt when the subagent starts',
              description: <>Works the same as your <A href="/en/memory#auto-memory">main auto memory</A>: the subagent creates and updates this file itself. You do not write it. The subagent reads it at the start of each task and writes back what it learns.</>,
              example: `# code-reviewer memory

## Patterns seen
- Project uses custom Result<T, E> type, not exceptions
- Auth middleware expects Bearer token in Authorization header
- Tests use factory functions in test/factories/

## Recurring issues
- Missing null checks on API responses (src/api/*)
- Unhandled promise rejections in background jobs`
            }]
          }]
        }]
      }]
    },
    global: {
      label: '~/',
      children: [{
        id: 'claude-json',
        label: '.claude.json',
        type: 'file',
        icon: 'json',
        color: 'var(--ce-text-3)',
        badge: 'local',
        oneLiner: 'App state and UI preferences',
        when: <>Read at session start for your preferences and MCP servers. Claude Code writes back to it when you change settings in <C>/config</C> or approve trust prompts</>,
        description: <>Holds state that does not belong in settings.json: theme, OAuth session, per-project trust decisions, your personal MCP servers, and UI toggles. Mostly managed through <C>/config</C> rather than editing directly.</>,
        tips: [<>UI toggles like <C>showTurnDuration</C> and <C>terminalProgressBarEnabled</C> live here, not in settings.json</>, <>The <C>projects</C> key tracks per-project state like trust-dialog acceptance and last-session metrics. Permission rules you approve in-session go to <C>.claude/settings.local.json</C> instead</>, <>MCP servers here are yours only: user scope applies across all projects, local scope is per-project but not committed. Team-shared servers go in <C>.mcp.json</C> at the project root instead</>],
        example: `{
  "editorMode": "vim",
  "showTurnDuration": false,
  "mcpServers": {
    "my-tools": {
      "command": "npx",
      "args": ["-y", "@example/mcp-server"]
    }
  }
}`,
        docsLink: '/en/settings#global-config-settings'
      }, {
        id: 'global-dot-claude',
        label: '.claude/',
        type: 'folder',
        icon: 'folder',
        color: 'var(--ce-accent)',
        oneLiner: 'Your personal configuration across all projects',
        description: 'The global counterpart to your project .claude/ directory. Files here apply to every project you work in and are never committed to any repository.',
        children: [{
          id: 'global-claude-md',
          label: 'CLAUDE.md',
          type: 'file',
          icon: 'md',
          color: '#6A9BCC',
          badge: 'local',
          oneLiner: 'Personal preferences across every project',
          when: 'Loaded at the start of every session, in every project',
          description: 'Your global instruction file. Loaded alongside the project CLAUDE.md at session start, so both are in context together. When instructions conflict, project-level instructions take priority. Keep this to preferences that apply everywhere: response style, commit format, personal conventions.',
          tips: ['Keep it short since it loads into context for every project, alongside that project\'s own CLAUDE.md', 'Good for response style, commit format, and personal conventions'],
          example: `# Global preferences

- Keep explanations concise
- Use conventional commit format
- Show the terminal command to verify changes
- Prefer composition over inheritance`,
          docsLink: '/en/memory'
        }, {
          id: 'global-settings',
          label: 'settings.json',
          type: 'file',
          icon: 'json',
          color: 'var(--ce-text-3)',
          badge: 'local',
          oneLiner: 'Default settings for all projects',
          when: 'Your defaults. Project and local settings.json override any keys you also set there',
          description: [<>Same keys as project <C>settings.json</C>: permissions, hooks, model, environment variables, and the rest. Put settings here that you want in every project, like permissions you always allow, a preferred model, or a notification hook that runs regardless of which project you're in.</>, <>Settings follow a precedence order: project <C>settings.json</C> overrides any matching keys you set here. This is different from CLAUDE.md, where global and project files are both loaded into context rather than merged key by key.</>],
          example: `{
  "permissions": {
    "allow": [
      "Bash(git log *)",
      "Bash(git diff *)"
    ]
  }
}`,
          docsLink: '/en/settings'
        }, {
          id: 'keybindings',
          label: 'keybindings.json',
          type: 'file',
          icon: 'json',
          color: 'var(--ce-text-3)',
          badge: 'local',
          oneLiner: 'Custom keyboard shortcuts',
          when: 'Read at session start and hot-reloaded when you edit the file',
          description: <>Rebind keyboard shortcuts in the interactive CLI. Run <C>/keybindings</C> to create or open this file with a schema reference. Ctrl+C, Ctrl+D, and Ctrl+M are reserved and cannot be rebound.</>,
          exampleIntro: <>This example binds <C>Ctrl+E</C> to open your external editor and unbinds <C>Ctrl+U</C> by setting it to <C>null</C>. The <C>context</C> field scopes bindings to a specific part of the CLI, here the main chat input.</>,
          example: `{
  "$schema": "https://www.schemastore.org/claude-code-keybindings.json",
  "$docs": "https://code.claude.com/docs/en/keybindings",
  "bindings": [
    {
      "context": "Chat",
      "bindings": {
        "ctrl+e": "chat:externalEditor",
        "ctrl+u": null
      }
    }
  ]
}`,
          docsLink: '/en/keybindings'
        }, {
          id: 'global-projects',
          label: 'projects/',
          type: 'folder',
          icon: 'folder',
          color: '#E8A45C',
          autogen: true,
          oneLiner: "Auto memory: Claude's notes to itself, per project",
          when: 'MEMORY.md loaded at session start; topic files read on demand',
          description: 'Auto memory lets Claude accumulate knowledge across sessions without you writing anything. Claude saves notes as it works: build commands, debugging insights, architecture notes. Each project gets its own memory directory keyed by the repository path.',
          tips: [<>On by default. Toggle with <C>/memory</C> or <C>autoMemoryEnabled</C> in settings</>, 'MEMORY.md is the index loaded each session. The first 200 lines, or 25KB, whichever comes first, are read', 'Topic files like debugging.md are read on demand, not at startup', 'These are plain markdown. Edit or delete them anytime'],
          docsLink: '/en/memory#auto-memory',
          children: [{
            id: 'memory-dir',
            label: '<project>/memory/',
            type: 'folder',
            icon: 'folder',
            color: '#E8A45C',
            autogen: true,
            oneLiner: "Claude's accumulated knowledge for one project",
            children: [{
              id: 'memory-md',
              label: 'MEMORY.md',
              type: 'file',
              icon: 'md',
              color: '#E8A45C',
              badge: 'local',
              autogen: true,
              oneLiner: 'Claude writes and maintains this file automatically',
              when: 'First 200 lines (capped at 25KB) loaded at session start',
              description: 'Claude creates and updates this file as it works; you do not write it yourself. It acts as an index that Claude reads at the start of every session, pointing to topic files for detail. You can edit or delete it, but Claude will keep updating it.',
              example: `# Memory Index

## Project
- [build-and-test.md](build-and-test.md): npm run build (~45s), Vitest, dev server on 3001
- [architecture.md](architecture.md): API client singleton, refresh-token auth

## Reference
- [debugging.md](debugging.md): auth token rotation and DB connection troubleshooting`,
              docsLink: '/en/memory'
            }, {
              id: 'memory-topic',
              label: 'debugging.md',
              type: 'file',
              icon: 'md',
              color: '#E8A45C',
              badge: 'local',
              autogen: true,
              oneLiner: 'Topic notes Claude writes when MEMORY.md gets long',
              when: 'Claude reads this when a related task comes up',
              description: 'An example of a topic file Claude creates when MEMORY.md grows too long. Claude picks the filename based on what it splits out: debugging.md, architecture.md, build-commands.md, or similar. You never create these yourself. Claude reads a topic file back only when the current task relates to it.',
              example: `---
name: Debugging patterns
description: Auth token rotation and database connection troubleshooting for this project
type: reference
---

## Auth Token Issues
- Refresh token rotation: old token invalidated immediately
- If 401 after refresh: check clock skew between client and server

## Database Connection Drops
- Connection pool: max 10 in dev, 50 in prod
- Always check \`docker compose ps\` first`
            }]
          }]
        }, {
          id: 'global-rules',
          label: 'rules/',
          type: 'folder',
          icon: 'folder',
          color: '#9B7BC4',
          oneLiner: 'User-level rules that apply to every project',
          when: <>Rules without <C>paths:</C> load at session start. Rules with <C>paths:</C> load when a matching file enters context</>,
          description: 'Same as project .claude/rules/ but applies everywhere. Use this for conventions you want across all your work, like personal code style or commit message format.',
          docsLink: '/en/memory#organize-rules-with-claude/rules/',
          children: []
        }, {
          id: 'global-skills',
          label: 'skills/',
          type: 'folder',
          icon: 'folder',
          color: '#D4A843',
          oneLiner: 'Personal skills available in every project',
          when: <>Invoked with <C>/skill-name</C> in any project</>,
          description: 'Skills you built for yourself that work everywhere. Same structure as project skills: each is a folder with SKILL.md, scoped to your user account instead of a single project.',
          docsLink: '/en/skills',
          children: []
        }, {
          id: 'global-commands',
          label: 'commands/',
          type: 'folder',
          icon: 'folder',
          color: '#788C5D',
          oneLiner: 'Personal single-file commands available in every project',
          note: commandsNote,
          when: <>User types <C>/command-name</C> in any project</>,
          description: 'Same as project commands/ but scoped to your user account. Each markdown file becomes a command available everywhere.',
          docsLink: '/en/skills',
          children: []
        }, {
          id: 'global-output-styles',
          label: 'output-styles/',
          type: 'folder',
          icon: 'folder',
          color: '#5AA7A7',
          oneLiner: 'Custom system-prompt sections that adjust how Claude works',
          when: 'Applied at session start when selected via the outputStyle setting',
          description: [<>Each markdown file defines an output style: a section appended to the system prompt that, by default, also drops the built-in software-engineering task instructions. Use this to adapt Claude Code for uses beyond coding, or to add teaching or review modes.</>, <>Select a built-in or custom style with <C>/config</C> or the <C>outputStyle</C> key in settings. Styles here are available in every project; project-level styles with the same name take precedence.</>],
          tips: ['Built-in styles Explanatory and Learning are included with Claude Code; custom styles go here', <>Set <C>keep-coding-instructions: true</C> in frontmatter to keep the default task instructions alongside your additions</>, 'Changes take effect on the next session since the system prompt is fixed at startup for caching'],
          docsLink: '/en/output-styles',
          children: [{
            id: 'output-style-example',
            label: 'teaching.md',
            type: 'file',
            icon: 'md',
            color: '#5AA7A7',
            badge: 'local',
            oneLiner: 'Example style that adds explanations and leaves small changes for you',
            when: <>Active when <C>outputStyle</C> in settings is set to <C>teaching</C></>,
            description: <>This style appends instructions to the system prompt: Claude adds a "Why this approach" note after each task and leaves TODO(human) markers for changes under 10 lines instead of writing them itself. Select it by setting <C>outputStyle</C> to the filename without .md, or to the <C>name</C> field if you set one in frontmatter.</>,
            example: `---
description: Explains reasoning and asks you to implement small pieces
keep-coding-instructions: true
---

After completing each task, add a brief "Why this approach" note
explaining the key design decision.

When a change is under 10 lines, ask the user to implement it
themselves by leaving a TODO(human) marker instead of writing it.`
          }]
        }, {
          id: 'global-agents',
          label: 'agents/',
          type: 'folder',
          icon: 'folder',
          color: '#C46686',
          oneLiner: 'Personal subagents available in every project',
          when: 'Claude delegates or you @-mention in any project',
          description: 'Subagents defined here are available across all your projects. Same format as project agents.',
          docsLink: '/en/sub-agents',
          children: []
        }, {
          id: 'global-agent-memory',
          label: 'agent-memory/',
          type: 'folder',
          icon: 'folder',
          color: '#C46686',
          autogen: true,
          oneLiner: <>Persistent memory for subagents with <C>memory: user</C></>,
          when: 'Loaded into the subagent system prompt when the subagent starts',
          description: <>Subagents with <C>memory: user</C> in their frontmatter store knowledge here that persists across all projects. For project-scoped subagent memory, see <C>.claude/agent-memory/</C> instead.</>,
          docsLink: '/en/sub-agents#enable-persistent-memory',
          children: []
        }]
      }]
    }
  }), []);
  const BADGE_STYLES = useMemo(() => ({
    committed: {
      bg: 'rgba(85,138,66,0.08)',
      color: 'var(--ce-badge-committed)',
      border: 'rgba(85,138,66,0.15)',
      label: 'committed'
    },
    gitignored: {
      bg: 'rgba(217,119,87,0.06)',
      color: 'var(--ce-badge-gitignored)',
      border: 'rgba(217,119,87,0.15)',
      label: 'gitignored'
    },
    local: {
      bg: 'rgba(115,114,108,0.06)',
      color: 'var(--ce-badge-local)',
      border: 'rgba(115,114,108,0.12)',
      label: 'local only'
    },
    autogen: {
      bg: 'rgba(232,164,92,0.1)',
      color: 'var(--ce-badge-autogen)',
      border: 'rgba(232,164,92,0.2)',
      label: 'Claude writes'
    }
  }), []);
  const allNodes = useMemo(() => {
    const flatten = (nodes, acc, path, parentId) => {
      for (const node of nodes) {
        const nextPath = [...path, node.label];
        acc[node.id] = {
          ...node,
          path: nextPath,
          parentId
        };
        if (node.children) flatten(node.children, acc, nextPath, node.id);
      }
      return acc;
    };
    const project = flatten(FILE_TREE.project.children, {}, [FILE_TREE.project.label]);
    const global = flatten(FILE_TREE.global.children, {}, [FILE_TREE.global.label]);
    for (const id in project) project[id].root = 'project';
    for (const id in global) global[id].root = 'global';
    return {
      ...project,
      ...global
    };
  }, [FILE_TREE]);
  const allFolderIds = useMemo(() => Object.keys(allNodes).filter(id => allNodes[id].type === 'folder'), [allNodes]);
  const DEFAULT_EXPANDED = ['dot-claude', 'rules', 'skills', 'skill-review', 'commands', 'agents', 'agent-memory', 'agent-memory-sub', 'global-dot-claude', 'global-output-styles', 'global-projects', 'memory-dir'];
  const [mounted, setMounted] = useState(false);
  const [activeRoot, setActiveRoot] = useState('project');
  const [selectedId, setSelectedId] = useState('claude-md');
  const [expandedFolders, setExpandedFolders] = useState(() => new Set(DEFAULT_EXPANDED));
  const [forceMobile, setForceMobile] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const copyTimeoutRef = useRef(null);
  const rootRef = useRef(null);
  useEffect(() => {
    setMounted(true);
    const applyHash = scroll => {
      const hash = window.location.hash.slice(1);
      if (!hash.startsWith('ce-')) return;
      const id = hash.slice(3);
      const node = allNodes[id];
      if (!node) return;
      setActiveRoot(node.root);
      setSelectedId(id);
      setExpandedFolders(new Set(allFolderIds));
      if (scroll && rootRef.current) rootRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    };
    applyHash(false);
    const onHashChange = () => applyHash(true);
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    window.addEventListener('hashchange', onHashChange);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      window.removeEventListener('hashchange', onHashChange);
      document.removeEventListener('fullscreenchange', onFsChange);
    };
  }, []);
  useEffect(() => {
    if (!mounted || !rootRef.current) return;
    const hash = window.location.hash.slice(1);
    if (hash.startsWith('ce-') && allNodes[hash.slice(3)]) {
      rootRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  }, [mounted]);
  if (!mounted) return null;
  const selected = allNodes[selectedId];
  const tree = FILE_TREE[activeRoot];
  const isCopied = copiedId === selected.id;
  const toggleFolder = id => {
    const next = new Set(expandedFolders);
    next.has(id) ? next.delete(id) : next.add(id);
    setExpandedFolders(next);
  };
  const switchRoot = root => {
    if (root === activeRoot) return;
    setActiveRoot(root);
    const firstId = FILE_TREE[root].children[0].id;
    setSelectedId(firstId);
    try {
      history.replaceState(null, '', '#ce-' + firstId);
    } catch (e) {}
  };
  const toggleFullscreen = () => {
    if (!rootRef.current) return;
    if (document.fullscreenElement) document.exitFullscreen(); else rootRef.current.requestFullscreen().catch(() => {});
  };
  const selectNode = n => {
    setSelectedId(n.id);
    if (n.type === 'folder' && !expandedFolders.has(n.id)) toggleFolder(n.id);
    try {
      history.replaceState(null, '', '#ce-' + n.id);
    } catch (e) {}
  };
  const iconBtn = {
    width: 28,
    flexShrink: 0,
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    background: 'transparent',
    color: 'var(--ce-text-4)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  };
  const visibleFolderIds = allFolderIds.filter(id => allNodes[id].root === activeRoot);
  const allExpanded = visibleFolderIds.every(id => expandedFolders.has(id));
  const toggleAllFolders = () => {
    const next = new Set(expandedFolders);
    visibleFolderIds.forEach(id => allExpanded ? next.delete(id) : next.add(id));
    setExpandedFolders(next);
  };
  const onTreeKeyDown = e => {
    if (!['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft'].includes(e.key)) return;
    const visible = [];
    const walk = nodes => {
      for (const n of nodes) {
        visible.push(n.id);
        if (n.children && expandedFolders.has(n.id)) walk(n.children);
      }
    };
    walk(tree.children);
    const i = visible.indexOf(selectedId);
    if (i === -1) return;
    e.preventDefault();
    if (e.key === 'ArrowDown' && i < visible.length - 1) selectNode(allNodes[visible[i + 1]]); else if (e.key === 'ArrowUp' && i > 0) selectNode(allNodes[visible[i - 1]]); else if (e.key === 'ArrowRight' && selected.type === 'folder') {
      if (!expandedFolders.has(selectedId)) toggleFolder(selectedId); else if (selected.children && selected.children.length) selectNode(allNodes[selected.children[0].id]);
    } else if (e.key === 'ArrowLeft') {
      if (selected.type === 'folder' && expandedFolders.has(selectedId)) toggleFolder(selectedId); else if (selected.parentId) selectNode(allNodes[selected.parentId]);
    }
  };
  const copyExample = (id, text) => {
    const done = () => {
      setCopiedId(id);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopiedId(null), 2000);
    };
    const fallback = () => {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        if (document.execCommand('copy')) done();
      } catch (e) {}
      document.body.removeChild(ta);
    };
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }
  };
  const renderIcon = (icon, color, size) => {
    const sz = size || 14;
    if (icon === 'folder') {
      return <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
          <path d="M1.5 3.5a1 1 0 0 1 1-1h2.6l1 1.2h5.4a1 1 0 0 1 1 1v5.8a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1V3.5z" fill={color} fillOpacity="0.15" stroke={color} strokeWidth="1" />
        </svg>;
    }
    if (icon === 'json') {
      return <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
          <rect x="2" y="1.5" width="10" height="11" rx="1.5" fill={color} fillOpacity="0.15" stroke={color} strokeWidth="1" />
          <text x="7" y="9" fontSize="6" fontFamily="monospace" fill={color} textAnchor="middle" fontWeight="700">{'{}'}</text>
        </svg>;
    }
    return <svg width={sz} height={sz} viewBox="0 0 14 14" fill="none">
        <rect x="2" y="1.5" width="10" height="11" rx="1.5" fill={color} fillOpacity="0.15" stroke={color} strokeWidth="1" />
        <line x1="4.5" y1="5" x2="9.5" y2="5" stroke={color} strokeWidth="1" />
        <line x1="4.5" y1="7" x2="9.5" y2="7" stroke={color} strokeWidth="1" />
        <line x1="4.5" y1="9" x2="8" y2="9" stroke={color} strokeWidth="1" />
      </svg>;
  };
  const renderNode = (node, depth) => {
    const isFolder = node.type === 'folder';
    const isExpanded = expandedFolders.has(node.id);
    const isSelected = selectedId === node.id;
    return <div key={node.id}>
        <button role="treeitem" tabIndex={-1} onClick={() => selectNode(node)} aria-selected={isSelected} aria-expanded={isFolder ? isExpanded : undefined} style={{
      display: 'flex',
      alignItems: 'center',
      gap: '5px',
      width: '100%',
      padding: `4px 8px 4px ${8 + depth * 16}px`,
      background: isSelected ? 'var(--ce-accent-bg)' : 'transparent',
      borderTop: 'none',
      borderRight: 'none',
      borderBottom: 'none',
      borderLeft: isSelected ? '2px solid var(--ce-accent)' : '2px solid transparent',
      outline: 'none',
      cursor: 'pointer',
      textAlign: 'left',
      fontFamily: 'var(--ce-mono)',
      fontSize: '13.5px',
      color: isSelected ? 'var(--ce-accent)' : 'var(--ce-text-2)',
      fontWeight: isSelected ? 550 : 400,
      transition: 'all 0.1s'
    }}>
          {isFolder ? <span onClick={e => {
      e.stopPropagation();
      toggleFolder(node.id);
    }} style={{
      fontSize: '14px',
      color: 'var(--ce-text-4)',
      width: '20px',
      height: '20px',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
      borderRadius: '4px',
      marginLeft: '-6px',
      flexShrink: 0
    }} onMouseEnter={e => {
      e.currentTarget.style.background = 'var(--ce-arrow-hover)';
      e.currentTarget.style.color = 'var(--ce-text-2)';
    }} onMouseLeave={e => {
      e.currentTarget.style.background = 'transparent';
      e.currentTarget.style.color = 'var(--ce-text-4)';
    }}>{isExpanded ? '▾' : '▸'}</span> : <span style={{
      width: '14px',
      flexShrink: 0
    }} />}
          {renderIcon(node.icon, node.color)}
          <span style={{
      flex: 1,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap'
    }}>{node.label}</span>
          {node.badge && BADGE_STYLES[node.badge] && <span title={BADGE_STYLES[node.badge].label} style={{
      width: 6,
      height: 6,
      borderRadius: '50%',
      background: BADGE_STYLES[node.badge].color,
      flexShrink: 0,
      opacity: 0.7
    }} />}
        </button>
        {isFolder && isExpanded && node.children && <div role="group">{node.children.map(child => renderNode(child, depth + 1))}</div>}
      </div>;
  };
  return <>
    <style>{`
      .ce-root {
        --ce-mono: var(--font-mono, ui-monospace, monospace);
        --ce-accent: #D97757;
        --ce-accent-bg: rgba(217,119,87,0.06);
        --ce-accent-border: rgba(217,119,87,0.12);
        --ce-bg: #fff;
        --ce-surface: #FAFAF7;
        --ce-surface-hover: #F0EEE6;
        --ce-border: #E8E6DC;
        --ce-border-subtle: #F0EEE6;
        --ce-text: #141413;
        --ce-text-2: #5E5D59;
        --ce-text-3: #73726C;
        --ce-text-4: #9C9A92;
        --ce-text-5: #B8B6AE;
        --ce-sep: #D1CFC5;
        --ce-code-header: #F5F4ED;
        --ce-code-bg: #1A1918;
        --ce-arrow-hover: rgba(0,0,0,0.08);
        --ce-badge-committed: #3d6b2e;
        --ce-badge-gitignored: #b85c3a;
        --ce-badge-local: #5e5d59;
        --ce-badge-autogen: #b07520;
        --ce-when-text: #4a7fb5;
      }
      .dark .ce-root {
        --ce-bg: #1a1918;
        --ce-surface: #232221;
        --ce-surface-hover: #2e2d2b;
        --ce-border: #3a3936;
        --ce-border-subtle: #2e2d2b;
        --ce-text: #e8e6dc;
        --ce-text-2: #c4c2b8;
        --ce-text-3: #9c9a92;
        --ce-text-4: #73726c;
        --ce-text-5: #5e5d59;
        --ce-sep: #4a4946;
        --ce-code-header: #2e2d2b;
        --ce-code-bg: #0d0d0c;
        --ce-arrow-hover: rgba(255,255,255,0.08);
        --ce-badge-committed: #6fa85c;
        --ce-badge-gitignored: #e08a60;
        --ce-badge-local: #9c9a92;
        --ce-badge-autogen: #e8a45c;
        --ce-when-text: #8bb4e0;
      }
      .ce-mobile-fallback { display: none; border: 1px solid rgba(0,0,0,0.1); background: rgba(0,0,0,0.03); }
      .dark .ce-mobile-fallback { border-color: rgba(255,255,255,0.15); background: rgba(255,255,255,0.04); }
      @media (max-width: 700px) {
        .ce-root:not(.ce-force) { display: none !important; }
        .ce-mobile-fallback { display: block; }
      }
    `}</style>
    {!forceMobile && <div className="ce-mobile-fallback" style={{
    padding: '14px 16px',
    borderRadius: '8px',
    fontSize: '14px'
  }}>
      The interactive explorer works best on a larger screen. See the <a href="#file-reference" style={{
    color: '#D97757'
  }}>file reference table</a> below, or <button onClick={() => setForceMobile(true)} style={{
    border: 'none',
    background: 'none',
    padding: 0,
    color: '#D97757',
    textDecoration: 'underline',
    cursor: 'pointer',
    font: 'inherit'
  }}>show the explorer anyway</button>.
    </div>}
    <div ref={rootRef} className={forceMobile ? 'ce-root ce-force' : 'ce-root'} style={{
    borderRadius: isFullscreen ? 0 : '12px',
    border: '1px solid var(--ce-border)',
    background: 'var(--ce-bg)',
    display: 'flex',
    alignItems: 'stretch',
    overflow: 'hidden',
    fontFamily: 'var(--font-sans, -apple-system, sans-serif)',
    ...isFullscreen && ({
      height: '100vh'
    })
  }}>
      {}
      <div style={{
    width: 'min(240px, 35%)',
    minWidth: '180px',
    flexShrink: 0,
    borderRight: '1px solid var(--ce-border-subtle)',
    background: 'var(--ce-surface)',
    display: 'flex',
    flexDirection: 'column'
  }}>
        <div style={{
    padding: '8px 8px 4px',
    borderBottom: '1px solid var(--ce-border-subtle)',
    display: 'flex',
    gap: '4px'
  }}>
          {['project', 'global'].map(root => <button key={root} onClick={() => switchRoot(root)} style={{
    flex: 1,
    padding: '6px 0',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    fontFamily: 'var(--ce-mono)',
    fontSize: '11.5px',
    background: activeRoot === root ? 'var(--ce-accent-bg)' : 'transparent',
    color: activeRoot === root ? 'var(--ce-accent)' : 'var(--ce-text-4)',
    fontWeight: activeRoot === root ? 600 : 430
  }}>
              {root === 'project' ? 'Project' : 'Global (~/)'}
            </button>)}
          <button onClick={toggleAllFolders} title={allExpanded ? 'Collapse all' : 'Expand all'} style={{
    ...iconBtn,
    fontSize: 11
  }}>
            {allExpanded ? '⊟' : '⊞'}
          </button>
          <button onClick={toggleFullscreen} title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'} style={{
    ...iconBtn,
    fontSize: 13
  }}>
            {isFullscreen ? '⤡' : '⛶'}
          </button>
        </div>
        <div role="tree" aria-label="Configuration files" tabIndex={0} onKeyDown={onTreeKeyDown} style={{
    padding: '6px 0',
    overflowY: 'auto',
    flex: 1,
    outline: 'none'
  }}>
          {tree.children.map(node => renderNode(node, 0))}
        </div>
      </div>

      {}
      <div style={{
    flex: 1,
    minWidth: 0,
    padding: '20px 24px',
    minHeight: '400px',
    overflowY: 'auto'
  }}>
            <span aria-live="polite" style={{
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)'
  }}>{selected.label} selected</span>
            {}
            <div style={{
    fontFamily: 'var(--ce-mono)',
    fontSize: '11px',
    color: 'var(--ce-text-4)',
    marginBottom: '10px',
    cursor: 'default'
  }}>
              {selected.path.map((seg, i) => <span key={i}>
                  <span style={{
    color: i === selected.path.length - 1 ? 'var(--ce-accent)' : 'var(--ce-text-4)'
  }}>{seg.replace(/\/$/, '')}</span>
                  {i < selected.path.length - 1 && <span style={{
    color: 'var(--ce-sep)'
  }}> / </span>}
                </span>)}
            </div>

            {}
            <div style={{
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    marginBottom: '10px'
  }}>
              <span style={{
    flexShrink: 0,
    display: 'flex'
  }}>{renderIcon(selected.icon, selected.color, 24)}</span>
              <div style={{
    flex: 1,
    minWidth: 0
  }}>
                <div style={{
    fontSize: '22px',
    fontWeight: 600,
    color: 'var(--ce-text)',
    letterSpacing: '-0.3px',
    lineHeight: '26px'
  }}>{selected.label}</div>
                {selected.oneLiner && <div style={{
    fontSize: '15px',
    color: 'var(--ce-text-3)',
    marginTop: '3px'
  }}>{selected.oneLiner}</div>}
              </div>
              <div style={{
    display: 'flex',
    gap: '4px',
    flexShrink: 0
  }}>
                {[selected.autogen && 'autogen', selected.badge].filter(Boolean).map(k => {
    const s = BADGE_STYLES[k];
    if (!s) return null;
    return <span key={k} style={{
      fontFamily: 'var(--ce-mono)',
      fontSize: '10px',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.3px',
      padding: '2px 6px',
      borderRadius: '4px',
      background: s.bg,
      color: s.color,
      border: `0.5px solid ${s.border}`
    }}>{s.label}</span>;
  })}
              </div>
            </div>

            {}
            {selected.note && <div style={{
    padding: '10px 12px',
    borderRadius: '8px',
    marginBottom: '14px',
    background: 'rgba(217,119,87,0.06)',
    border: '1px solid rgba(217,119,87,0.2)',
    borderLeft: '3px solid var(--ce-accent)',
    fontSize: '15px',
    color: 'var(--ce-text-2)',
    lineHeight: 1.6
  }}>
                {selected.note}
              </div>}

            {}
            {selected.when && <div style={{
    padding: '8px 12px',
    borderRadius: '6px',
    background: 'rgba(106,155,204,0.06)',
    border: '0.5px solid rgba(106,155,204,0.12)',
    fontSize: '15px',
    color: 'var(--ce-when-text)',
    marginBottom: '16px'
  }}>
                <div style={{
    fontSize: '10px',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    opacity: 0.65,
    marginBottom: '3px'
  }}>When it loads</div>
                <div style={{
    fontWeight: 500
  }}>{selected.when}</div>
              </div>}

            {}
            {selected.description && <div style={{
    fontSize: '16px',
    color: 'var(--ce-text-2)',
    lineHeight: 1.65,
    marginBottom: '16px'
  }}>
                {Array.isArray(selected.description) ? selected.description.map((para, i) => <div key={i} style={{
    marginBottom: i < selected.description.length - 1 ? '12px' : 0
  }}>{para}</div>) : selected.description}
              </div>}

            {}
            {selected.contains && selected.contains.length > 0 && <div style={{
    marginBottom: '16px'
  }}>
                <div style={{
    fontSize: '11px',
    fontWeight: 700,
    color: 'var(--ce-text-4)',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: '8px'
  }}>Common keys</div>
                {selected.contains.map((item, i) => <div key={i} style={{
    display: 'flex',
    gap: '7px',
    fontSize: '15px',
    color: 'var(--ce-text-2)',
    lineHeight: 1.5,
    marginBottom: '5px'
  }}>
                    <span style={{
    fontSize: '7px',
    color: 'var(--ce-text-4)',
    marginTop: '6px'
  }}>●</span>
                    <span>{item}</span>
                  </div>)}
              </div>}

            {}
            {selected.tips && selected.tips.length > 0 && <div style={{
    padding: '12px 14px',
    borderRadius: '8px',
    background: 'var(--ce-surface)',
    border: '1px solid var(--ce-border-subtle)',
    marginBottom: '16px'
  }}>
                <div style={{
    fontSize: '11px',
    fontWeight: 700,
    color: 'var(--ce-accent)',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: '6px'
  }}>Tips</div>
                {selected.tips.map((tip, i) => <div key={i} style={{
    display: 'flex',
    gap: '7px',
    fontSize: '14.5px',
    color: 'var(--ce-text-2)',
    marginBottom: i < selected.tips.length - 1 ? '5px' : 0
  }}>
                    <span style={{
    fontSize: '7px',
    color: 'var(--ce-accent)',
    marginTop: '6px'
  }}>●</span>
                    <span>{tip}</span>
                  </div>)}
              </div>}

            {}
            {selected.example && <div style={{
    marginBottom: '16px'
  }}>
                {selected.exampleIntro && <div style={{
    fontSize: '15px',
    color: 'var(--ce-text-2)',
    lineHeight: 1.6,
    marginBottom: '10px'
  }}>
                    {selected.exampleIntro}
                  </div>}
                <div style={{
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '6px 10px',
    background: 'var(--ce-code-header)',
    border: '1px solid var(--ce-border)',
    borderRadius: '8px 8px 0 0'
  }}>
                  <span style={{
    fontFamily: 'var(--ce-mono)',
    fontSize: '11px',
    fontWeight: 600,
    color: 'var(--ce-text-3)'
  }}>{selected.label}</span>
                  <button onClick={() => copyExample(selected.id, selected.example)} style={{
    padding: '3px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.15s',
    background: isCopied ? 'rgba(85,138,66,0.08)' : 'var(--ce-code-header)',
    border: isCopied ? '0.5px solid rgba(85,138,66,0.2)' : '0.5px solid var(--ce-border)',
    color: isCopied ? '#558A42' : 'var(--ce-text-3)'
  }}>
                    {isCopied ? '✓ Copied' : 'Copy'}
                  </button>
                </div>
                <pre style={{
    margin: 0,
    padding: '12px 14px',
    background: 'var(--ce-code-bg)',
    color: '#E8E6DC',
    fontFamily: 'var(--ce-mono)',
    fontSize: '13px',
    lineHeight: 1.65,
    borderRadius: '0 0 8px 8px',
    overflowX: 'auto',
    whiteSpace: 'pre'
  }}>{selected.example}</pre>
              </div>}

            {}
            {selected.docsLink && <a href={selected.docsLink} style={{
    display: 'inline-flex',
    padding: '5px 12px',
    borderRadius: '6px',
    background: 'var(--ce-accent-bg)',
    border: '1px solid var(--ce-accent-border)',
    color: 'var(--ce-accent)',
    fontSize: '12px',
    fontWeight: 600,
    textDecoration: 'none'
  }}>Full docs →</a>}

            {}
            {selected.children && selected.children.length > 0 && <div style={{
    marginTop: '20px'
  }}>
                <div style={{
    fontSize: '11px',
    fontWeight: 700,
    color: 'var(--ce-text-4)',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: '8px'
  }}>Contents</div>
                <div style={{
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  }}>
                  {selected.children.map(child => <button key={child.id} onClick={() => selectNode(child)} style={{
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 8px',
    width: '100%',
    background: 'var(--ce-surface)',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'background 0.1s'
  }} onMouseEnter={e => e.currentTarget.style.background = 'var(--ce-surface-hover)'} onMouseLeave={e => e.currentTarget.style.background = 'var(--ce-surface)'}>
                      {renderIcon(child.icon, child.color, 13)}
                      <span style={{
    fontFamily: 'var(--ce-mono)',
    fontSize: '12px',
    color: 'var(--ce-text-2)'
  }}>{child.label}</span>
                      {child.oneLiner && <span style={{
    fontSize: '11px',
    color: 'var(--ce-text-4)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  }}>{child.oneLiner}</span>}
                    </button>)}
                </div>
              </div>}
      </div>
    </div>
    </>;
};

Claude Code reads instructions, settings, skills, subagents, and memory from your project directory and from `~/.claude` in your home directory. Commit project files to git to share them with your team; files in `~/.claude` are personal configuration that applies across all your projects.

Most users only edit `CLAUDE.md` and `settings.json`. The rest of the directory is optional: add skills, rules, or subagents as you need them.

This page is an interactive explorer: click files in the tree to see what each one does, when it loads, and an example. For a quick reference, see the [file reference table](#file-reference) below.

<ClaudeExplorer />

## What's not shown

The explorer covers the files you'll interact with most. A few things live elsewhere:

| File                    | Location                   | Purpose                                                                                                               |
| ----------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `managed-settings.json` | System-level, varies by OS | Enterprise-enforced settings that you can't override. See [server-managed settings](/en/server-managed-settings).     |
| `CLAUDE.local.md`       | Project root               | Your private preferences for this project, loaded alongside CLAUDE.md. Create it manually and add it to `.gitignore`. |

## File reference

This table lists every file the explorer covers. Project-scope files live in your repo under `.claude/` (or at the root for `CLAUDE.md`, `.mcp.json`, and `.worktreeinclude`). Global-scope files live in `~/.claude/` and apply across all projects.

<Note>
  Several things can override what you put in these files:

  * [Managed settings](/en/server-managed-settings) deployed by your organization take precedence over everything
  * CLI flags like `--permission-mode` or `--settings` override `settings.json` for that session
  * Some environment variables take precedence over their equivalent setting, but this varies: check the [environment variables reference](/en/env-vars) for each one

  See [settings precedence](/en/settings#settings-precedence) for the full order.
</Note>

Click a filename to open that node in the explorer above.

| File                                                | Scope              | Commit | What it does                                          | Reference                                                            |
| --------------------------------------------------- | ------------------ | ------ | ----------------------------------------------------- | -------------------------------------------------------------------- |
| [`CLAUDE.md`](#ce-claude-md)                        | Project and global | ✓      | Instructions loaded every session                     | [Memory](/en/memory)                                                 |
| [`rules/*.md`](#ce-rules)                           | Project and global | ✓      | Topic-scoped instructions, optionally path-gated      | [Rules](/en/memory#organize-rules-with-claude/rules/)                |
| [`settings.json`](#ce-settings-json)                | Project and global | ✓      | Permissions, hooks, env vars, model defaults          | [Settings](/en/settings)                                             |
| [`settings.local.json`](#ce-settings-local-json)    | Project only       |        | Your personal overrides, auto-gitignored              | [Settings scopes](/en/settings#settings-files)                       |
| [`.mcp.json`](#ce-mcp-json)                         | Project only       | ✓      | Team-shared MCP servers                               | [MCP scopes](/en/mcp#mcp-installation-scopes)                        |
| [`.worktreeinclude`](#ce-worktreeinclude)           | Project only       | ✓      | Gitignored files to copy into new worktrees           | [Worktrees](/en/common-workflows#copy-gitignored-files-to-worktrees) |
| [`skills/<name>/SKILL.md`](#ce-skills)              | Project and global | ✓      | Reusable prompts invoked with `/name` or auto-invoked | [Skills](/en/skills)                                                 |
| [`commands/*.md`](#ce-commands)                     | Project and global | ✓      | Single-file prompts; same mechanism as skills         | [Skills](/en/skills)                                                 |
| [`output-styles/*.md`](#ce-output-styles)           | Project and global | ✓      | Custom system-prompt sections                         | [Output styles](/en/output-styles)                                   |
| [`agents/*.md`](#ce-agents)                         | Project and global | ✓      | Subagent definitions with their own prompt and tools  | [Subagents](/en/sub-agents)                                          |
| [`agent-memory/<name>/`](#ce-agent-memory)          | Project and global | ✓      | Persistent memory for subagents                       | [Persistent memory](/en/sub-agents#enable-persistent-memory)         |
| [`~/.claude.json`](#ce-claude-json)                 | Global only        |        | App state, OAuth, UI toggles, personal MCP servers    | [Global config](/en/settings#global-config-settings)                 |
| [`projects/<project>/memory/`](#ce-global-projects) | Global only        |        | Auto memory: Claude's notes to itself across sessions | [Auto memory](/en/memory#auto-memory)                                |
| [`keybindings.json`](#ce-keybindings)               | Global only        |        | Custom keyboard shortcuts                             | [Keybindings](/en/keybindings)                                       |

## Check what loaded

The explorer shows what files can exist. To see what actually loaded in your current session, use these commands:

| Command        | Shows                                                                                 |
| -------------- | ------------------------------------------------------------------------------------- |
| `/context`     | Token usage by category: system prompt, memory files, skills, MCP tools, and messages |
| `/memory`      | Which CLAUDE.md and rules files loaded, plus auto-memory entries                      |
| `/agents`      | Configured subagents and their settings                                               |
| `/hooks`       | Active hook configurations                                                            |
| `/mcp`         | Connected MCP servers and their status                                                |
| `/skills`      | Available skills from project, user, and plugin sources                               |
| `/permissions` | Current allow and deny rules                                                          |
| `/doctor`      | Installation and configuration diagnostics                                            |

Run `/context` first for the overview, then the specific command for the area you want to investigate.

## Related resources

* [Manage Claude's memory](/en/memory): write and organize CLAUDE.md, rules, and auto memory
* [Configure settings](/en/settings): set permissions, hooks, environment variables, and model defaults
* [Create skills](/en/skills): build reusable prompts and workflows
* [Configure subagents](/en/sub-agents): define specialized agents with their own context
