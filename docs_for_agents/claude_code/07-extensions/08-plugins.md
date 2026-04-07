# Create plugins

> Create custom plugins to extend Claude Code with skills, agents, hooks, and MCP servers.

Plugins let you extend Claude Code with custom functionality that can be shared across projects and teams. This guide covers creating your own plugins with skills, agents, hooks, and MCP servers.

Looking to install existing plugins? See [Discover and install plugins](/en/discover-plugins). For complete technical specifications, see [Plugins reference](/en/plugins-reference).

## When to use plugins vs standalone configuration

Claude Code supports two ways to add custom skills, agents, and hooks:

| Approach                                                    | Skill names          | Best for                                                                                        |
| :---------------------------------------------------------- | :------------------- | :---------------------------------------------------------------------------------------------- |
| **Standalone** (`.claude/` directory)                       | `/hello`             | Personal workflows, project-specific customizations, quick experiments                          |
| **Plugins** (directories with `.claude-plugin/plugin.json`) | `/plugin-name:hello` | Sharing with teammates, distributing to community, versioned releases, reusable across projects |

**Use standalone configuration when**:

* You're customizing Claude Code for a single project
* The configuration is personal and doesn't need to be shared
* You're experimenting with skills or hooks before packaging them
* You want short skill names like `/hello` or `/deploy`

**Use plugins when**:

* You want to share functionality with your team or community
* You need the same skills/agents across multiple projects
* You want version control and easy updates for your extensions
* You're distributing through a marketplace
* You're okay with namespaced skills like `/my-plugin:hello`

**Tip**: Start with standalone configuration in `.claude/` for quick iteration, then convert to a plugin when you're ready to share.

## Quickstart

This quickstart walks you through creating a plugin with a custom skill. You'll create a manifest, add a skill, and test it locally using the `--plugin-dir` flag.

### Prerequisites

* Claude Code installed and authenticated

**Note**: If you don't see the `/plugin` command, update Claude Code to the latest version.

### Create your first plugin

**Step 1: Create the plugin directory**

```bash
mkdir my-first-plugin
```

**Step 2: Create the plugin manifest**

The manifest file at `.claude-plugin/plugin.json` defines your plugin's identity: its name, description, and version.

Create `.claude-plugin` directory inside your plugin folder:

```bash
mkdir my-first-plugin/.claude-plugin
```

Then create `my-first-plugin/.claude-plugin/plugin.json`:

```json
{
  "name": "my-first-plugin",
  "description": "A greeting plugin to learn the basics",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

| Field         | Purpose                                                                                                |
| :------------ | :----------------------------------------------------------------------------------------------------- |
| `name`        | Unique identifier and skill namespace. Skills are prefixed with this (e.g., `/my-first-plugin:hello`). |
| `description` | Shown in the plugin manager when browsing or installing plugins.                                       |
| `version`     | Track releases using semantic versioning.                                                              |
| `author`      | Optional. Helpful for attribution.                                                                     |

**Step 3: Add a skill**

Skills live in the `skills/` directory. Each skill is a folder containing a `SKILL.md` file.

Create a skill directory:

```bash
mkdir -p my-first-plugin/skills/hello
```

Then create `my-first-plugin/skills/hello/SKILL.md`:

```markdown
---
description: Greet the user with a friendly message
disable-model-invocation: true
---

Greet the user warmly and ask how you can help them today.
```

**Step 4: Test your plugin**

Run Claude Code with the `--plugin-dir` flag:

```bash
claude --plugin-dir ./my-first-plugin
```

Once Claude Code starts, try your new skill:

```shell
/my-first-plugin:hello
```

You'll see Claude respond with a greeting.

**Step 5: Add skill arguments**

Make your skill dynamic by accepting user input via `$ARGUMENTS`:

```markdown
---
description: Greet the user with a personalized message
---

# Hello Skill

Greet the user named "$ARGUMENTS" warmly and ask how you can help them today. Make the greeting personal and encouraging.
```

Run `/reload-plugins` to pick up changes, then try:

```shell
/my-first-plugin:hello Alex
```

## Plugin structure overview

Plugins can include much more: custom agents, hooks, MCP servers, and LSP servers.

| Directory         | Purpose                                                                        |
| :---------------- | :----------------------------------------------------------------------------- |
| `.claude-plugin/` | Contains `plugin.json` manifest (optional)                                    |
| `commands/`       | Skills as Markdown files                                                       |
| `agents/`         | Custom agent definitions                                                       |
| `skills/`         | Agent Skills with `SKILL.md` files                                             |
| `hooks/`          | Event handlers in `hooks.json`                                                 |
| `.mcp.json`       | MCP server configurations                                                      |
| `.lsp.json`       | LSP server configurations for code intelligence                                |
| `bin/`            | Executables added to the Bash tool's `PATH` while the plugin is enabled        |
| `settings.json`   | Default settings applied when the plugin is enabled                            |

## Develop more complex plugins

Once you're comfortable with basic plugins, add:

* **Skills**: extend Claude's capabilities with model-invoked workflows
* **Agents**: custom subagents for specialized tasks
* **Hooks**: automate actions on specific events
* **MCP servers**: connect external services
* **LSP servers**: provide code intelligence for languages

For complete details, see the [full plugins guide](/en/plugins).

## Share your plugins

When your plugin is ready:

1. **Add documentation**: include a `README.md` with usage instructions
2. **Version your plugin**: use semantic versioning in `plugin.json`
3. **Create or use a marketplace**: distribute via [plugin marketplaces](/en/plugin-marketplaces)
4. **Test with others**: have team members test before wider distribution

## Convert existing configurations to plugins

If you already have skills or hooks in `.claude/`, you can migrate them to a plugin for easier sharing:

1. Create the plugin structure with `.claude-plugin/plugin.json`
2. Copy existing configurations to the plugin directory
3. Migrate hooks from settings files to `hooks/hooks.json`
4. Test using `--plugin-dir`
5. Remove originals from `.claude/` to avoid duplicates

For more details, see the [full plugins guide](/en/plugins#convert-existing-configurations-to-plugins).

## Learn more

* [Create and distribute a marketplace](/en/plugin-marketplaces): package and share your plugins
* [Plugins reference](/en/plugins-reference): complete technical specifications
* [Skills](/en/skills): skill development details
* [Subagents](/en/sub-agents): agent configuration and capabilities
* [Hooks](/en/hooks): event handling and automation
* [MCP](/en/mcp): external tool integration
