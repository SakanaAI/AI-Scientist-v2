# Plugins reference

> Complete technical reference for Claude Code plugin system, including schemas, CLI commands, and component specifications.

Looking to install plugins? See [Discover and install plugins](/en/discover-plugins). For creating plugins, see [Plugins](/en/plugins). For distributing plugins, see [Plugin marketplaces](/en/plugin-marketplaces).

This reference provides complete technical specifications for the Claude Code plugin system, including component schemas, CLI commands, and development tools.

A **plugin** is a self-contained directory of components that extends Claude Code with custom functionality. Plugin components include skills, agents, hooks, MCP servers, and LSP servers.

## Plugin components reference

### Skills

Plugins add skills to Claude Code, creating `/name` shortcuts that you or Claude can invoke.

**Location**: `skills/` or `commands/` directory in plugin root

**File format**: Skills are directories with `SKILL.md`; commands are simple markdown files

**Skill structure**:

```text
skills/
├── pdf-processor/
│   ├── SKILL.md
│   ├── reference.md (optional)
│   └── scripts/ (optional)
└── code-reviewer/
    └── SKILL.md
```

For complete details, see [Skills](/en/skills).

### Agents

Plugins can provide specialized subagents for specific tasks.

**Location**: `agents/` directory in plugin root

**File format**: Markdown files describing agent capabilities

Plugin agents support `name`, `description`, `model`, `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, and `isolation` frontmatter fields.

For complete details, see [Subagents](/en/sub-agents).

### Hooks

Plugins can provide event handlers that respond to Claude Code events automatically.

**Location**: `hooks/hooks.json` in plugin root

**Format**: JSON configuration with event matchers and actions

Plugin hooks respond to the same lifecycle events as user-defined hooks. See [Hooks](/en/hooks) for details.

### MCP servers

Plugins can bundle Model Context Protocol (MCP) servers.

**Location**: `.mcp.json` in plugin root

**Format**: Standard MCP server configuration

### LSP servers

Plugins can provide Language Server Protocol servers for code intelligence.

**Location**: `.lsp.json` in plugin root

**Format**: JSON configuration mapping language server names to their configurations

## Plugin manifest schema

The `.claude-plugin/plugin.json` file defines your plugin's metadata and configuration.

### Required fields

| Field  | Type   | Description                                                 | Example        |
| :----- | :----- | :---------------------------------------------------------- | :------------- |
| `name` | string | Unique identifier (kebab-case, no spaces).                  | `"acme-tools"` |

### Metadata fields

| Field         | Type   | Description                                  |
| :------------ | :----- | :------------------------------------------- |
| `version`     | string | Semantic version                             |
| `description` | string | Brief explanation of plugin purpose          |
| `author`      | object | Author information                           |
| `homepage`    | string | Documentation URL                            |
| `repository`  | string | Source code URL                              |
| `license`     | string | License identifier                           |
| `keywords`    | array  | Discovery tags                               |

## CLI commands reference

Claude Code provides CLI commands for plugin management:

### plugin install

Install a plugin from available marketplaces.

```bash
claude plugin install <plugin> [options]
```

**Options**:

| Option                | Description                                       | Default |
| :-------------------- | :------------------------------------------------ | :------ |
| `-s, --scope <scope>` | Installation scope: `user`, `project`, or `local` | `user`  |

### plugin uninstall

Remove an installed plugin.

```bash
claude plugin uninstall <plugin> [options]
```

**Options**:

| Option                | Description                                                   |
| :-------------------- | :------------------------------------------------------------ |
| `-s, --scope <scope>` | Uninstall from scope: `user`, `project`, or `local`           |
| `--keep-data`         | Preserve the plugin's persistent data directory               |

### plugin enable

Enable a disabled plugin.

```bash
claude plugin enable <plugin> [options]
```

### plugin disable

Disable a plugin without uninstalling it.

```bash
claude plugin disable <plugin> [options]
```

### plugin update

Update a plugin to the latest version.

```bash
claude plugin update <plugin> [options]
```

## Directory structure

A complete plugin follows this structure:

```text
enterprise-plugin/
├── .claude-plugin/
│   └── plugin.json
├── commands/
├── agents/
├── skills/
├── hooks/
├── .mcp.json
├── .lsp.json
├── bin/
└── settings.json
```

Only `.claude-plugin/plugin.json` is required. All other directories are optional.

## Debugging and development tools

### Debugging commands

Use `claude --debug` to see plugin loading details.

### Common issues

| Issue                               | Cause                           | Solution                                                                        |
| :---------------------------------- | :------------------------------ | :------------------------------------------------------------------------------- |
| Plugin not loading                  | Invalid `plugin.json`           | Run `/plugin validate` to check for errors                                      |
| Commands not appearing              | Wrong directory structure       | Ensure `commands/` at root, not in `.claude-plugin/`                            |
| Hooks not firing                    | Script not executable           | Run `chmod +x script.sh`                                                        |
| MCP server fails                    | Missing `${CLAUDE_PLUGIN_ROOT}` | Use variable for all plugin paths                                               |
| LSP `Executable not found in $PATH` | Language server not installed   | Install the binary (e.g., `npm install -g typescript-language-server`)          |

For more details, see the [full plugins reference](/en/plugins-reference).
