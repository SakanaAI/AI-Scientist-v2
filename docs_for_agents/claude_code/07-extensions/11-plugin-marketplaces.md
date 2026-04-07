# Create and distribute a plugin marketplace

> Build and host plugin marketplaces to distribute Claude Code extensions across teams and communities.

A **plugin marketplace** is a catalog that lets you distribute plugins to others. Marketplaces provide centralized discovery, version tracking, automatic updates, and support for multiple source types (git repositories, local paths, and more).

Creating and distributing a marketplace involves:

1. **Creating plugins**: build one or more plugins with commands, agents, hooks, MCP servers, or LSP servers
2. **Creating a marketplace file**: define a `marketplace.json` that lists your plugins and where to find them
3. **Host the marketplace**: push to GitHub, GitLab, or another git host
4. **Share with users**: users add your marketplace with `/plugin marketplace add`

## Create a marketplace file

Create `.claude-plugin/marketplace.json` in your repository root. This file defines your marketplace's name, owner information, and a list of plugins with their sources.

```json
{
  "name": "company-tools",
  "owner": {
    "name": "DevTools Team",
    "email": "devtools@example.com"
  },
  "plugins": [
    {
      "name": "code-formatter",
      "source": "./plugins/formatter",
      "description": "Automatic code formatting on save",
      "version": "2.1.0"
    },
    {
      "name": "deployment-tools",
      "source": {
        "source": "github",
        "repo": "company/deploy-plugin"
      },
      "description": "Deployment automation tools"
    }
  ]
}
```

## Marketplace schema

### Required fields

| Field     | Type   | Description                                                                                                                                           | Example        |
| :-------- | :----- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| `name`    | string | Marketplace identifier (kebab-case, no spaces). This is public-facing.                                                                               | `"acme-tools"` |
| `owner`   | object | Marketplace maintainer information with `name` (required) and optional `email`                                                                       |                |
| `plugins` | array  | List of available plugins                                                                                                                            | See below      |

**Reserved names**: The following marketplace names are reserved for official Anthropic use: `claude-code-marketplace`, `claude-code-plugins`, `claude-plugins-official`, `anthropic-marketplace`, `anthropic-plugins`, `agent-skills`, `knowledge-work-plugins`, `life-sciences`.

### Plugin entries

Each plugin entry in the `plugins` array describes a plugin and where to find it.

**Required fields**:

| Field    | Type           | Description                                                                                                                            |
| :------- | :------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `name`   | string         | Plugin identifier (kebab-case, no spaces).                                                                                             |
| `source` | string\|object | Where to fetch the plugin from (see [Plugin sources](#plugin-sources) below)                                                           |

**Optional plugin fields**:

| Field         | Type    | Description                                              |
| :------------ | :------ | :------------------------------------------------------- |
| `description` | string  | Brief plugin description                                 |
| `version`     | string  | Plugin version                                           |
| `author`      | object  | Plugin author information                                |
| `homepage`    | string  | Plugin homepage or documentation URL                     |
| `repository`  | string  | Source code repository URL                               |
| `license`     | string  | SPDX license identifier                                  |
| `keywords`    | array   | Tags for plugin discovery                                |

## Plugin sources

Plugin sources tell Claude Code where to fetch each individual plugin. These are set in the `source` field of each plugin entry.

| Source        | Type                      | Example                                    |
| ------------- | ------------------------- | ------------------------------------------ |
| Relative path | `string`                  | `"./plugins/my-plugin"`                    |
| GitHub        | object with `source`, `repo` | `{"source": "github", "repo": "owner/repo"}` |
| Git URL       | object with `source`, `url`  | `{"source": "url", "url": "https://..."}`  |
| npm           | object with `source`, `package` | `{"source": "npm", "package": "@org/plugin"}` |

## Host and distribute marketplaces

### Host on GitHub (recommended)

1. **Create a repository**: Set up a new repository for your marketplace
2. **Add marketplace file**: Create `.claude-plugin/marketplace.json`
3. **Share with users**: Users add with `/plugin marketplace add owner/repo`

### Test locally before distribution

Test your marketplace locally:

```shell
/plugin marketplace add ./my-local-marketplace
/plugin install test-plugin@my-local-marketplace
```

### Configure team marketplaces

Add your marketplace to `.claude/settings.json` so team members are automatically prompted to install it:

```json
{
  "extraKnownMarketplaces": {
    "company-tools": {
      "source": {
        "source": "github",
        "repo": "your-org/claude-plugins"
      }
    }
  }
}
```

## Manage marketplaces from the CLI

### Plugin marketplace add

Add a marketplace from GitHub, git URL, or local path:

```bash
claude plugin marketplace add <source> [options]
```

Examples:

```bash
# From GitHub
claude plugin marketplace add acme-corp/claude-plugins

# From Git URL
claude plugin marketplace add https://gitlab.com/team/plugins.git

# From local directory
claude plugin marketplace add ./my-marketplace
```

### Plugin marketplace list

List all configured marketplaces:

```bash
claude plugin marketplace list [options]
```

### Plugin marketplace remove

Remove a configured marketplace:

```bash
claude plugin marketplace remove <name>
```

**Warning**: Removing a marketplace also uninstalls any plugins you installed from it.

### Plugin marketplace update

Refresh marketplaces from their sources:

```bash
claude plugin marketplace update [name]
```

## Troubleshooting

### Marketplace not loading

* Verify the marketplace URL is accessible
* Check that `.claude-plugin/marketplace.json` exists at the path
* Ensure JSON syntax is valid using `/plugin validate`

### Marketplace validation errors

Run `/plugin validate .` to check for issues:

| Error                                             | Cause                            | Solution                          |
| :------------------------------------------------ | :------------------------------- | :-------------------------------- |
| `File not found: .claude-plugin/marketplace.json` | Missing manifest                 | Create `.claude-plugin/marketplace.json` |
| `Invalid JSON syntax: ...`                        | JSON syntax error                | Fix JSON syntax                   |
| `Duplicate plugin name "x" found`                 | Two plugins share the same name  | Give each plugin a unique name    |

### Plugin installation failures

* Verify plugin source URLs are accessible
* Check that plugin directories contain required files
* For GitHub sources, ensure repositories are public or you have access

### Marketplace updates fail in offline environments

Set `CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE=1` to keep the existing cache when `git pull` fails:

```bash
export CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE=1
```

## See also

* [Discover and install prebuilt plugins](/en/discover-plugins) - Installing plugins from existing marketplaces
* [Plugins](/en/plugins) - Creating your own plugins
* [Plugins reference](/en/plugins-reference) - Complete technical specifications
