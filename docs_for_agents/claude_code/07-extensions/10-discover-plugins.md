# Discover and install prebuilt plugins through marketplaces

> Find and install plugins from marketplaces to extend Claude Code with new commands, agents, and capabilities.

Plugins extend Claude Code with skills, agents, hooks, and MCP servers. Plugin marketplaces are catalogs that help you discover and install these extensions without building them yourself.

Looking to create and distribute your own marketplace? See [Create and distribute a plugin marketplace](/en/plugin-marketplaces).

## How marketplaces work

A marketplace is a catalog of plugins that someone else has created and shared. Using a marketplace is a two-step process:

1. **Add the marketplace**: This registers the catalog with Claude Code so you can browse what's available. No plugins are installed yet.
2. **Install individual plugins**: Browse the catalog and install the plugins you want.

Think of it like adding an app store: adding the store gives you access to browse its collection, but you still choose which apps to download individually.

## Official Anthropic marketplace

The official Anthropic marketplace (`claude-plugins-official`) is automatically available when you start Claude Code. Run `/plugin` and go to the **Discover** tab to browse what's available, or view the catalog at [claude.com/plugins](https://claude.com/plugins).

To install a plugin from the official marketplace, use `/plugin install <name>@claude-plugins-official`. For example:

```shell
/plugin install github@claude-plugins-official
```

The official marketplace is maintained by Anthropic. To submit a plugin to the official marketplace, use one of the in-app submission forms:

* **Claude.ai**: [claude.ai/settings/plugins/submit](https://claude.ai/settings/plugins/submit)
* **Console**: [platform.claude.com/plugins/submit](https://platform.claude.com/plugins/submit)

To distribute plugins independently, create your own marketplace and share it with users.

## Add marketplaces

Use the `/plugin marketplace add` command to add marketplaces from different sources:

* **GitHub repositories**: `owner/repo` format
* **Git URLs**: any git repository URL
* **Local paths**: directories or direct paths to `marketplace.json` files
* **Remote URLs**: direct URLs to hosted `marketplace.json` files

### Add from GitHub

```shell
/plugin marketplace add anthropics/claude-code
```

### Add from other Git hosts

Using HTTPS:

```shell
/plugin marketplace add https://gitlab.com/company/plugins.git
```

Using SSH:

```shell
/plugin marketplace add git@gitlab.com:company/plugins.git
```

### Add from local paths

```shell
/plugin marketplace add ./my-marketplace
```

### Add from remote URLs

```shell
/plugin marketplace add https://example.com/marketplace.json
```

## Install plugins

Once you've added marketplaces, install plugins directly:

```shell
/plugin install plugin-name@marketplace-name
```

To choose a different installation scope (user, project, or local), use the interactive UI: run `/plugin`, go to the **Discover** tab, and press **Enter** on a plugin.

Run `/plugin` and go to the **Installed** tab to see your plugins grouped by scope.

**Warning**: Make sure you trust a plugin before installing it. Anthropic does not control what MCP servers, files, or other software are included in plugins.

## Manage installed plugins

Disable a plugin without uninstalling:

```shell
/plugin disable plugin-name@marketplace-name
```

Re-enable a disabled plugin:

```shell
/plugin enable plugin-name@marketplace-name
```

Completely remove a plugin:

```shell
/plugin uninstall plugin-name@marketplace-name
```

When you install, enable, or disable plugins during a session, run `/reload-plugins` to pick up changes without restarting.

## Manage marketplaces

List all configured marketplaces:

```shell
/plugin marketplace list
```

Update marketplace listings:

```shell
/plugin marketplace update marketplace-name
```

Remove a marketplace:

```shell
/plugin marketplace remove marketplace-name
```

**Warning**: Removing a marketplace will uninstall any plugins you installed from it.

## Configure team marketplaces

Team admins can set up automatic marketplace installation for projects by adding marketplace configuration to `.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "my-team-tools": {
      "source": {
        "source": "github",
        "repo": "your-org/claude-plugins"
      }
    }
  }
}
```

## Security

Plugins and marketplaces are highly trusted components that can execute arbitrary code on your machine with your user privileges. Only install plugins and add marketplaces from sources you trust.

## Troubleshooting

### /plugin command not recognized

1. Check your version: `claude --version`
2. Update Claude Code to the latest version
3. Restart your terminal and run `claude` again

### Marketplace not loading

* Verify the URL is accessible
* Check that `.claude-plugin/marketplace.json` exists at the path
* Ensure JSON syntax is valid

### Plugin installation failures

* Verify plugin source URLs are accessible
* Check that plugin directories contain required files
* For GitHub sources, ensure repositories are public or you have access

For more help, see [Troubleshooting](/en/plugin-marketplaces#troubleshooting) in the marketplace guide.

## Next steps

* **Build your own plugins**: See [Plugins](/en/plugins) to create skills, agents, and hooks
* **Create a marketplace**: See [Create a plugin marketplace](/en/plugin-marketplaces) to distribute plugins to your team or community
* **Technical reference**: See [Plugins reference](/en/plugins-reference) for complete specifications
