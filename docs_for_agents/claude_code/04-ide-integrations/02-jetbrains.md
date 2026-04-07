# JetBrains IDEs

Use Claude Code with JetBrains IDEs including IntelliJ, PyCharm, WebStorm, and more

Claude Code integrates with JetBrains IDEs through a dedicated plugin, providing features like interactive diff viewing, selection context sharing, and more.

## Supported IDEs

The Claude Code plugin works with most JetBrains IDEs, including:

* IntelliJ IDEA
* PyCharm
* Android Studio
* WebStorm
* PhpStorm
* GoLand

## Features

* **Quick launch**: Use `Cmd+Esc` (Mac) or `Ctrl+Esc` (Windows/Linux) to open Claude Code directly from your editor, or click the Claude Code button in the UI
* **Diff viewing**: Code changes can be displayed directly in the IDE diff viewer instead of the terminal
* **Selection context**: The current selection/tab in the IDE is automatically shared with Claude Code
* **File reference shortcuts**: Use `Cmd+Option+K` (Mac) or `Alt+Ctrl+K` (Linux/Windows) to insert file references (for example, @File#L1-99)
* **Diagnostic sharing**: Diagnostic errors (lint, syntax, etc.) from the IDE are automatically shared with Claude as you work

## Installation

### Marketplace Installation

Find and install the [Claude Code plugin](https://plugins.jetbrains.com/plugin/27310-claude-code-beta-) from the JetBrains marketplace and restart your IDE.

If you haven't installed Claude Code yet, see [our quickstart guide](/en/quickstart) for installation instructions.

> After installing the plugin, you may need to restart your IDE completely for it to take effect.

## Usage

### From Your IDE

Run `claude` from your IDE's integrated terminal, and all integration features will be active.

### From External Terminals

Use the `/ide` command in any external terminal to connect Claude Code to your JetBrains IDE and activate all features:

```bash
claude
```

```
/ide
```

If you want Claude to have access to the same files as your IDE, start Claude Code from the same directory as your IDE project root.

## Configuration

### Claude Code Settings

Configure IDE integration through Claude Code's settings:

1. Run `claude`
2. Enter the `/config` command
3. Set the diff tool to `auto` for automatic IDE detection

### Plugin Settings

Configure the Claude Code plugin by going to **Settings → Tools → Claude Code \[Beta]**:

#### General Settings

* **Claude command**: Specify a custom command to run Claude (for example, `claude`, `/usr/local/bin/claude`, or `npx @anthropic/claude`)
* **Suppress notification for Claude command not found**: Skip notifications about not finding the Claude command
* **Enable using Option+Enter for multi-line prompts** (macOS only): When enabled, Option+Enter inserts new lines in Claude Code prompts. Disable if experiencing issues with the Option key being captured unexpectedly (requires terminal restart)
* **Enable automatic updates**: Automatically check for and install plugin updates (applied on restart)

> For WSL users: Set `wsl -d Ubuntu -- bash -lic "claude"` as your Claude command (replace `Ubuntu` with your WSL distribution name)

#### ESC Key Configuration

If the ESC key doesn't interrupt Claude Code operations in JetBrains terminals:

1. Go to **Settings → Tools → Terminal**
2. Either:
   * Uncheck "Move focus to the editor with Escape", or
   * Click "Configure terminal keybindings" and delete the "Switch focus to Editor" shortcut
3. Apply the changes

This allows the ESC key to properly interrupt Claude Code operations.

## Special Configurations

### Remote Development

> When using JetBrains Remote Development, you must install the plugin in the remote host via **Settings → Plugin (Host)**.

The plugin must be installed on the remote host, not on your local client machine.

### WSL Configuration

> WSL users may need additional configuration for IDE detection to work properly. See our [WSL troubleshooting guide](/en/troubleshooting#jetbrains-ide-not-detected-on-wsl2) for detailed setup instructions.

WSL configuration may require:

* Proper terminal configuration
* Networking mode adjustments
* Firewall settings updates

## Troubleshooting

### Plugin Not Working

* Ensure you're running Claude Code from the project root directory
* Check that the JetBrains plugin is enabled in the IDE settings
* Completely restart the IDE (you may need to do this multiple times)
* For Remote Development, ensure the plugin is installed in the remote host

### IDE Not Detected

* Verify the plugin is installed and enabled
* Restart the IDE completely
* Check that you're running Claude Code from the integrated terminal
* For WSL users, see the [WSL troubleshooting guide](/en/troubleshooting#jetbrains-ide-not-detected-on-wsl2)

### Command Not Found

If clicking the Claude icon shows "command not found":

1. Verify Claude Code is installed: `npm list -g @anthropic-ai/claude-code`
2. Configure the Claude command path in plugin settings
3. For WSL users, use the WSL command format mentioned in the configuration section

## Security Considerations

When Claude Code runs in a JetBrains IDE with auto-edit permissions enabled, it may be able to modify IDE configuration files that can be automatically executed by your IDE. This may increase the risk of running Claude Code in auto-edit mode and allow bypassing Claude Code's permission prompts for bash execution.

When running in JetBrains IDEs, consider:

* Using manual approval mode for edits
* Taking extra care to ensure Claude is only used with trusted prompts
* Being aware of which files Claude Code has access to modify

For additional help, see our [troubleshooting guide](/en/troubleshooting).
