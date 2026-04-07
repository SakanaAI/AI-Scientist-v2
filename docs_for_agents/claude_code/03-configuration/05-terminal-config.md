# Optimize your terminal setup

> Claude Code works best when your terminal is properly configured. Follow these guidelines to optimize your experience.

### Themes and appearance

Claude cannot control the theme of your terminal. That's handled by your terminal application. You can match Claude Code's theme to your terminal any time via the `/config` command.

For additional customization of the Claude Code interface itself, you can configure a [custom status line](/en/statusline) to display contextual information like the current model, working directory, or git branch at the bottom of your terminal.

### Line breaks

You have several options for entering line breaks into Claude Code:

* **Quick escape**: Type `\` followed by Enter to create a newline
* **Ctrl+J**: Sends a line feed character, which works as a newline in any terminal without configuration
* **Shift+Enter**: Works out of the box in iTerm2, WezTerm, Ghostty, and Kitty
* **Keyboard shortcut**: Set up a keybinding to insert a newline in other terminals

**Set up Shift+Enter for other terminals**

Run `/terminal-setup` within Claude Code to automatically configure Shift+Enter for VS Code, Alacritty, Zed, and Warp.

<Note>
  The `/terminal-setup` command is only visible in terminals that require manual configuration. If you're using iTerm2, WezTerm, Ghostty, or Kitty, you won't see this command because Shift+Enter already works natively.
</Note>

**Set up Option+Enter (VS Code, iTerm2 or macOS Terminal.app)**

**For Mac Terminal.app:**

1. Open Settings → Profiles → Keyboard
2. Check "Use Option as Meta Key"

**For iTerm2:**

1. Open Settings → Profiles → Keys
2. Under General, set Left/Right Option key to "Esc+"

**For VS Code terminal:**

Set `"terminal.integrated.macOptionIsMeta": true` in VS Code settings.

### Notification setup

When Claude finishes working and is waiting for your input, it fires a notification event. You can surface this event as a desktop notification through your terminal or run custom logic with [notification hooks](/en/hooks#notification).

#### Terminal notifications

Kitty and Ghostty support desktop notifications without additional configuration. iTerm 2 requires setup:

1. Open iTerm 2 Settings → Profiles → Terminal
2. Enable "Notification Center Alerts"
3. Click "Filter Alerts" and check "Send escape sequence-generated alerts"

If notifications aren't appearing, verify that your terminal app has notification permissions in your OS settings.

When running Claude Code inside tmux, notifications and the [terminal progress bar](/en/settings#global-config-settings) only reach the outer terminal, such as iTerm2, Kitty, or Ghostty, if you enable passthrough in your tmux configuration:

```
set -g allow-passthrough on
```

Without this setting, tmux intercepts the escape sequences and they do not reach the terminal application.

Other terminals, including the default macOS Terminal, do not support native notifications. Use [notification hooks](/en/hooks#notification) instead.

#### Notification hooks

To add custom behavior when notifications fire, such as playing a sound or sending a message, configure a [notification hook](/en/hooks#notification). Hooks run alongside terminal notifications, not as a replacement.

### Reduce flicker and memory usage

If you see flicker during long sessions, or your terminal scroll position jumps to the top while Claude is working, try [fullscreen rendering](/en/fullscreen). It uses an alternate rendering path that keeps memory flat and adds mouse support. Enable it with `CLAUDE_CODE_NO_FLICKER=1`.

### Handling large inputs

When working with extensive code or long instructions:

* **Avoid direct pasting**: Claude Code may struggle with very long pasted content
* **Use file-based workflows**: Write content to a file and ask Claude to read it
* **Be aware of VS Code limitations**: The VS Code terminal is particularly prone to truncating long pastes

### Vim Mode

Claude Code supports a subset of Vim keybindings that can be enabled via `/config` → Editor mode. To set the mode directly in your config file, set the [`editorMode`](/en/settings#global-config-settings) global config key to `"vim"` in `~/.claude.json`.

The supported subset includes:

* Mode switching: `Esc` (to NORMAL), `i`/`I`, `a`/`A`, `o`/`O` (to INSERT)
* Navigation: `h`/`j`/`k`/`l`, `w`/`e`/`b`, `0`/`$`/`^`, `gg`/`G`, `f`/`F`/`t`/`T` with `;`/`,` repeat
* Editing: `x`, `dw`/`de`/`db`/`dd`/`D`, `cw`/`ce`/`cb`/`cc`/`C`, `.` (repeat)
* Yank/paste: `yy`/`Y`, `yw`/`ye`/`yb`, `p`/`P`
* Text objects: `iw`/`aw`, `iW`/`aW`, `i"`/`a"`, `i'`/`a'`, `i(`/`a(`, `i[`/`a[`, `i{`/`a{`
* Indentation: `>>`/`<<`
* Line operations: `J` (join lines)

See [Interactive mode](/en/interactive-mode#vim-editor-mode) for the complete reference.
