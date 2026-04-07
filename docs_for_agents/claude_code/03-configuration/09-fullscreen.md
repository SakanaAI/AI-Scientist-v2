# Fullscreen rendering

> Enable a smoother, flicker-free rendering mode with mouse support and stable memory usage in long conversations.

<Note>
  Fullscreen rendering is an opt-in [research preview](#research-preview) and requires Claude Code v2.1.89 or later. Enable it with `CLAUDE_CODE_NO_FLICKER=1`. Behavior may change based on feedback.
</Note>

Fullscreen rendering is an alternative rendering path for the Claude Code CLI that eliminates flicker, keeps memory usage flat in long conversations, and adds mouse support. It draws the interface on the terminal's alternate screen buffer, like `vim` or `htop`, and only renders messages that are currently visible. This reduces the amount of data sent to your terminal on each update.

The difference is most noticeable in terminal emulators where rendering throughput is the bottleneck, such as the VS Code integrated terminal, tmux, and iTerm2. If your terminal scroll position jumps to the top while Claude is working, or the screen flashes as tool output streams in, this mode addresses those.

<Note>
  The term fullscreen describes how Claude Code takes over the terminal's drawing surface, the way `vim` does. It has nothing to do with maximizing your terminal window, and works at any window size.
</Note>

## Enable fullscreen rendering

Set the `CLAUDE_CODE_NO_FLICKER` environment variable when starting Claude Code:

```bash  theme={null}
CLAUDE_CODE_NO_FLICKER=1 claude
```

To enable it for every session, export the variable in your shell profile such as `~/.zshrc` or `~/.bashrc`:

```bash  theme={null}
export CLAUDE_CODE_NO_FLICKER=1
```

## What changes

Fullscreen rendering changes how the CLI draws to your terminal. The input box stays fixed at the bottom of the screen instead of moving as output streams in. If the input stays put while Claude is working, fullscreen rendering is active. Only visible messages are kept in the render tree, so memory stays constant regardless of conversation length.

Because the conversation lives in the alternate screen buffer instead of your terminal's scrollback, a few things work differently:

| Before                                              | Now                                                                                                                     | Details                                                                   |
| :-------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| `Cmd+f` or tmux search to find text                 | `Ctrl+o` then `/` to search, or `Ctrl+o` then `[` to write the conversation to native scrollback so `Cmd+f` works again | [Search and review the conversation](#search-and-review-the-conversation) |
| Terminal's native click-and-drag to select and copy | In-app selection, copies automatically on mouse release                                                                 | [Use the mouse](#use-the-mouse)                                           |
| `Cmd`-click to open a URL                           | Click the URL                                                                                                           | [Use the mouse](#use-the-mouse)                                           |

If mouse capture interferes with your workflow, you can [turn it off](#keep-native-text-selection) while keeping the flicker-free rendering.

## Use the mouse

Fullscreen rendering captures mouse events and handles them inside Claude Code:

* **Click in the prompt input** to position your cursor anywhere in the text you're typing.
* **Click a collapsed tool result** to expand it and see the full output. Click again to collapse. The tool call and its result expand together. Only messages that have more to show are clickable.
* **Click a URL or file path** to open it. File paths in tool output, like the ones printed after an Edit or Write, open in your default application. Plain `http://` and `https://` URLs open in your browser. In most terminals this replaces native `Cmd`-click or `Ctrl`-click, which mouse capture intercepts. In the VS Code integrated terminal and similar xterm.js-based terminals, keep using `Cmd`-click. Claude Code defers to the terminal's own link handler there to avoid opening links twice.
* **Click and drag** to select text anywhere in the conversation. Double-click selects a word, matching iTerm2's word boundaries so a file path selects as one unit. Triple-click selects the line.
* **Scroll with the mouse wheel** to move through the conversation.

Selected text copies to your clipboard automatically on mouse release. To turn this off, toggle Copy on select in `/config`. With it off, press `Ctrl+Shift+c` to copy manually. On terminals that support the kitty keyboard protocol, such as kitty, WezTerm, Ghostty, and iTerm2, `Cmd+c` also works. If you have a selection active, `Ctrl+c` copies instead of cancelling.

## Scroll the conversation

Fullscreen rendering handles scrolling inside the app. Use these shortcuts to navigate:

| Shortcut        | Action                                               |
| :-------------- | :--------------------------------------------------- |
| `PgUp` / `PgDn` | Scroll up or down by half a screen                   |
| `Ctrl+Home`     | Jump to the start of the conversation                |
| `Ctrl+End`      | Jump to the latest message and re-enable auto-follow |
| Mouse wheel     | Scroll a few lines at a time                         |

On keyboards without dedicated `PgUp`, `PgDn`, `Home`, or `End` keys, like MacBook keyboards, hold `Fn` with the arrow keys: `Fn+↑` sends `PgUp`, `Fn+↓` sends `PgDn`, `Fn+←` sends `Home`, and `Fn+→` sends `End`. That makes `Ctrl+Fn+→` the jump-to-bottom shortcut. If that feels awkward, scroll to the bottom with the mouse wheel to resume following.

Scrolling up pauses auto-follow so new output does not pull you back to the bottom. Press `Ctrl+End` or scroll to the bottom to resume following.

Mouse wheel scrolling requires your terminal to forward mouse events to Claude Code. Most terminals do this whenever an application requests it. iTerm2 makes it a per-profile setting: if the wheel does nothing but `PgUp` and `PgDn` work, open Settings → Profiles → Terminal and turn on Enable mouse reporting. The same setting is also required for click-to-expand and text selection to work.

### Adjust wheel scroll speed

If mouse wheel scrolling feels slow, your terminal may be sending one scroll event per physical notch with no multiplier. Some terminals, like Ghostty and iTerm2 with faster scrolling enabled, already amplify wheel events. Others, including the VS Code integrated terminal, send exactly one event per notch. Claude Code cannot detect which.

Set `CLAUDE_CODE_SCROLL_SPEED` to multiply the base scroll distance:

```bash  theme={null}
export CLAUDE_CODE_SCROLL_SPEED=3
```

A value of `3` matches the default in `vim` and similar applications. The setting accepts values from 1 to 20.

## Search and review the conversation

Press `Ctrl+o` to enter transcript mode. With fullscreen rendering active, transcript mode gains `less`-style navigation and search:

| Key                                  | Action                                                                                                 |
| :----------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `/`                                  | Open search. Type to find matches, `Enter` to accept, `Esc` to cancel and restore your scroll position |
| `n` / `N`                            | Jump to next or previous match. Works after you've closed the search bar                               |
| `j` / `k` or `↑` / `↓`               | Scroll one line                                                                                        |
| `g` / `G` or `Home` / `End`          | Jump to top or bottom                                                                                  |
| `Ctrl+u` / `Ctrl+d`                  | Scroll half a page                                                                                     |
| `Ctrl+b` / `Ctrl+f` or `Space` / `b` | Scroll a full page                                                                                     |
| `Esc`, `q`, or `Ctrl+o`              | Exit transcript mode                                                                                   |

Your terminal's `Cmd+f` and tmux search don't see the conversation because it lives in the alternate screen buffer, not the native scrollback. To hand the content back to your terminal, press `Ctrl+o` to enter transcript mode first, then:

* **`[`**: writes the full conversation into your terminal's native scrollback buffer, with all tool output expanded. The conversation is now ordinary text in your terminal, so `Cmd+f`, tmux copy mode, and any other native tool can search or select it. Long sessions may pause for a moment while this happens. This lasts until you exit transcript mode with `Esc` or `q`, which returns you to fullscreen rendering. The next `Ctrl+o` starts fresh.
* **`v`**: writes the conversation to a temporary file and opens it in `$VISUAL` or `$EDITOR`.

Press `Esc`, `q`, or `Ctrl+o` to return to the prompt.

## Use with tmux

Fullscreen rendering works inside tmux, with two caveats.

Mouse wheel scrolling requires tmux's mouse mode. If your `~/.tmux.conf` does not already enable it, add this line and reload your config:

```bash  theme={null}
set -g mouse on
```

Without mouse mode, wheel events go to tmux instead of Claude Code. Keyboard scrolling with `PgUp` and `PgDn` works either way. Claude Code prints a one-time hint at startup if it detects tmux with mouse mode off.

Fullscreen rendering is incompatible with iTerm2's tmux integration mode, which is the mode you enter with `tmux -CC`. In integration mode, iTerm2 renders each tmux pane as a native split rather than letting tmux draw to the terminal. The alternate screen buffer and mouse tracking do not work correctly there: the mouse wheel does nothing, and double-click can corrupt the terminal state. Don't enable fullscreen rendering in `tmux -CC` sessions. Regular tmux inside iTerm2, without `-CC`, works fine.

## Keep native text selection

Mouse capture is the most common friction point, especially over SSH or inside tmux. When Claude Code captures mouse events, your terminal's native copy-on-select stops working. The selection you make with click-and-drag exists inside Claude Code, not in your terminal's selection buffer, so tmux copy mode, Kitty hints, and similar tools don't see it.

Claude Code tries to write the selection to your clipboard, but the path it uses depends on your setup. Inside tmux it writes to the tmux paste buffer. Over SSH it falls back to OSC 52 escape sequences, which some terminals block by default. Claude Code prints a toast after each copy telling you which path it used.

If you rely on your terminal's native selection, set `CLAUDE_CODE_DISABLE_MOUSE=1` to opt out of mouse capture while keeping the flicker-free rendering and flat memory:

```bash  theme={null}
CLAUDE_CODE_NO_FLICKER=1 CLAUDE_CODE_DISABLE_MOUSE=1 claude
```

With mouse capture disabled, keyboard scrolling with `PgUp`, `PgDn`, `Ctrl+Home`, and `Ctrl+End` still works, and your terminal handles selection natively. You lose click-to-position-cursor, click-to-expand tool output, URL clicking, and wheel scrolling inside Claude Code.

## Research preview

Fullscreen rendering is a research preview feature. It has been tested on common terminal emulators, but you may encounter rendering issues on less common terminals or unusual configurations.

If you encounter a problem, run `/feedback` inside Claude Code to report it, or open an issue on the [claude-code GitHub repo](https://github.com/anthropics/claude-code/issues). Include your terminal emulator name and version.

To turn fullscreen rendering off, unset the environment variable or set `CLAUDE_CODE_NO_FLICKER=0`.
