# Interactive mode

Complete reference for keyboard shortcuts, input modes, and interactive features in Claude Code sessions.

## Keyboard shortcuts

Keyboard shortcuts may vary by platform and terminal. Press `?` to see available shortcuts for your environment.

**macOS users**: Option/Alt key shortcuts (`Alt+B`, `Alt+F`, `Alt+Y`, `Alt+M`, `Alt+P`, `Alt+T`) require configuring Option as Meta in your terminal:

* **iTerm2**: settings → Profiles → Keys → set Left/Right Option key to "Esc+"
* **Terminal.app**: settings → Profiles → Keyboard → check "Use Option as Meta Key"
* **VS Code**: set `"terminal.integrated.macOptionIsMeta": true` in VS Code settings

See [Terminal configuration](/en/terminal-config) for details.

### General controls

| Shortcut                                          | Description                                                         | Context                                                                                                                                                              |
| :------------------------------------------------ | :------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ctrl+C`                                          | Cancel current input or generation                                  | Standard interrupt                                                                                                                                                   |
| `Ctrl+X Ctrl+K`                                   | Kill all background agents. Press twice within 3 seconds to confirm | Background agent control                                                                                                                                             |
| `Ctrl+D`                                          | Exit Claude Code session                                            | EOF signal                                                                                                                                                           |
| `Ctrl+G` or `Ctrl+X Ctrl+E`                       | Open in default text editor                                         | Edit your prompt or custom response in your default text editor. `Ctrl+X Ctrl+E` is the readline-native binding                                                      |
| `Ctrl+L`                                          | Redraw the screen                                                   | Repaints the current UI without clearing conversation history                                                                                                        |
| `Ctrl+O`                                          | Toggle verbose output                                               | Shows detailed tool usage and execution. Also expands MCP read and search calls, which collapse to a single line like "Queried slack" by default                     |
| `Ctrl+R`                                          | Reverse search command history                                      | Search through previous commands interactively                                                                                                                       |
| `Ctrl+V` or `Cmd+V` (iTerm2) or `Alt+V` (Windows) | Paste image from clipboard                                          | Inserts an `[Image #N]` chip at the cursor so you can reference it positionally in your prompt                                                                       |
| `Ctrl+B`                                          | Background running tasks                                            | Backgrounds bash commands and agents. Tmux users press twice                                                                                                         |
| `Ctrl+T`                                          | Toggle task list                                                    | Show or hide the [task list](#task-list) in the terminal status area                                                                                                 |
| `Left/Right arrows`                               | Cycle through dialog tabs                                           | Navigate between tabs in permission dialogs and menus                                                                                                                |
| `Up/Down arrows`                                  | Navigate command history                                            | Recall previous inputs                                                                                                                                               |
| `Esc` + `Esc`                                     | Rewind or summarize                                                 | Restore code and/or conversation to a previous point, or summarize from a selected message                                                                           |
| `Shift+Tab` or `Alt+M` (some configurations)      | Cycle permission modes                                              | Cycle through `default`, `acceptEdits`, `plan`, and any modes you have enabled, such as `auto` or `bypassPermissions`. See [permission modes](/en/permission-modes). |
| `Option+P` (macOS) or `Alt+P` (Windows/Linux)     | Switch model                                                        | Switch models without clearing your prompt                                                                                                                           |
| `Option+T` (macOS) or `Alt+T` (Windows/Linux)     | Toggle extended thinking                                            | Enable or disable extended thinking mode. On macOS, configure your terminal to send Option as Meta for this shortcut to work                                         |
| `Option+O` (macOS) or `Alt+O` (Windows/Linux)     | Toggle fast mode                                                    | Enable or disable [fast mode](/en/fast-mode)                                                                                                                         |

### Text editing

| Shortcut                 | Description                      | Context                                                                                                       |
| :----------------------- | :------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| `Ctrl+K`                 | Delete to end of line            | Stores deleted text for pasting                                                                               |
| `Ctrl+U`                 | Delete from cursor to line start | Stores deleted text for pasting. Repeat to clear across lines in multiline input                              |
| `Ctrl+Y`                 | Paste deleted text               | Paste text deleted with `Ctrl+K` or `Ctrl+U`                                                                  |
| `Alt+Y` (after `Ctrl+Y`) | Cycle paste history              | After pasting, cycle through previously deleted text. Requires [Option as Meta](#keyboard-shortcuts) on macOS |
| `Alt+B`                  | Move cursor back one word        | Word navigation. Requires [Option as Meta](#keyboard-shortcuts) on macOS                                      |
| `Alt+F`                  | Move cursor forward one word     | Word navigation. Requires [Option as Meta](#keyboard-shortcuts) on macOS                                      |

### Theme and display

| Shortcut | Description                                                        | Context                                                                                                      |
| :------- | :----------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| `Ctrl+T` | Toggle syntax highlighting for code blocks (inside `/theme` menu)  | Only works inside the `/theme` picker menu. Controls whether code in Claude's responses uses syntax coloring |

### Multiline input

| Method           | Shortcut       | Context                                                 |
| :--------------- | :------------- | :------------------------------------------------------ |
| Quick escape     | `\` + `Enter`  | Works in all terminals                                  |
| macOS default    | `Option+Enter` | Default on macOS                                        |
| Shift+Enter      | `Shift+Enter`  | Works out of the box in iTerm2, WezTerm, Ghostty, Kitty |
| Control sequence | `Ctrl+J`       | Line feed character for multiline                       |
| Paste mode       | Paste directly | For code blocks, logs                                   |

Shift+Enter works without configuration in iTerm2, WezTerm, Ghostty, and Kitty. For other terminals (VS Code, Alacritty, Zed, Warp), run `/terminal-setup` to install the binding.

### Quick commands

| Shortcut     | Description       | Notes                                                                |
| :----------- | :---------------- | :------------------------------------------------------------------- |
| `/` at start | Command or skill  | See [built-in commands](#built-in-commands) and [skills](/en/skills) |
| `!` at start | Bash mode         | Run commands directly and add execution output to the session        |
| `@`          | File path mention | Trigger file path autocomplete                                       |

### Transcript viewer

When the transcript viewer is open (toggled with `Ctrl+O`), these shortcuts are available. `Ctrl+E` can be rebound via [`transcript:toggleShowAll`](/en/keybindings).

| Shortcut             | Description                                                                             |
| :------------------- | :-------------------------------------------------------------------------------------- |
| `Ctrl+E`             | Toggle show all content                                                                 |
| `q`, `Ctrl+C`, `Esc` | Exit transcript view. All three can be rebound via [`transcript:exit`](/en/keybindings) |

### Voice input

| Shortcut     | Description            | Notes                                                                                                                                                      |
| :----------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hold `Space` | Push-to-talk dictation | Requires [voice dictation](/en/voice-dictation) to be enabled. Transcript inserts at cursor. [Rebindable](/en/voice-dictation#rebind-the-push-to-talk-key) |

## Built-in commands

Type `/` in Claude Code to see all available commands, or type `/` followed by any letters to filter. The `/` menu shows both built-in commands and [bundled skills](/en/skills#bundled-skills) like `/simplify`. Not all commands are visible to every user since some depend on your platform or plan.

See the [commands reference](/en/commands) for the full list of built-in commands. To create your own commands, see [skills](/en/skills).

## Vim editor mode

Enable vim-style editing via `/config` → Editor mode.

### Mode switching

| Command | Action                      | From mode |
| :------ | :--------------------------- | :-------- |
| `Esc`   | Enter NORMAL mode           | INSERT    |
| `i`     | Insert before cursor        | NORMAL    |
| `I`     | Insert at beginning of line | NORMAL    |
| `a`     | Insert after cursor         | NORMAL    |
| `A`     | Insert at end of line       | NORMAL    |
| `o`     | Open line below             | NORMAL    |
| `O`     | Open line above             | NORMAL    |

### Navigation (NORMAL mode)

| Command         | Action                                              |
| :-------------- | :-------------------------------------------------- |
| `h`/`j`/`k`/`l` | Move left/down/up/right                             |
| `w`             | Next word                                           |
| `e`             | End of word                                         |
| `b`             | Previous word                                       |
| `0`             | Beginning of line                                   |
| `$`             | End of line                                         |
| `^`             | First non-blank character                           |
| `gg`            | Beginning of input                                  |
| `G`             | End of input                                        |
| `f{char}`       | Jump to next occurrence of character                |
| `F{char}`       | Jump to previous occurrence of character            |
| `t{char}`       | Jump to just before next occurrence of character    |
| `T{char}`       | Jump to just after previous occurrence of character |
| `;`             | Repeat last f/F/t/T motion                          |
| `,`             | Repeat last f/F/t/T motion in reverse               |

In vim normal mode, if the cursor is at the beginning or end of input and cannot move further, the arrow keys navigate command history instead.

### Editing (NORMAL mode)

| Command        | Action                  |
| :------------- | :---------------------- |
| `x`            | Delete character        |
| `dd`           | Delete line             |
| `D`            | Delete to end of line   |
| `dw`/`de`/`db` | Delete word/to end/back |
| `cc`           | Change line             |
| `C`            | Change to end of line   |
| `cw`/`ce`/`cb` | Change word/to end/back |
| `yy`/`Y`       | Yank (copy) line        |
| `yw`/`ye`/`yb` | Yank word/to end/back   |
| `p`            | Paste after cursor      |
| `P`            | Paste before cursor     |
| `>>`           | Indent line             |
| `<<`           | Dedent line             |
| `J`            | Join lines              |
| `.`            | Repeat last change      |

### Text objects (NORMAL mode)

Text objects work with operators like `d`, `c`, and `y`:

| Command   | Action                                   |
| :-------- | :--------------------------------------- |
| `iw`/`aw` | Inner/around word                        |
| `iW`/`aW` | Inner/around WORD (whitespace-delimited) |
| `i"`/`a"` | Inner/around double quotes               |
| `i'`/`a'` | Inner/around single quotes               |
| `i(`/`a(` | Inner/around parentheses                 |
| `i[`/`a[` | Inner/around brackets                    |
| `i{`/`a{` | Inner/around braces                      |

## Command history

Claude Code maintains command history for the current session:

* Input history is stored per working directory
* Input history resets when you run `/clear` to start a new session. The previous session's conversation is preserved and can be resumed.
* Use Up/Down arrows to navigate (see keyboard shortcuts above)
* Note: history expansion (`!`) is disabled by default

### Reverse search with Ctrl+R

Press `Ctrl+R` to interactively search through your command history:

1. **Start search**: press `Ctrl+R` to activate reverse history search
2. **Type query**: enter text to search for in previous commands. The search term is highlighted in matching results
3. **Navigate matches**: press `Ctrl+R` again to cycle through older matches
4. **Accept match**:
   * Press `Tab` or `Esc` to accept the current match and continue editing
   * Press `Enter` to accept and execute the command immediately
5. **Cancel search**:
   * Press `Ctrl+C` to cancel and restore your original input
   * Press `Backspace` on empty search to cancel

The search displays matching commands with the search term highlighted, so you can find and reuse previous inputs.

## Background bash commands

Claude Code supports running bash commands in the background, allowing you to continue working while long-running processes execute.

### How backgrounding works

When Claude Code runs a command in the background, it runs the command asynchronously and immediately returns a background task ID. Claude Code can respond to new prompts while the command continues executing in the background.

To run commands in the background, you can either:

* Prompt Claude Code to run a command in the background
* Press Ctrl+B to move a regular Bash tool invocation to the background. (Tmux users must press Ctrl+B twice due to tmux's prefix key.)

**Key features:**

* Output is written to a file and Claude can retrieve it using the Read tool
* Background tasks have unique IDs for tracking and output retrieval
* Background tasks are automatically cleaned up when Claude Code exits
* Background tasks are automatically terminated if output exceeds 5GB, with a note in stderr explaining why

To disable all background task functionality, set the `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` environment variable to `1`. See [Environment variables](/en/env-vars) for details.

**Common backgrounded commands:**

* Build tools (webpack, vite, make)
* Package managers (npm, yarn, pnpm)
* Test runners (jest, pytest)
* Development servers
* Long-running processes (docker, terraform)

### Bash mode with `!` prefix

Run bash commands directly without going through Claude by prefixing your input with `!`:

```bash
! npm test
! git status
! ls -la
```

Bash mode:

* Adds the command and its output to the conversation context
* Shows real-time progress and output
* Supports the same `Ctrl+B` backgrounding for long-running commands
* Does not require Claude to interpret or approve the command
* Supports history-based autocomplete: type a partial command and press **Tab** to complete from previous `!` commands in the current project
* Exit with `Escape`, `Backspace`, or `Ctrl+U` on an empty prompt
* Pasting text that starts with `!` into an empty prompt enters bash mode automatically, matching typed `!` behavior

This is useful for quick shell operations while maintaining conversation context.

## Prompt suggestions

When you first open a session, a grayed-out example command appears in the prompt input to help you get started. Claude Code picks this from your project's git history, so it reflects files you've been working on recently.

After Claude responds, suggestions continue to appear based on your conversation history, such as a follow-up step from a multi-part request or a natural continuation of your workflow.

* Press **Tab** or **Right arrow** to accept the suggestion, or press **Enter** to accept and submit
* Start typing to dismiss it

The suggestion runs as a background request that reuses the parent conversation's prompt cache, so the additional cost is minimal. Claude Code skips suggestion generation when the cache is cold to avoid unnecessary cost.

Suggestions are automatically skipped after the first turn of a conversation, in non-interactive mode, and in plan mode.

To disable prompt suggestions entirely, set the environment variable or toggle the setting in `/config`:

```bash
export CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false
```

## Side questions with /btw

Use `/btw` to ask a quick question about your current work without adding to the conversation history. This is useful when you want a fast answer but don't want to clutter the main context or derail Claude from a long-running task.

```
/btw what was the name of that config file again?
```

Side questions have full visibility into the current conversation, so you can ask about code Claude has already read, decisions it made earlier, or anything else from the session. The question and answer are ephemeral: they appear in a dismissible overlay and never enter the conversation history.

* **Available while Claude is working**: you can run `/btw` even while Claude is processing a response. The side question runs independently and does not interrupt the main turn.
* **No tool access**: side questions answer only from what is already in context. Claude cannot read files, run commands, or search when answering a side question.
* **Single response**: there are no follow-up turns. If you need a back-and-forth, use a normal prompt instead.
* **Low cost**: the side question reuses the parent conversation's prompt cache, so the additional cost is minimal.

Press **Space**, **Enter**, or **Escape** to dismiss the answer and return to the prompt.

`/btw` is the inverse of a [subagent](/en/sub-agents): it sees your full conversation but has no tools, while a subagent has full tools but starts with an empty context. Use `/btw` to ask about what Claude already knows from this session; use a subagent to go find out something new.

## Task list

When working on complex, multi-step work, Claude creates a task list to track progress. Tasks appear in the status area of your terminal with indicators showing what's pending, in progress, or complete.

* Press `Ctrl+T` to toggle the task list view. The display shows up to 10 tasks at a time
* To see all tasks or clear them, ask Claude directly: "show me all tasks" or "clear all tasks"
* Tasks persist across context compactions, helping Claude stay organized on larger projects
* To share a task list across sessions, set `CLAUDE_CODE_TASK_LIST_ID` to use a named directory in `~/.claude/tasks/`: `CLAUDE_CODE_TASK_LIST_ID=my-project claude`

## PR review status

When working on a branch with an open pull request, Claude Code displays a clickable PR link in the footer (for example, "PR #446"). The link has a colored underline indicating the review state:

* Green: approved
* Yellow: pending review
* Red: changes requested
* Gray: draft
* Purple: merged

`Cmd+click` (Mac) or `Ctrl+click` (Windows/Linux) the link to open the pull request in your browser. The status updates automatically every 60 seconds.

PR status requires the `gh` CLI to be installed and authenticated (`gh auth login`).

## See also

* [Skills](/en/skills) - Custom prompts and workflows
* [Checkpointing](/en/checkpointing) - Rewind Claude's edits and restore previous states
* [CLI reference](/en/cli-reference) - Command-line flags and options
* [Settings](/en/settings) - Configuration options
* [Memory management](/en/memory) - Managing CLAUDE.md files
