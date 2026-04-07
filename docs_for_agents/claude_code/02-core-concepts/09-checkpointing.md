# Checkpointing

Track, rewind, and summarize Claude's edits and conversation to manage session state.

Claude Code automatically tracks Claude's file edits as you work, allowing you to quickly undo changes and rewind to previous states if anything gets off track.

## How checkpoints work

As you work with Claude, checkpointing automatically captures the state of your code before each edit. This safety net lets you pursue ambitious, wide-scale tasks knowing you can always return to a prior code state.

### Automatic tracking

Claude Code tracks all changes made by its file editing tools:

* Every user prompt creates a new checkpoint
* Checkpoints persist across sessions, so you can access them in resumed conversations
* Automatically cleaned up along with sessions after 30 days (configurable)

### Rewind and summarize

Press `Esc` twice (`Esc` + `Esc`) or use the `/rewind` command to open the rewind menu. A scrollable list shows each of your prompts from the session. Select the point you want to act on, then choose an action:

* **Restore code and conversation**: revert both code and conversation to that point
* **Restore conversation**: rewind to that message while keeping current code
* **Restore code**: revert file changes while keeping the conversation
* **Summarize from here**: compress the conversation from this point forward into a summary, freeing context window space
* **Never mind**: return to the message list without making changes

After restoring the conversation or summarizing, the original prompt from the selected message is restored into the input field so you can re-send or edit it.

#### Restore vs. summarize

The three restore options revert state: they undo code changes, conversation history, or both. "Summarize from here" works differently:

* Messages before the selected message stay intact
* The selected message and all subsequent messages get replaced with a compact AI-generated summary
* No files on disk are changed
* The original messages are preserved in the session transcript, so Claude can reference the details if needed

This is similar to `/compact`, but targeted: instead of summarizing the entire conversation, you keep early context in full detail and only compress the parts that are using up space. You can type optional instructions to guide what the summary focuses on.

Summarize keeps you in the same session and compresses context. If you want to branch off and try a different approach while preserving the original session intact, use [fork](/en/how-claude-code-works#resume-or-fork-sessions) instead (`claude --continue --fork-session`).

## Common use cases

Checkpoints are particularly useful when:

* **Exploring alternatives**: try different implementation approaches without losing your starting point
* **Recovering from mistakes**: quickly undo changes that introduced bugs or broke functionality
* **Iterating on features**: experiment with variations knowing you can revert to working states
* **Freeing context space**: summarize a verbose debugging session from the midpoint forward, keeping your initial instructions intact

## Limitations

### Bash command changes not tracked

Checkpointing does not track files modified by bash commands. For example, if Claude Code runs:

```bash
rm file.txt
mv old.txt new.txt
cp source.txt dest.txt
```

These file modifications cannot be undone through rewind. Only direct file edits made through Claude's file editing tools are tracked.

### External changes not tracked

Checkpointing only tracks files that have been edited within the current session. Manual changes you make to files outside of Claude Code and edits from other concurrent sessions are normally not captured, unless they happen to modify the same files as the current session.

### Not a replacement for version control

Checkpoints are designed for quick, session-level recovery. For permanent version history and collaboration:

* Continue using version control (ex. Git) for commits, branches, and long-term history
* Checkpoints complement but don't replace proper version control
* Think of checkpoints as "local undo" and Git as "permanent history"

## See also

* [Interactive mode](/en/interactive-mode) - Keyboard shortcuts and session controls
* [Built-in commands](/en/commands) - Accessing checkpoints using `/rewind`
* [CLI reference](/en/cli-reference) - Command-line options
