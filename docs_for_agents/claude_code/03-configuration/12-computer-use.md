# Let Claude use your computer from the CLI

> Enable computer use in the Claude Code CLI so Claude can open apps, click, type, and see your screen on macOS. Test native apps, debug visual issues, and automate GUI-only tools without leaving your terminal.

<Note>
  Computer use is a research preview on macOS that requires a Pro or Max plan. It is not available on Team or Enterprise plans. It requires Claude Code v2.1.85 or later and an interactive session, so it is not available in non-interactive mode with the `-p` flag.
</Note>

Computer use lets Claude open apps, control your screen, and work on your machine the way you would. From the CLI, Claude can compile a Swift app, launch it, click through every button, and screenshot the result, all in the same conversation where it wrote the code.

This page covers how computer use works in the CLI. For the Desktop app on macOS or Windows, see [computer use in Desktop](/en/desktop#let-claude-use-your-computer).

## What you can do with computer use

Computer use handles tasks that require a GUI: anything you'd normally have to leave the terminal and do by hand.

* **Build and validate native apps**: ask Claude to build a macOS menu bar app. Claude writes the Swift, compiles it, launches it, and clicks through every control to verify it works before you ever open it.
* **End-to-end UI testing**: point Claude at a local Electron app and say "test the onboarding flow." Claude opens the app, clicks through signup, and screenshots each step. No Playwright config, no test harness.
* **Debug visual and layout issues**: tell Claude "the modal is clipping on small windows." Claude resizes the window, reproduces the bug, screenshots it, patches the CSS, and verifies the fix. Claude sees what you see.
* **Drive GUI-only tools**: interact with design tools, hardware control panels, the iOS Simulator, or proprietary apps that have no CLI or API.

## When computer use applies

Claude has several ways to interact with an app or service. Computer use is the broadest and slowest, so Claude tries the most precise tool first:

* If you have an [MCP server](/en/mcp) for the service, Claude uses that.
* If the task is a shell command, Claude uses Bash.
* If the task is browser work and you have [Claude in Chrome](/en/chrome) set up, Claude uses that.
* If none of those apply, Claude uses computer use.

Screen control is reserved for things nothing else can reach: native apps, simulators, and tools without an API.

## Enable computer use

Computer use is available as a built-in MCP server called `computer-use`. It's off by default until you enable it.

<Steps>
  <Step title="Open the MCP menu">
    In an interactive Claude Code session, run:

    ```text  theme={null}
    /mcp
    ```

    Find `computer-use` in the server list. It shows as disabled.
  </Step>

  <Step title="Enable the server">
    Select `computer-use` and choose **Enable**. The setting persists per project, so you only do this once for each project where you want computer use.
  </Step>

  <Step title="Grant macOS permissions">
    The first time Claude tries to use your computer, you'll see a prompt to grant two macOS permissions:

    * **Accessibility**: lets Claude click, type, and scroll
    * **Screen Recording**: lets Claude see what's on your screen

    The prompt includes links to open the relevant System Settings pane. Grant both, then select **Try again** in the prompt. macOS may require you to restart Claude Code after granting Screen Recording.
  </Step>
</Steps>

After setup, ask Claude to do something that needs the GUI:

```text  theme={null}
Build the app target, launch it, and click through each tab to make
sure nothing crashes. Screenshot any error states you find.
```

## Approve apps per session

Enabling the `computer-use` server doesn't grant Claude access to every app on your machine. The first time Claude needs a specific app in a session, a prompt appears in your terminal showing:

* Which apps Claude wants to control
* Any extra permissions requested, such as clipboard access
* How many other apps will be hidden while Claude works

Choose **Allow for this session** or **Deny**. Approvals last for the current session. You can approve multiple apps at once when Claude requests them together.

Apps with broad reach show an extra warning in the prompt so you know what approving them grants:

| Warning                    | Applies to                                                   |
| :------------------------- | :----------------------------------------------------------- |
| Equivalent to shell access | Terminal, iTerm, VS Code, Warp, and other terminals and IDEs |
| Can read or write any file | Finder                                                       |
| Can change system settings | System Settings                                              |

These apps aren't blocked. The warning lets you decide whether the task warrants that level of access.

Claude's level of control also varies by app category: browsers and trading platforms are view-only, terminals and IDEs are click-only, and everything else gets full control. See [app permissions in Desktop](/en/desktop#app-permissions) for the complete tier breakdown.

## How Claude works on your screen

Understanding the flow helps you anticipate what Claude will do and how to intervene.

### One session at a time

Computer use holds a machine-wide lock while active. If another Claude Code session is already using your computer, new attempts fail with a message telling you which session holds the lock. Finish or exit that session first.

### Apps are hidden while Claude works

When Claude starts controlling your screen, other visible apps are hidden so Claude interacts with only the approved apps. Your terminal window stays visible and is excluded from screenshots, so you can watch the session and Claude never sees its own output.

When Claude finishes the turn, hidden apps are restored automatically.

### Stop at any time

When Claude acquires the lock, a macOS notification appears: "Claude is using your computer · press Esc to stop." Press `Esc` anywhere to abort the current action immediately, or press `Ctrl+C` in the terminal. Either way, Claude releases the lock, unhides your apps, and returns control to you.

A second notification appears when Claude is done.

## Safety and the trust boundary

<Warning>
  Unlike the [sandboxed Bash tool](/en/sandboxing), computer use runs on your actual desktop with access to the apps you approve. Claude checks each action and flags potential prompt injection from on-screen content, but the trust boundary is different. See the [computer use safety guide](https://support.claude.com/en/articles/14128542) for best practices.
</Warning>

The built-in guardrails reduce risk without requiring configuration:

* **Per-app approval**: Claude can only control apps you've approved in the current session.
* **Sentinel warnings**: apps that grant shell, filesystem, or system settings access are flagged before you approve.
* **Terminal excluded from screenshots**: Claude never sees your terminal window, so on-screen prompts in your session can't feed back into the model.
* **Global escape**: the `Esc` key aborts computer use from anywhere, and the key press is consumed so prompt injection can't use it to dismiss dialogs.
* **Lock file**: only one session can control your machine at a time.

## Example workflows

These examples show common ways to combine computer use with coding tasks.

### Validate a native build

After making changes to a macOS or iOS app, have Claude compile and verify in one pass:

```text  theme={null}
Build the MenuBarStats target, launch it, open the preferences window,
and verify the interval slider updates the label. Screenshot the
preferences window when you're done.
```

Claude runs `xcodebuild`, launches the app, interacts with the UI, and reports what it finds.

### Reproduce a layout bug

When a visual bug only appears at certain window sizes, let Claude find it:

```text  theme={null}
The settings modal clips its footer on narrow windows. Resize the app
window down until you can reproduce it, screenshot the clipped state,
then check the CSS for the modal container.
```

Claude resizes the window, captures the broken state, and reads the relevant stylesheets.

### Test a simulator flow

Drive the iOS Simulator without writing XCTest:

```text  theme={null}
Open the iOS Simulator, launch the app, tap through the onboarding
screens, and tell me if any screen takes more than a second to load.
```

Claude controls the simulator the same way you would with a mouse.

## Differences from the Desktop app

The CLI and Desktop surfaces share the same computer use engine, with a few differences:

| Feature              | Desktop                                                  | CLI                             |
| :------------------- | :------------------------------------------------------- | :------------------------------ |
| Platforms            | macOS and Windows                                        | macOS only                      |
| Enable               | Toggle in **Settings > General** (under **Desktop app**) | Enable `computer-use` in `/mcp` |
| Denied apps list     | Configurable in Settings                                 | Not yet available               |
| Auto-unhide toggle   | Optional                                                 | Always on                       |
| Dispatch integration | Dispatch-spawned sessions can use computer use           | Not applicable                  |

## Troubleshooting

### "Computer use is in use by another Claude session"

Another Claude Code session holds the lock. Finish the task in that session or exit it. If the other session crashed, the lock is released automatically when Claude detects the process is no longer running.

### macOS permissions prompt keeps reappearing

macOS sometimes requires a restart of the requesting process after you grant Screen Recording. Quit Claude Code completely and start a new session. If the prompt persists, open **System Settings > Privacy & Security > Screen Recording** and confirm your terminal app is listed and enabled.

### `computer-use` doesn't appear in `/mcp`

The server only appears on eligible setups. Check that:

* You're on macOS. Computer use in the CLI is not available on Linux or Windows. On Windows, use [computer use in Desktop](/en/desktop#let-claude-use-your-computer) instead.
* You're running Claude Code v2.1.85 or later. Run `claude --version` to check.
* You're on a Pro or Max plan. Run `/status` to confirm your subscription.
* You're authenticated through claude.ai. Computer use is not available with third-party providers like Amazon Bedrock, Google Cloud Vertex AI, or Microsoft Foundry. If you access Claude exclusively through a third-party provider, you need a separate claude.ai account to use this feature.
* You're in an interactive session. Computer use is not available in non-interactive mode with the `-p` flag.

## See also

* [Computer use in Desktop](/en/desktop#let-claude-use-your-computer): the same capability with a graphical settings page
* [Claude in Chrome](/en/chrome): browser automation for web-based tasks
* [MCP](/en/mcp): connect Claude to structured tools and APIs
* [Sandboxing](/en/sandboxing): how Claude's Bash tool isolates filesystem and network access
* [Computer use safety guide](https://support.claude.com/en/articles/14128542): best practices for safe computer use
