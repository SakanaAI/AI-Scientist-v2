# Plan in the cloud with ultraplan

> Start a plan from your CLI, draft it on Claude Code on the web, then execute it remotely or back in your terminal

**Note**: Ultraplan is in research preview. Behavior and capabilities may change based on feedback.

Ultraplan hands a planning task from your local CLI to a Claude Code on the web session running in plan mode. Claude drafts the plan in the cloud while you keep working in your terminal. When the plan is ready, you open it in your browser to comment on specific sections, ask for revisions, and choose where to execute it.

This is useful when you want a richer review surface than the terminal offers:

* **Targeted feedback**: comment on individual sections of the plan instead of replying to the whole thing
* **Hands-off drafting**: the plan is generated remotely, so your terminal stays free for other work
* **Flexible execution**: approve the plan to run on the web and open a pull request, or send it back to your terminal

Ultraplan requires a Claude Code on the web account and a GitHub repository.

## Launch ultraplan from the CLI

From your local CLI session, you can launch ultraplan in three ways:

* **Command**: run `/ultraplan` followed by your prompt
* **Keyword**: include the word `ultraplan` anywhere in a normal prompt
* **From a local plan**: when Claude finishes a local plan and shows the approval dialog, choose **No, refine with Ultraplan on Claude Code on the web** to send the draft to the cloud for further iteration

For example, to plan a service migration:

```
/ultraplan migrate the auth service from sessions to JWTs
```

The command and keyword paths open a confirmation dialog before launching. The local plan path skips this dialog because that selection already serves as confirmation.

After the cloud session launches, your CLI's prompt input shows a status indicator while the remote session works:

| Status                         | Meaning                                                            |
| :----------------------------- | :----------------------------------------------------------------- |
| `◇ ultraplan`                  | Claude is researching your codebase and drafting the plan          |
| `◇ ultraplan needs your input` | Claude has a clarifying question; open the session link to respond |
| `◆ ultraplan ready`            | The plan is ready to review in your browser                        |

Run `/tasks` and select the ultraplan entry to open a detail view with the session link, agent activity, and a **Stop ultraplan** action.

## Review and revise the plan in your browser

When the status changes to `◆ ultraplan ready`, open the session link to view the plan on claude.ai. The plan appears in a dedicated review view:

* **Inline comments**: highlight any passage and leave a comment for Claude to address
* **Emoji reactions**: react to a section to signal approval or concern
* **Outline sidebar**: jump between sections of the plan

When you ask Claude to address your comments, it revises the plan and presents an updated draft. You can iterate as many times as needed before choosing where to execute.

## Choose where to execute

When the plan looks right, you choose from the browser whether Claude implements it in the same cloud session or sends it back to your waiting terminal.

### Execute on the web

Select **Approve Claude's plan and start coding** in your browser to have Claude implement it in the same session. When the implementation finishes, review the diff and create a pull request from the web interface.

### Send the plan back to your terminal

Select **Approve plan and teleport back to terminal** in your browser to implement the plan locally with full access to your environment. The web session is archived so it doesn't continue working in parallel.

Your terminal shows the plan in a dialog titled **Ultraplan approved** with three options:

* **Implement here**: inject the plan into your current conversation
* **Start new session**: clear the current conversation and begin fresh with only the plan
* **Cancel**: save the plan to a file without executing it

## Related resources

* [Claude Code on the web](/en/claude-code-on-the-web): the cloud infrastructure ultraplan runs on
* [Plan mode](/en/permission-modes#analyze-before-you-edit-with-plan-mode): how planning works in a local session
