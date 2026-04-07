# Schedule tasks on the web

> Automate recurring work with cloud scheduled tasks

A scheduled task runs a prompt on a recurring cadence using Anthropic-managed infrastructure. Tasks keep working even when your computer is off.

A few examples of recurring work you can automate:

* Reviewing open pull requests each morning
* Analyzing CI failures overnight and surfacing summaries
* Syncing documentation after PRs merge
* Running dependency audits every week

Scheduled tasks are available to all Claude Code on the web users, including Pro, Max, Team, and Enterprise.

## Compare scheduling options

Claude Code offers three ways to schedule recurring work:

|                            | [Cloud](/en/web-scheduled-tasks) | [Desktop](/en/desktop-scheduled-tasks) | [`/loop`](/en/scheduled-tasks) |
| :------------------------- | :------------------------------- | :------------------------------------- | :----------------------------- |
| Runs on                    | Anthropic cloud                  | Your machine                           | Your machine                   |
| Requires machine on        | No                               | Yes                                    | Yes                            |
| Requires open session      | No                               | No                                     | Yes                            |
| Persistent across restarts | Yes                              | Yes                                    | No (session-scoped)            |
| Access to local files      | No (fresh clone)                 | Yes                                    | Yes                            |
| Minimum interval           | 1 hour                           | 1 minute                               | 1 minute                       |

**Tip**: Use **cloud tasks** for work that should run reliably without your machine. Use **Desktop tasks** when you need access to local files and tools. Use **`/loop`** for quick polling during a session.

## Create a scheduled task

You can create a scheduled task from three places:

* **Web**: visit [claude.ai/code/scheduled](https://claude.ai/code/scheduled) and click **New scheduled task**
* **Desktop app**: open the **Schedule** page, click **New task**, and choose **New remote task**
* **CLI**: run `/schedule` in any session. Claude walks you through the setup conversationally.

## Steps to create a scheduled task

**Step 1: Open the creation form**

Visit [claude.ai/code/scheduled](https://claude.ai/code/scheduled) and click **New scheduled task**.

**Step 2: Name the task and write the prompt**

Give the task a descriptive name and write the prompt Claude runs each time. The prompt must be self-contained and explicit about what to do and what success looks like.

**Step 3: Select repositories**

Add one or more GitHub repositories for Claude to work in. Each repository is cloned at the start of a run, starting from the default branch.

**Step 4: Select an environment**

Select a cloud environment for the task. Environments control:

* **Network access**: level of internet access available during each run
* **Environment variables**: API keys, tokens, or other secrets
* **Setup script**: install commands before each session starts

A **Default** environment is available out of the box.

**Step 5: Choose a schedule**

Pick how often the task runs from the frequency options. The default is daily at 9:00 AM in your local time zone.

Frequency options:

| Frequency | Description                                              |
| :-------- | :------------------------------------------------------- |
| Hourly    | Runs every hour.                                         |
| Daily     | Runs once per day at the time you specify.               |
| Weekdays  | Same as Daily but skips Saturday and Sunday.             |
| Weekly    | Runs once per week on the day and time you specify.      |

**Step 6: Review connectors**

All of your connected MCP connectors are included by default. Remove any that the task doesn't need.

**Step 7: Create the task**

Click **Create**. The task appears in the scheduled tasks list and runs automatically at the next scheduled time.

## Manage scheduled tasks

Click a task in the **Scheduled** list to open its detail page. From there you can:

* Click **Run now** to start a run immediately
* Use the toggle in the **Repeats** section to pause or resume the schedule
* Click the edit icon to change the name, prompt, schedule, repositories, or connectors
* Click the delete icon to remove the task

You can also manage tasks from the CLI with `/schedule`:

* `/schedule list` - see all tasks
* `/schedule update` - change a task
* `/schedule run` - trigger one immediately

## View and interact with runs

Click any run to open it as a full session. Each run session works like any other: use the dropdown menu next to the session title to rename, archive, or delete it.

## Related resources

* [Desktop scheduled tasks](/en/desktop-scheduled-tasks): schedule tasks that run on your machine with access to local files
* [`/loop` and CLI scheduled tasks](/en/scheduled-tasks): lightweight scheduling within a CLI session
* [Cloud environment](/en/claude-code-on-the-web#cloud-environment): configure the runtime environment for cloud tasks
* [MCP connectors](/en/mcp): connect external services like Slack, Linear, and Google Drive
