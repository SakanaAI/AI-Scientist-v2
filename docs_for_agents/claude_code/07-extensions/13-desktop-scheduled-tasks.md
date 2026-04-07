# Schedule recurring tasks in Claude Code Desktop

> Set up scheduled tasks in Claude Code Desktop to run Claude automatically on a recurring basis for daily code reviews, dependency audits, or morning briefings.

By default, scheduled tasks start a new session automatically at a time and frequency you choose. Use them for recurring work like daily code reviews, dependency update checks, or morning briefings.

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

The Schedule page supports two kinds of tasks:

* **Local tasks**: run on your machine. They have direct access to your local files and tools.
* **Remote tasks**: run on Anthropic-managed cloud infrastructure.

Both kinds appear in the same task grid. Click **New task** to pick which kind to create. This page covers local tasks; for remote tasks, see [Cloud scheduled tasks](/en/web-scheduled-tasks).

## Create a scheduled task

To create a local scheduled task, click **Schedule** in the sidebar, click **New task**, and choose **New local task**. Configure:

| Field       | Description                                                                                                                         |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Name        | Identifier for the task. Converted to lowercase kebab-case and used as the folder name on disk. Must be unique.                    |
| Description | Short summary shown in the task list.                                                                                              |
| Prompt      | The instructions sent to Claude when the task runs. Write this the same way you'd write any message in the prompt box.             |
| Frequency   | How often the task runs. See [frequency options](#frequency-options) below.                                                        |

## Frequency options

Pick a preset from the frequency dropdown:

* **Manual**: no schedule, only runs when you click **Run now**
* **Hourly**: runs every hour
* **Daily**: shows a time picker, defaults to 9:00 AM local time
* **Weekdays**: same as Daily but skips Saturday and Sunday
* **Weekly**: shows a time picker and a day picker

For custom intervals, ask Claude in any Desktop session, for example: "schedule a task to run all the tests every 6 hours."

## How scheduled tasks run

Local scheduled tasks run on your machine. Desktop checks the schedule every minute while the app is open and starts a fresh session when a task is due.

When a task fires, you get a desktop notification and a new session appears under a **Scheduled** section in the sidebar. Open it to see what Claude did, review changes, or respond to permission prompts.

Tasks only run while the desktop app is running and your computer is awake. To prevent idle-sleep, enable **Keep computer awake** in Settings under **Desktop app → General**. For tasks that need to run even when your computer is off, use a [remote task](/en/web-scheduled-tasks) instead.

## Missed runs

When the app starts or your computer wakes, Desktop checks whether each task missed any runs in the last seven days. If it did, Desktop starts exactly one catch-up run for the most recently missed time.

Keep this in mind when writing prompts. A task scheduled for 9am might run at 11pm if your computer was asleep all day. If timing matters, add guardrails to the prompt itself.

## Permissions for scheduled tasks

Each task has its own permission mode, which you set when creating or editing the task. To avoid stalls, click **Run now** after creating a task, watch for permission prompts, and select "always allow" for tools the task needs.

You can review and revoke these approvals from the task's detail page.

## Manage scheduled tasks

Click a task in the **Schedule** list to open its detail page. From here you can:

* **Run now**: start the task immediately
* **Toggle repeats**: pause or resume scheduled runs
* **Edit**: change the prompt, frequency, folder, or other settings
* **Review history**: see every past run
* **Review allowed permissions**: see and revoke saved tool approvals
* **Delete**: remove the task and archive all sessions it created

You can also manage tasks by asking Claude in any Desktop session: "pause my dependency-audit task" or "show me my scheduled tasks."

## Related resources

* [Cloud scheduled tasks](/en/web-scheduled-tasks): schedule tasks that run on Anthropic-managed infrastructure even when your computer is off
* [Run prompts on a schedule](/en/scheduled-tasks): session-scoped scheduling with `/loop`
* [Claude Code GitHub Actions](/en/github-actions): run Claude on a schedule in CI
* [Use Claude Code Desktop](/en/desktop): the full Desktop app guide
