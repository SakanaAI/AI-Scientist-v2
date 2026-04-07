# Run prompts on a schedule

> Use /loop and the cron scheduling tools to run prompts repeatedly, poll for status, or set one-time reminders within a Claude Code session.

**Note**: Scheduled tasks require Claude Code v2.1.72 or later. Check your version with `claude --version`.

Scheduled tasks let Claude re-run a prompt automatically on an interval. Use them to poll a deployment, babysit a PR, check back on a long-running build, or remind yourself to do something later in the session. To react to events as they happen instead of polling, see [Channels](/en/channels): your CI can push the failure into the session directly.

Tasks are session-scoped: they live in the current Claude Code process and are gone when you exit. For durable scheduling that survives restarts, use [Cloud](/en/web-scheduled-tasks) or [Desktop](/en/desktop-scheduled-tasks) scheduled tasks, or [GitHub Actions](/en/github-actions).

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

## Schedule a recurring prompt with /loop

The `/loop` skill is the quickest way to schedule a recurring prompt. Pass an optional interval and a prompt:

```text
/loop 5m check if the deployment finished and tell me what happened
```

Claude parses the interval, converts it to a cron expression, schedules the job, and confirms the cadence.

### Interval syntax

Intervals are optional. Supported units are `s`, `m`, `h`, `d`.

| Form                    | Example                               | Parsed interval              |
| :---------------------- | :------------------------------------ | :--------------------------- |
| Leading token           | `/loop 30m check the build`           | every 30 minutes             |
| Trailing `every` clause | `/loop check the build every 2 hours` | every 2 hours                |
| No interval             | `/loop check the build`               | defaults to every 10 minutes |

## Set a one-time reminder

For one-shot reminders, describe what you want in natural language:

```text
remind me at 3pm to push the release branch
```

```text
in 45 minutes, check whether the integration tests passed
```

Claude schedules a single-fire task that deletes itself after running.

## Manage scheduled tasks

Ask Claude in natural language to list or cancel tasks:

```text
what scheduled tasks do I have?
```

```text
cancel the deploy check job
```

## How scheduled tasks run

The scheduler checks every second for due tasks and enqueues them at low priority. A scheduled prompt fires between your turns, not while Claude is mid-response.

All times are interpreted in your local timezone. A cron expression like `0 9 * * *` means 9am wherever you're running Claude Code.

### Seven-day expiry

Recurring tasks automatically expire 7 days after creation. If you need a recurring task to last longer, cancel and recreate it before it expires, or use [Cloud scheduled tasks](/en/web-scheduled-tasks) or [Desktop scheduled tasks](/en/desktop-scheduled-tasks).

## Cron expression reference

`CronCreate` accepts standard 5-field cron expressions: `minute hour day-of-month month day-of-week`.

| Example        | Meaning                      |
| :------------- | :--------------------------- |
| `*/5 * * * *`  | Every 5 minutes              |
| `0 * * * *`    | Every hour on the hour       |
| `0 9 * * *`    | Every day at 9am local       |
| `0 9 * * 1-5`  | Weekdays at 9am local        |

## Limitations

Session-scoped scheduling has inherent constraints:

* Tasks only fire while Claude Code is running and idle
* No catch-up for missed fires
* No persistence across restarts

For cron-driven automation that needs to run unattended:

* [Cloud scheduled tasks](/en/web-scheduled-tasks): run on Anthropic infrastructure
* [GitHub Actions](/en/github-actions): use a `schedule` trigger in CI
* [Desktop scheduled tasks](/en/desktop-scheduled-tasks): run locally on your machine
