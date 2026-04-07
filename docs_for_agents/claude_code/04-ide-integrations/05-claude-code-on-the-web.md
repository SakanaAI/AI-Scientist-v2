# Claude Code on the web

Run Claude Code tasks asynchronously on secure cloud infrastructure

> Claude Code on the web is currently in research preview.

## What is Claude Code on the web?

Claude Code on the web lets developers kick off Claude Code from the Claude app. This is perfect for:

* **Answering questions**: Ask about code architecture and how features are implemented
* **Bug fixes and routine tasks**: Well-defined tasks that don't require frequent steering
* **Parallel work**: Tackle multiple bug fixes in parallel
* **Repositories not on your local machine**: Work on code you don't have checked out locally
* **Backend changes**: Where Claude Code can write tests and then write code to pass those tests

Claude Code is also available on the Claude app for [iOS](https://apps.apple.com/us/app/claude-by-anthropic/id6473753684) and [Android](https://play.google.com/store/apps/details?id=com.anthropic.claude) for kicking off tasks on the go and monitoring work in progress.

You can [kick off new tasks on the web from your terminal](#from-terminal-to-web) with `--remote`, or [teleport web sessions back to your terminal](#from-web-to-terminal) to continue locally. To use the web interface while running Claude Code on your own machine instead of cloud infrastructure, see [Remote Control](/en/remote-control).

## Who can use Claude Code on the web?

Claude Code on the web is available in research preview to:

* **Pro users**
* **Max users**
* **Team users**
* **Enterprise users** with premium seats or Chat + Claude Code seats

## Getting started

Set up Claude Code on the web from the browser or from your terminal.

### From the browser

1. Visit [claude.ai/code](https://claude.ai/code)
2. Connect your GitHub account
3. Install the Claude GitHub App in your repositories
4. Select your default environment
5. Submit your coding task
6. Review changes in diff view, iterate with comments, then create a pull request

### From the terminal

Run `/web-setup` inside Claude Code to connect GitHub using your local `gh` CLI credentials. The command syncs your `gh auth token` to Claude Code on the web, creates a default cloud environment, and opens claude.ai/code in your browser when it finishes.

This path requires the `gh` CLI to be installed and authenticated with `gh auth login`. If `gh` is not available, `/web-setup` opens claude.ai/code so you can connect GitHub from the browser instead.

Your `gh` credentials give Claude access to clone and push, so you can skip the GitHub App for basic sessions. Install the App later if you want [Auto-fix](#auto-fix-pull-requests), which uses the App to receive PR webhooks.

> Team and Enterprise admins can disable terminal setup with the Quick web setup toggle at [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code).

## How it works

When you start a task on Claude Code on the web:

1. **Repository cloning**: Your repository is cloned to an Anthropic-managed virtual machine
2. **Environment setup**: Claude prepares a secure cloud environment with your code, then runs your [setup script](#setup-scripts) if configured
3. **Network configuration**: Internet access is configured based on your settings
4. **Task execution**: Claude analyzes code, makes changes, runs tests, and checks its work
5. **Completion**: You're notified when finished and can create a PR with the changes
6. **Results**: Changes are pushed to a branch, ready for pull request creation

## Review changes with diff view

Diff view lets you see exactly what Claude changed before creating a pull request. Instead of clicking "Create PR" to review changes in GitHub, view the diff directly in the app and iterate with Claude until the changes are ready.

When Claude makes changes to files, a diff stats indicator appears showing the number of lines added and removed (for example, `+12 -1`). Select this indicator to open the diff viewer, which displays a file list on the left and the changes for each file on the right.

From the diff view, you can:

* Review changes file by file
* Comment on specific changes to request modifications
* Continue iterating with Claude based on what you see

This lets you refine changes through multiple rounds of feedback without creating draft PRs or switching to GitHub.

## Auto-fix pull requests

Claude can watch a pull request and automatically respond to CI failures and review comments. Claude subscribes to GitHub activity on the PR, and when a check fails or a reviewer leaves a comment, Claude investigates and pushes a fix if one is clear.

> Auto-fix requires the Claude GitHub App to be installed on your repository. If you haven't already, install it from the [GitHub App page](https://github.com/apps/claude) or when prompted during [setup](#getting-started).

There are a few ways to turn on auto-fix depending on where the PR came from and what device you're using:

* **PRs created in Claude Code on the web**: open the CI status bar and select **Auto-fix**
* **From the mobile app**: tell Claude to auto-fix the PR, for example "watch this PR and fix any CI failures or review comments"
* **Any existing PR**: paste the PR URL into a session and tell Claude to auto-fix it

### How Claude responds to PR activity

When auto-fix is active, Claude receives GitHub events for the PR including new review comments and CI check failures. For each event, Claude investigates and decides how to proceed:

* **Clear fixes**: if Claude is confident in a fix and it doesn't conflict with earlier instructions, Claude makes the change, pushes it, and explains what was done in the session
* **Ambiguous requests**: if a reviewer's comment could be interpreted multiple ways or involves something architecturally significant, Claude asks you before acting
* **Duplicate or no-action events**: if an event is a duplicate or requires no change, Claude notes it in the session and moves on

Claude may reply to review comment threads on GitHub as part of resolving them. These replies are posted using your GitHub account, so they appear under your username, but each reply is labeled as coming from Claude Code so reviewers know it was written by the agent and not by you directly.

> If your repository uses comment-triggered automation such as Atlantis, Terraform Cloud, or custom GitHub Actions that run on `issue_comment` events, be aware that Claude can reply on your behalf, which can trigger those workflows. Review your repository's automation before enabling auto-fix, and consider disabling auto-fix for repositories where a PR comment can deploy infrastructure or run privileged operations.

## Moving tasks between web and terminal

You can start new tasks on the web from your terminal, or pull web sessions into your terminal to continue locally. Web sessions persist even if you close your laptop, and you can monitor them from anywhere including the Claude mobile app.

> Session handoff is one-way: you can pull web sessions into your terminal, but you can't push an existing terminal session to the web. The `--remote` flag creates a *new* web session for your current repository.

### From terminal to web

Start a web session from the command line with the `--remote` flag:

```bash
claude --remote "Fix the authentication bug in src/auth/login.ts"
```

This creates a new web session on claude.ai. The task runs in the cloud while you continue working locally. Use `/tasks` to check progress, or open the session on claude.ai or the Claude mobile app to interact directly. From there you can steer Claude, provide feedback, or answer questions just like any other conversation.

#### Tips for remote tasks

**Plan locally, execute remotely**: For complex tasks, start Claude in plan mode to collaborate on the approach, then send work to the web:

```bash
claude --permission-mode plan
```

In plan mode, Claude can only read files and explore the codebase. Once you're satisfied with the plan, start a remote session for autonomous execution:

```bash
claude --remote "Execute the migration plan in docs/migration-plan.md"
```

This pattern gives you control over the strategy while letting Claude execute autonomously in the cloud.

**Plan in the cloud with ultraplan**: To draft and review the plan itself in a web session, use [ultraplan](/en/ultraplan). Claude generates the plan on Claude Code on the web while you keep working, then you comment on sections in your browser and choose to execute remotely or send the plan back to your terminal.

**Run tasks in parallel**: Each `--remote` command creates its own web session that runs independently. You can kick off multiple tasks and they'll all run simultaneously in separate sessions:

```bash
claude --remote "Fix the flaky test in auth.spec.ts"
claude --remote "Update the API documentation"
claude --remote "Refactor the logger to use structured output"
```

Monitor all sessions with `/tasks`. When a session completes, you can create a PR from the web interface or [teleport](#from-web-to-terminal) the session to your terminal to continue working.

### From web to terminal

There are several ways to pull a web session into your terminal:

* **Using `/teleport`**: From within Claude Code, run `/teleport` (or `/tp`) to see an interactive picker of your web sessions. If you have uncommitted changes, you'll be prompted to stash them first.
* **Using `--teleport`**: From the command line, run `claude --teleport` for an interactive session picker, or `claude --teleport <session-id>` to resume a specific session directly.
* **From `/tasks`**: Run `/tasks` to see your background sessions, then press `t` to teleport into one
* **From the web interface**: Click "Open in CLI" to copy a command you can paste into your terminal

When you teleport a session, Claude verifies you're in the correct repository, fetches and checks out the branch from the remote session, and loads the full conversation history into your terminal.

#### Requirements for teleporting

Teleport checks these requirements before resuming a session. If any requirement isn't met, you'll see an error or be prompted to resolve the issue.

| Requirement        | Details                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Clean git state    | Your working directory must have no uncommitted changes. Teleport prompts you to stash changes if needed.              |
| Correct repository | You must run `--teleport` from a checkout of the same repository, not a fork.                                          |
| Branch available   | The branch from the web session must have been pushed to the remote. Teleport automatically fetches and checks it out. |
| Same account       | You must be authenticated to the same Claude.ai account used in the web session.                                       |

### Sharing sessions

To share a session, toggle its visibility according to the account
types below. After that, share the session link as-is. Recipients who open your
shared session will see the latest state of the session upon load, but the
recipient's page will not update in real time.

#### Sharing from an Enterprise or Team account

For Enterprise and Team accounts, the two visibility options are **Private**
and **Team**. Team visibility makes the session visible to other members of your
Claude.ai organization. Repository access verification is enabled by default,
based on the GitHub account connected to the recipient's account. Your account's
display name is visible to all recipients with access. [Claude in Slack](/en/slack)
sessions are automatically shared with Team visibility.

#### Sharing from a Max or Pro account

For Max and Pro accounts, the two visibility options are **Private**
and **Public**. Public visibility makes the session visible to any user logged
into claude.ai.

Check your session for sensitive content before sharing. Sessions may contain
code and credentials from private GitHub repositories. Repository access
verification is not enabled by default.

Enable repository access verification and/or withhold your name from your shared
sessions by going to Settings > Claude Code > Sharing settings.

## Schedule recurring tasks

Run Claude on a recurring schedule to automate work like daily PR reviews, dependency audits, and CI failure analysis. See [Schedule tasks on the web](/en/web-scheduled-tasks) for the full guide.

## Managing sessions

### Archiving sessions

You can archive sessions to keep your session list organized. Archived sessions are hidden from the default session list but can be viewed by filtering for archived sessions.

To archive a session, hover over the session in the sidebar and click the archive icon.

### Deleting sessions

Deleting a session permanently removes the session and its data. This action cannot be undone. You can delete a session in two ways:

* **From the sidebar**: Filter for archived sessions, then hover over the session you want to delete and click the delete icon
* **From the session menu**: Open a session, click the dropdown next to the session title, and select **Delete**

You will be asked to confirm before a session is deleted.

## Cloud environment

### Default image

We build and maintain a universal image with common toolchains and language ecosystems pre-installed. This image includes:

* Popular programming languages and runtimes
* Common build tools and package managers
* Testing frameworks and linters

#### Checking available tools

To see what's pre-installed in your environment, ask Claude Code to run:

```bash
check-tools
```

This command displays:

* Programming languages and their versions
* Available package managers
* Installed development tools

#### Language-specific setups

The universal image includes pre-configured environments for:

* **Python**: Python 3.x with pip, poetry, and common scientific libraries
* **Node.js**: Latest LTS versions with npm, yarn, pnpm, and bun
* **Ruby**: Versions 3.1.6, 3.2.6, 3.3.6 (default: 3.3.6) with gem, bundler, and rbenv for version management
* **PHP**: Version 8.4.14
* **Java**: OpenJDK with Maven and Gradle
* **Go**: Latest stable version with module support
* **Rust**: Rust toolchain with cargo
* **C++**: GCC and Clang compilers

#### Databases

The universal image includes the following databases:

* **PostgreSQL**: Version 16
* **Redis**: Version 7.0

### Environment configuration

When you start a session in Claude Code on the web, here's what happens under the hood:

1. **Environment preparation**: We clone your repository and run any configured [setup script](#setup-scripts). The repo will be cloned with the default branch on your GitHub repo. If you would like to check out a specific branch, you can specify that in the prompt.

2. **Network configuration**: We configure internet access for the agent. Internet access is limited by default, but you can configure the environment to have no internet or full internet access based on your needs.

3. **Claude Code execution**: Claude Code runs to complete your task, writing code, running tests, and checking its work. You can guide and steer Claude throughout the session via the web interface. Claude respects context you've defined in your `CLAUDE.md`.

4. **Outcome**: When Claude completes its work, it will push the branch to remote. You will be able to create a PR for the branch.

> Claude operates entirely through the terminal and CLI tools available in the environment. It uses the pre-installed tools in the universal image and any additional tools you install through hooks or dependency management.

**To add a new environment:** Select the current environment to open the environment selector, and then select "Add environment". This will open a dialog where you can specify the environment name, network access level, environment variables, and a [setup script](#setup-scripts).

**To update an existing environment:** Select the current environment, to the right of the environment name, and select the settings button. This will open a dialog where you can update the environment name, network access, environment variables, and setup script.

**To select your default environment from the terminal:** If you have multiple environments configured, run `/remote-env` to choose which one to use when starting web sessions from your terminal with `--remote`. With a single environment, this command shows your current configuration.

> Environment variables must be specified as key-value pairs, in [`.env` format](https://www.dotenv.org/). For example:
>
> ```
> API_KEY=your_api_key
> DEBUG=true
> ```

### Setup scripts

A setup script is a Bash script that runs when a new cloud session starts, before Claude Code launches. Use setup scripts to install dependencies, configure tools, or prepare anything the cloud environment needs that isn't in the [default image](#default-image).

Scripts run as root on Ubuntu 24.04, so `apt install` and most language package managers work.

> To check what's already installed before adding it to your script, ask Claude to run `check-tools` in a cloud session.

To add a setup script, open the environment settings dialog and enter your script in the **Setup script** field.

This example installs the `gh` CLI, which isn't in the default image:

```bash
#!/bin/bash
apt update && apt install -y gh
```

Setup scripts run only when creating a new session. They are skipped when resuming an existing session.

If the script exits non-zero, the session fails to start. Append `|| true` to non-critical commands to avoid blocking the session on a flaky install.

> Setup scripts that install packages need network access to reach registries. The default network access allows connections to [common package registries](#default-allowed-domains) including npm, PyPI, RubyGems, and crates.io. Scripts will fail to install packages if your environment has network access disabled.

#### Setup scripts vs. SessionStart hooks

Use a setup script to install things the cloud needs but your laptop already has, like a language runtime or CLI tool. Use a [SessionStart hook](/en/hooks#sessionstart) for project setup that should run everywhere, cloud and local, like `npm install`.

Both run at the start of a session, but they belong to different places:

|               | Setup scripts                                     | SessionStart hooks                                             |
| ------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| Attached to   | The cloud environment                             | Your repository                                                |
| Configured in | Cloud environment UI                              | `.claude/settings.json` in your repo                           |
| Runs          | Before Claude Code launches, on new sessions only | After Claude Code launches, on every session including resumed |
| Scope         | Cloud environments only                           | Both local and cloud                                           |

SessionStart hooks can also be defined in your user-level `~/.claude/settings.json` locally, but user-level settings don't carry over to cloud sessions. In the cloud, only hooks committed to the repo run.

### Dependency management

Custom environment images and snapshots are not yet supported. Use [setup scripts](#setup-scripts) to install packages when a session starts, or [SessionStart hooks](/en/hooks#sessionstart) for dependency installation that should also run in local environments. SessionStart hooks have [known limitations](#dependency-management-limitations).

To configure automatic dependency installation with a setup script, open your environment settings and add a script:

```bash
#!/bin/bash
npm install
pip install -r requirements.txt
```

Alternatively, you can use SessionStart hooks in your repository's `.claude/settings.json` file for dependency installation that should also run in local environments:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/install_pkgs.sh"
          }
        ]
      }
    ]
  }
}
```

Create the corresponding script at `scripts/install_pkgs.sh`:

```bash
#!/bin/bash

# Only run in remote environments
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

npm install
pip install -r requirements.txt
exit 0
```

Make it executable: `chmod +x scripts/install_pkgs.sh`

#### Persist environment variables

SessionStart hooks can persist environment variables for subsequent Bash commands by writing to the file specified in the `CLAUDE_ENV_FILE` environment variable. For details, see [SessionStart hooks](/en/hooks#sessionstart) in the hooks reference.

#### Dependency management limitations

* **Hooks fire for all sessions**: SessionStart hooks run in both local and remote environments. There is no hook configuration to scope a hook to remote sessions only. To skip local execution, check the `CLAUDE_CODE_REMOTE` environment variable in your script as shown above.
* **Requires network access**: Install commands need network access to reach package registries. If your environment is configured with "No internet" access, these hooks will fail. Use "Limited" (the default) or "Full" network access. The [default allowlist](#default-allowed-domains) includes common registries like npm, PyPI, RubyGems, and crates.io.
* **Proxy compatibility**: All outbound traffic in remote environments passes through a [security proxy](#security-proxy). Some package managers do not work correctly with this proxy. Bun is a known example.
* **Runs on every session start**: Hooks run each time a session starts or resumes, adding startup latency. Keep install scripts fast by checking whether dependencies are already present before reinstalling.

## Network access and security

### Network policy

#### GitHub proxy

For security, all GitHub operations go through a dedicated proxy service that transparently handles all git interactions. Inside the sandbox, the git client authenticates using a custom-built scoped credential. This proxy:

* Manages GitHub authentication securely - the git client uses a scoped credential inside the sandbox, which the proxy verifies and translates to your actual GitHub authentication token
* Restricts git push operations to the current working branch for safety
* Enables seamless cloning, fetching, and PR operations while maintaining security boundaries

#### Security proxy

Environments run behind an HTTP/HTTPS network proxy for security and abuse prevention purposes. All outbound internet traffic passes through this proxy, which provides:

* Protection against malicious requests
* Rate limiting and abuse prevention
* Content filtering for enhanced security

### Access levels

By default, network access is limited to [allowlisted domains](#default-allowed-domains).

You can configure custom network access, including disabling network access.

### Default allowed domains

When using "Limited" network access, the following domains are allowed by default:

#### Anthropic Services

* api.anthropic.com
* statsig.anthropic.com
* platform.claude.com
* code.claude.com
* claude.ai

#### Version Control

* github.com
* www.github.com
* api.github.com
* npm.pkg.github.com
* raw\.githubusercontent.com
* pkg-npm.githubusercontent.com
* objects.githubusercontent.com
* codeload.github.com
* avatars.githubusercontent.com
* camo.githubusercontent.com
* gist.github.com
* gitlab.com
* www.gitlab.com
* registry.gitlab.com
* bitbucket.org
* www.bitbucket.org
* api.bitbucket.org

#### Container Registries

* registry-1.docker.io
* auth.docker.io
* index.docker.io
* hub.docker.com
* www.docker.com
* production.cloudflare.docker.com
* download.docker.com
* gcr.io
* \*.gcr.io
* ghcr.io
* mcr.microsoft.com
* \*.data.mcr.microsoft.com
* public.ecr.aws

#### Cloud Platforms

* cloud.google.com
* accounts.google.com
* gcloud.google.com
* \*.googleapis.com
* storage.googleapis.com
* compute.googleapis.com
* container.googleapis.com
* azure.com
* portal.azure.com
* microsoft.com
* www.microsoft.com
* \*.microsoftonline.com
* packages.microsoft.com
* dotnet.microsoft.com
* dot.net
* visualstudio.com
* dev.azure.com
* \*.amazonaws.com
* \*.api.aws
* oracle.com
* www.oracle.com
* java.com
* www.java.com
* java.net
* www.java.net
* download.oracle.com
* yum.oracle.com

#### Package Managers - JavaScript/Node

* registry.npmjs.org
* www.npmjs.com
* www.npmjs.org
* npmjs.com
* npmjs.org
* yarnpkg.com
* registry.yarnpkg.com

#### Package Managers - Python

* pypi.org
* www.pypi.org
* files.pythonhosted.org
* pythonhosted.org
* test.pypi.org
* pypi.python.org
* pypa.io
* www.pypa.io

#### Package Managers - Ruby

* rubygems.org
* www.rubygems.org
* api.rubygems.org
* index.rubygems.org
* ruby-lang.org
* www.ruby-lang.org
* rubyforge.org
* www.rubyforge.org
* rubyonrails.org
* www.rubyonrails.org
* rvm.io
* get.rvm.io

#### Package Managers - Rust

* crates.io
* www.crates.io
* index.crates.io
* static.crates.io
* rustup.rs
* static.rust-lang.org
* www.rust-lang.org

#### Package Managers - Go

* proxy.golang.org
* sum.golang.org
* index.golang.org
* golang.org
* www.golang.org
* goproxy.io
* pkg.go.dev

#### Package Managers - JVM

* maven.org
* repo.maven.org
* central.maven.org
* repo1.maven.org
* jcenter.bintray.com
* gradle.org
* www.gradle.org
* services.gradle.org
* plugins.gradle.org
* kotlin.org
* www.kotlin.org
* spring.io
* repo.spring.io

#### Package Managers - Other Languages

* packagist.org (PHP Composer)
* www.packagist.org
* repo.packagist.org
* nuget.org (.NET NuGet)
* www.nuget.org
* api.nuget.org
* pub.dev (Dart/Flutter)
* api.pub.dev
* hex.pm (Elixir/Erlang)
* www.hex.pm
* cpan.org (Perl CPAN)
* www.cpan.org
* metacpan.org
* www.metacpan.org
* api.metacpan.org
* cocoapods.org (iOS/macOS)
* www.cocoapods.org
* cdn.cocoapods.org
* haskell.org
* www.haskell.org
* hackage.haskell.org
* swift.org
* www.swift.org

#### Linux Distributions

* archive.ubuntu.com
* security.ubuntu.com
* ubuntu.com
* www.ubuntu.com
* \*.ubuntu.com
* ppa.launchpad.net
* launchpad.net
* www.launchpad.net

#### Development Tools & Platforms

* dl.k8s.io (Kubernetes)
* pkgs.k8s.io
* k8s.io
* www.k8s.io
* releases.hashicorp.com (HashiCorp)
* apt.releases.hashicorp.com
* rpm.releases.hashicorp.com
* archive.releases.hashicorp.com
* hashicorp.com
* www.hashicorp.com
* repo.anaconda.com (Anaconda/Conda)
* conda.anaconda.org
* anaconda.org
* www.anaconda.com
* anaconda.com
* continuum.io
* apache.org (Apache)
* www.apache.org
* archive.apache.org
* downloads.apache.org
* eclipse.org (Eclipse)
* www.eclipse.org
* download.eclipse.org
* nodejs.org (Node.js)
* www.nodejs.org

#### Cloud Services & Monitoring

* statsig.com
* www.statsig.com
* api.statsig.com
* sentry.io
* \*.sentry.io
* http-intake.logs.datadoghq.com
* \*.datadoghq.com
* \*.datadoghq.eu

#### Content Delivery & Mirrors

* sourceforge.net
* \*.sourceforge.net
* packagecloud.io
* \*.packagecloud.io

#### Schema & Configuration

* json-schema.org
* www.json-schema.org
* json.schemastore.org
* www.schemastore.org

#### Model Context Protocol

* \*.modelcontextprotocol.io

> Domains marked with `*` indicate wildcard subdomain matching. For example, `*.gcr.io` allows access to any subdomain of `gcr.io`.

### Security best practices for customized network access

1. **Principle of least privilege**: Only enable the minimum network access required
2. **Audit regularly**: Review allowed domains periodically
3. **Use HTTPS**: Always prefer HTTPS endpoints over HTTP

## Security and isolation

Claude Code on the web provides strong security guarantees:

* **Isolated virtual machines**: Each session runs in an isolated, Anthropic-managed VM
* **Network access controls**: Network access is limited by default, and can be disabled

> When running with network access disabled, Claude Code is allowed to communicate with the Anthropic API which may still allow data to exit the isolated Claude Code VM.

* **Credential protection**: Sensitive credentials (such as git credentials or signing keys) are never inside the sandbox with Claude Code. Authentication is handled through a secure proxy using scoped credentials
* **Secure analysis**: Code is analyzed and modified within isolated VMs before creating PRs

## Pricing and rate limits

Claude Code on the web shares rate limits with all other Claude and Claude Code usage within your account. Running multiple tasks in parallel will consume more rate limits proportionately.

## Limitations

* **Repository authentication**: You can only move sessions from web to local when you are authenticated to the same account
* **Platform restrictions**: Claude Code on the web only works with code hosted in GitHub. Self-hosted [GitHub Enterprise Server](/en/github-enterprise-server) instances are supported for Team and Enterprise plans. GitLab and other non-GitHub repositories cannot be used with cloud sessions

## Best practices

1. **Automate environment setup**: Use [setup scripts](#setup-scripts) to install dependencies and configure tools before Claude Code launches. For more advanced scenarios, configure [SessionStart hooks](/en/hooks#sessionstart).
2. **Document requirements**: Clearly specify dependencies and commands in your `CLAUDE.md` file. If you have an `AGENTS.md` file, you can source it in your `CLAUDE.md` using `@AGENTS.md` to maintain a single source of truth.

## Related resources

* [Hooks configuration](/en/hooks)
* [Settings reference](/en/settings)
* [Security](/en/security)
* [Data usage](/en/data-usage)
