# Claude Code with GitHub Enterprise Server

> Connect Claude Code to your self-hosted GitHub Enterprise Server instance for web sessions, code review, and plugin marketplaces.

<Note>
  GitHub Enterprise Server support is available for Team and Enterprise plans.
</Note>

GitHub Enterprise Server (GHES) support lets your organization use Claude Code with repositories hosted on your self-managed GitHub instance instead of github.com. Once an admin connects your GHES instance, developers can run web sessions, get automated code reviews, and install plugins from internal marketplaces without any per-repository configuration.

For repositories on github.com, see [Claude Code on the web](/en/claude-code-on-the-web) and [Code Review](/en/code-review). To run Claude in your own CI infrastructure, see [GitHub Actions](/en/github-actions).

## What works with GitHub Enterprise Server

The table below shows which Claude Code features support GHES and any differences from github.com behavior.

| Feature                | GHES support    | Notes                                                                                                                        |
| :--------------------- | :-------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| Claude Code on the web | ✅ Supported     | Admin connects the GHES instance once; developers use `claude --remote` or [claude.ai/code](https://claude.ai/code) as usual |
| Code Review            | ✅ Supported     | Same automated PR reviews as github.com                                                                                      |
| Teleport sessions      | ✅ Supported     | Move sessions between web and terminal with `/teleport`                                                                      |
| Plugin marketplaces    | ✅ Supported     | Use full git URLs instead of `owner/repo` shorthand                                                                          |
| Contribution metrics   | ✅ Supported     | Delivered via webhooks to the [analytics dashboard](/en/analytics)                                                           |
| GitHub Actions         | ✅ Supported     | Requires manual workflow setup; `/install-github-app` is github.com only                                                     |
| GitHub MCP server      | ❌ Not supported | The GitHub MCP server does not work with GHES instances                                                                      |

## Admin setup

An admin connects your GHES instance to Claude Code once. After that, developers in your organization can use GHES repositories without any additional configuration. You need admin access to your Claude organization and permission to create GitHub Apps on your GHES instance.

The guided setup generates a GitHub App manifest and redirects you to your GHES instance to create the app in one click. If your environment blocks the redirect flow, an [alternative manual setup](#manual-setup) is available.

<Steps>
  <Step title="Open Claude Code admin settings">
    Go to [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code) and find the GitHub Enterprise Server section.
  </Step>

  <Step title="Start the guided setup">
    Click **Connect**. Enter a display name for the connection and your GHES hostname, for example `github.example.com`. If your GHES instance uses a self-signed or private certificate authority, paste the CA certificate in the optional field.
  </Step>

  <Step title="Create the GitHub App">
    Click **Continue to GitHub Enterprise**. Your browser redirects to your GHES instance with a pre-filled app manifest. Review the configuration and click **Create GitHub App**. GHES redirects you back to Claude with the app credentials stored automatically.
  </Step>

  <Step title="Install the app on your repositories">
    From the GitHub App page on your GHES instance, install the app on the repositories or organizations you want Claude to access. You can start with a subset and add more later.
  </Step>

  <Step title="Enable features">
    Return to [claude.ai/admin-settings/claude-code](https://claude.ai/admin-settings/claude-code) and enable [Code Review](/en/code-review#set-up-code-review) and [contribution metrics](/en/analytics#enable-contribution-metrics) for your GHES repositories using the same configuration as github.com.
  </Step>
</Steps>

### GitHub App permissions

The manifest configures the GitHub App with the permissions and webhook events Claude needs across web sessions, Code Review, and contribution metrics:

| Permission       | Access         | Used for                                    |
| :--------------- | :------------- | :------------------------------------------ |
| Contents         | Read and write | Cloning repositories and pushing branches   |
| Pull requests    | Read and write | Creating PRs and posting review comments    |
| Issues           | Read and write | Responding to issue mentions                |
| Checks           | Read and write | Posting Code Review check runs              |
| Actions          | Read           | Reading CI status for auto-fix              |
| Repository hooks | Read and write | Receiving webhooks for contribution metrics |
| Metadata         | Read           | Required by GitHub for all apps             |

The app subscribes to `pull_request`, `issue_comment`, `pull_request_review_comment`, `pull_request_review`, and `check_run` events.

### Manual setup

If the guided redirect flow is blocked by your network configuration, click **Add manually** instead of Connect. Create a GitHub App on your GHES instance with the [permissions and events above](#github-app-permissions), then enter the app credentials in the form: hostname, OAuth client ID and secret, GitHub App ID, client ID, client secret, webhook secret, and private key.

### Network requirements

Your GHES instance must be reachable from Anthropic infrastructure so Claude can clone repositories and post review comments. If your GHES instance is behind a firewall, allowlist the [Anthropic API IP addresses](https://platform.claude.com/docs/en/api/ip-addresses).

## Developer workflow

Once your admin has connected the GHES instance, no developer-side configuration is needed. Claude Code detects your GHES hostname automatically from the git remote in your working directory.

Clone a repository from your GHES instance as you normally would:

```bash  theme={null}
git clone git@github.example.com:platform/api-service.git
cd api-service
```

Then start a web session. Claude detects the GHES host from your git remote and routes the session through your organization's configured instance:

```bash  theme={null}
claude --remote "Add retry logic to the payment webhook handler"
```

The session runs on Anthropic infrastructure, clones your repository from GHES, and pushes changes back to a branch. Monitor progress with `/tasks` or at [claude.ai/code](https://claude.ai/code). See [Claude Code on the web](/en/claude-code-on-the-web) for the full remote session workflow including diff review, auto-fix, and scheduled tasks.

### Teleport sessions to your terminal

Pull a web session into your local terminal with `/teleport` or `claude --teleport`. Teleport verifies you're in a checkout of the same GHES repository before fetching the branch and loading the session history. See [teleport requirements](/en/claude-code-on-the-web#requirements-for-teleporting) for details.

## Plugin marketplaces on GHES

Host plugin marketplaces on your GHES instance to distribute internal tooling across your organization. The marketplace structure is identical to github.com-hosted marketplaces; the only difference is how you reference them.

### Add a GHES marketplace

The `owner/repo` shorthand always resolves to github.com. For GHES-hosted marketplaces, use the full git URL:

```bash  theme={null}
/plugin marketplace add git@github.example.com:platform/claude-plugins.git
```

HTTPS URLs work as well:

```bash  theme={null}
/plugin marketplace add https://github.example.com/platform/claude-plugins.git
```

See [Create and distribute a plugin marketplace](/en/plugin-marketplaces) for the full guide to building marketplaces.

### Allowlist GHES marketplaces in managed settings

If your organization uses [managed settings](/en/settings) to restrict which marketplaces developers can add, use the `hostPattern` source type to allow all marketplaces from your GHES instance without enumerating each repository:

```json  theme={null}
{
  "strictKnownMarketplaces": [
    {
      "source": "hostPattern",
      "hostPattern": "^github\\.example\\.com$"
    }
  ]
}
```

You can also pre-register marketplaces for developers so they appear without manual setup. This example makes an internal tools marketplace available organization-wide:

```json  theme={null}
{
  "extraKnownMarketplaces": {
    "internal-tools": {
      "source": {
        "source": "git",
        "url": "git@github.example.com:platform/claude-plugins.git"
      }
    }
  }
}
```

See the [strictKnownMarketplaces](/en/settings#strictknownmarketplaces) and [extraKnownMarketplaces](/en/settings#extraknownmarketplaces) settings reference for the complete schema.

## Limitations

A few features behave differently on GHES than on github.com. The [feature table](#what-works-with-github-enterprise-server) summarizes support; this section covers the workarounds.

* **`/install-github-app` command**: follow the [admin setup](#admin-setup) flow on claude.ai instead. If you also want GitHub Actions workflows on GHES, adapt the [example workflow](https://github.com/anthropics/claude-code-action/blob/main/examples/claude.yml) manually.
* **GitHub MCP server**: use the `gh` CLI configured for your GHES host instead. Run `gh auth login --hostname github.example.com` to authenticate, then Claude can use `gh` commands in sessions.

## Troubleshooting

### Web session fails to clone repository

If `claude --remote` fails with a clone error, verify that your admin has completed setup for your GHES instance and that the GitHub App is installed on the repository you're working in. Check with your admin that the instance hostname registered in Claude settings matches the hostname in your git remote.

### Marketplace add fails with a policy error

If `/plugin marketplace add` is blocked for your GHES URL, your organization has restricted marketplace sources. Ask your admin to add a `hostPattern` entry for your GHES hostname in [managed settings](#allowlist-ghes-marketplaces-in-managed-settings).

### GHES instance not reachable

If reviews or web sessions time out, your GHES instance may not be reachable from Anthropic infrastructure. Confirm your firewall allows inbound connections from the [Anthropic API IP addresses](https://platform.claude.com/docs/en/api/ip-addresses).

## Related resources

These pages cover the features referenced throughout this guide in more depth:

* [Claude Code on the web](/en/claude-code-on-the-web): run Claude Code sessions on cloud infrastructure
* [Code Review](/en/code-review): automated PR reviews
* [Plugin marketplaces](/en/plugin-marketplaces): build and distribute plugin catalogs
* [Analytics](/en/analytics): track usage and contribution metrics
* [Managed settings](/en/settings): organization-wide policy configuration
* [Network configuration](/en/network-config): firewall and IP allowlist requirements
