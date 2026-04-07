# Configure server-managed settings (public beta)

> Centrally configure Claude Code for your organization through server-delivered settings, without requiring device management infrastructure.

Server-managed settings allow administrators to centrally configure Claude Code through a web-based interface on Claude.ai. Claude Code clients automatically receive these settings when users authenticate with their organization credentials.

This approach is designed for organizations that do not have device management infrastructure in place, or need to manage settings for users on unmanaged devices.

<Note>
  Server-managed settings are in public beta and available for [Claude for Teams](https://claude.com/pricing?utm_source=claude_code\&utm_medium=docs\&utm_content=server_settings_teams#team-&-enterprise) and [Claude for Enterprise](https://anthropic.com/contact-sales?utm_source=claude_code\&utm_medium=docs\&utm_content=server_settings_enterprise) customers. Features may evolve before general availability.
</Note>

## Requirements

To use server-managed settings, you need:

* Claude for Teams or Claude for Enterprise plan
* Claude Code version 2.1.38 or later for Claude for Teams, or version 2.1.30 or later for Claude for Enterprise
* Network access to `api.anthropic.com`

## Choose between server-managed and endpoint-managed settings

Claude Code supports two approaches for centralized configuration. Server-managed settings deliver configuration from Anthropic's servers. [Endpoint-managed settings](/en/settings#settings-files) are deployed directly to devices through native OS policies (macOS managed preferences, Windows registry) or managed settings files.

| Approach                                                     | Best for                                                 | Security model                                                                                            |
| :----------------------------------------------------------- | :------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| **Server-managed settings**                                  | Organizations without MDM, or users on unmanaged devices | Settings delivered from Anthropic's servers at authentication time                                        |
| **[Endpoint-managed settings](/en/settings#settings-files)** | Organizations with MDM or endpoint management            | Settings deployed to devices via MDM configuration profiles, registry policies, or managed settings files |

If your devices are enrolled in an MDM or endpoint management solution, endpoint-managed settings provide stronger security guarantees because the settings file can be protected from user modification at the OS level.

## Configure server-managed settings

<Steps>
  <Step title="Open the admin console">
    In [Claude.ai](https://claude.ai), navigate to **Admin Settings > Claude Code > Managed settings**.
  </Step>

  <Step title="Define your settings">
    Add your configuration as JSON. All [settings available in `settings.json`](/en/settings#available-settings) are supported, including [hooks](/en/hooks), [environment variables](/en/env-vars), and [managed-only settings](/en/permissions#managed-only-settings) like `allowManagedPermissionRulesOnly`.

    This example enforces a permission deny list, prevents users from bypassing permissions, and restricts permission rules to those defined in managed settings:

    ```json  theme={null}
    {
      "permissions": {
        "deny": [
          "Bash(curl *)",
          "Read(./.env)",
          "Read(./.env.*)",
          "Read(./secrets/**)"
        ],
        "disableBypassPermissionsMode": "disable"
      },
      "allowManagedPermissionRulesOnly": true
    }
    ```

    Hooks use the same format as in `settings.json`.

    This example runs an audit script after every file edit across the organization:

    ```json  theme={null}
    {
      "hooks": {
        "PostToolUse": [
          {
            "matcher": "Edit|Write",
            "hooks": [
              { "type": "command", "command": "/usr/local/bin/audit-edit.sh" }
            ]
          }
        ]
      }
    }
    ```

    To configure the [auto mode](/en/permission-modes#eliminate-prompts-with-auto-mode) classifier so it knows which repos, buckets, and domains your organization trusts:

    ```json  theme={null}
    {
      "autoMode": {
        "environment": [
          "Source control: github.example.com/acme-corp and all repos under it",
          "Trusted cloud buckets: s3://acme-build-artifacts, gs://acme-ml-datasets",
          "Trusted internal domains: *.corp.example.com"
        ]
      }
    }
    ```

    Because hooks execute shell commands, users see a [security approval dialog](#security-approval-dialogs) before they're applied. See [Configure the auto mode classifier](/en/permissions#configure-the-auto-mode-classifier) for how the `autoMode` entries affect what the classifier blocks and important warnings about the `allow` and `soft_deny` fields.
  </Step>

  <Step title="Save and deploy">
    Save your changes. Claude Code clients receive the updated settings on their next startup or hourly polling cycle.
  </Step>
</Steps>

### Verify settings delivery

To confirm that settings are being applied, ask a user to restart Claude Code. If the configuration includes settings that trigger the [security approval dialog](#security-approval-dialogs), the user sees a prompt describing the managed settings on startup. You can also verify that managed permission rules are active by having a user run `/permissions` to view their effective permission rules.

### Access control

The following roles can manage server-managed settings:

* **Primary Owner**
* **Owner**

Restrict access to trusted personnel, as settings changes apply to all users in the organization.

### Managed-only settings

Most [settings keys](/en/settings#available-settings) work in any scope. A handful of keys are only read from managed settings and have no effect when placed in user or project settings files. See [managed-only settings](/en/permissions#managed-only-settings) for the full list. Any setting not on that list can still be placed in managed settings and takes the highest precedence.

### Current limitations

Server-managed settings have the following limitations during the beta period:

* Settings apply uniformly to all users in the organization. Per-group configurations are not yet supported.
* [MCP server configurations](/en/mcp#managed-mcp-configuration) cannot be distributed through server-managed settings.

## Settings delivery

### Settings precedence

Server-managed settings and [endpoint-managed settings](/en/settings#settings-files) both occupy the highest tier in the Claude Code [settings hierarchy](/en/settings#settings-precedence). No other settings level can override them, including command line arguments.

Within the managed tier, the first source that delivers a non-empty configuration wins. Server-managed settings are checked first, then endpoint-managed settings. Sources do not merge: if server-managed settings deliver any keys at all, endpoint-managed settings are ignored entirely. If server-managed settings deliver nothing, endpoint-managed settings apply.

If you clear your server-managed configuration in the admin console with the intent of falling back to an endpoint-managed plist or registry policy, be aware that [cached settings](#fetch-and-caching-behavior) persist on client machines until the next successful fetch. Run `/status` to see which managed source is active.

### Fetch and caching behavior

Claude Code fetches settings from Anthropic's servers at startup and polls for updates hourly during active sessions.

**First launch without cached settings:**

* Claude Code fetches settings asynchronously
* If the fetch fails, Claude Code continues without managed settings
* There is a brief window before settings load where restrictions are not yet enforced

**Subsequent launches with cached settings:**

* Cached settings apply immediately at startup
* Claude Code fetches fresh settings in the background
* Cached settings persist through network failures

Claude Code applies settings updates automatically without a restart, except for advanced settings like OpenTelemetry configuration, which require a full restart to take effect.

### Enforce fail-closed startup

By default, if the remote settings fetch fails at startup, the CLI continues without managed settings. For environments where this brief unenforced window is unacceptable, set `forceRemoteSettingsRefresh: true` in your managed settings.

When this setting is active, the CLI blocks at startup until remote settings are freshly fetched. If the fetch fails, the CLI exits rather than proceeding without the policy. This setting self-perpetuates: once delivered from the server, it is also cached locally so that subsequent startups enforce the same behavior even before the first successful fetch of a new session.

To enable this, add the key to your managed settings configuration:

```json  theme={null}
{
  "forceRemoteSettingsRefresh": true
}
```

Before enabling this setting, ensure your network policies allow connectivity to `api.anthropic.com`. If that endpoint is unreachable, the CLI exits at startup and users cannot start Claude Code.

### Security approval dialogs

Certain settings that could pose security risks require explicit user approval before being applied:

* **Shell command settings**: settings that execute shell commands
* **Custom environment variables**: variables not in the known safe allowlist
* **Hook configurations**: any hook definition

When these settings are present, users see a security dialog explaining what is being configured. Users must approve to proceed. If a user rejects the settings, Claude Code exits.

<Note>
  In non-interactive mode with the `-p` flag, Claude Code skips security dialogs and applies settings without user approval.
</Note>

## Platform availability

Server-managed settings require a direct connection to `api.anthropic.com` and are not available when using third-party model providers:

* Amazon Bedrock
* Google Vertex AI
* Microsoft Foundry
* Custom API endpoints via `ANTHROPIC_BASE_URL` or [LLM gateways](/en/llm-gateway)

## Audit logging

Audit log events for settings changes are available through the compliance API or audit log export. Contact your Anthropic account team for access.

Audit events include the type of action performed, the account and device that performed the action, and references to the previous and new values.

## Security considerations

Server-managed settings provide centralized policy enforcement, but they operate as a client-side control. On unmanaged devices, users with admin or sudo access can modify the Claude Code binary, filesystem, or network configuration.

| Scenario                                         | Behavior                                                                                                                                                                                      |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| User edits the cached settings file              | Tampered file applies at startup, but correct settings restore on the next server fetch                                                                                                       |
| User deletes the cached settings file            | First-launch behavior occurs: settings fetch asynchronously with a brief unenforced window                                                                                                    |
| API is unavailable                               | Cached settings apply if available, otherwise managed settings are not enforced until the next successful fetch. With `forceRemoteSettingsRefresh: true`, the CLI exits instead of continuing |
| User authenticates with a different organization | Settings are not delivered for accounts outside the managed organization                                                                                                                      |
| User sets a non-default `ANTHROPIC_BASE_URL`     | Server-managed settings are bypassed when using third-party API providers                                                                                                                     |

To detect runtime configuration changes, use [`ConfigChange` hooks](/en/hooks#configchange) to log modifications or block unauthorized changes before they take effect.

For stronger enforcement guarantees, use [endpoint-managed settings](/en/settings#settings-files) on devices enrolled in an MDM solution.

## See also

Related pages for managing Claude Code configuration:

* [Settings](/en/settings): complete configuration reference including all available settings
* [Endpoint-managed settings](/en/settings#settings-files): managed settings deployed to devices by IT
* [Authentication](/en/authentication): set up user access to Claude Code
* [Security](/en/security): security safeguards and best practices
