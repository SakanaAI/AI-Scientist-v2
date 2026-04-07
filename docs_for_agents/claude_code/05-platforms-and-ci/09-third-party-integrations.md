# Enterprise deployment overview

> Learn how Claude Code can integrate with various third-party services and infrastructure to meet enterprise deployment requirements.

Organizations can deploy Claude Code through Anthropic directly or through a cloud provider. This page helps you choose the right configuration.

## Compare deployment options

For most organizations, Claude for Teams or Claude for Enterprise provides the best experience. Team members get access to both Claude Code and Claude on the web with a single subscription, centralized billing, and no infrastructure setup required.

**Claude for Teams** is self-service and includes collaboration features, admin tools, and billing management. Best for smaller teams that need to get started quickly.

**Claude for Enterprise** adds SSO and domain capture, role-based permissions, compliance API access, and managed policy settings for deploying organization-wide Claude Code configurations. Best for larger organizations with security and compliance requirements.

Learn more about [Team plans](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) and [Enterprise plans](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan).

If your organization has specific infrastructure requirements, compare the options below:

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Claude for Teams/Enterprise</th>
      <th>Anthropic Console</th>
      <th>Amazon Bedrock</th>
      <th>Google Vertex AI</th>
      <th>Microsoft Foundry</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>Best for</td>
      <td>Most organizations (recommended)</td>
      <td>Individual developers</td>
      <td>AWS-native deployments</td>
      <td>GCP-native deployments</td>
      <td>Azure-native deployments</td>
    </tr>

    <tr>
      <td>Billing</td>
      <td><strong>Teams:</strong> \$150/seat (Premium) with PAYG available<br /><strong>Enterprise:</strong> <a href="https://claude.com/contact-sales?utm_source=claude_code&utm_medium=docs&utm_content=third_party_enterprise">Contact Sales</a></td>
      <td>PAYG</td>
      <td>PAYG through AWS</td>
      <td>PAYG through GCP</td>
      <td>PAYG through Azure</td>
    </tr>

    <tr>
      <td>Regions</td>
      <td>Supported [countries](https://www.anthropic.com/supported-countries)</td>
      <td>Supported [countries](https://www.anthropic.com/supported-countries)</td>
      <td>Multiple AWS [regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)</td>
      <td>Multiple GCP [regions](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations)</td>
      <td>Multiple Azure [regions](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/)</td>
    </tr>

    <tr>
      <td>Prompt caching</td>
      <td>Enabled by default</td>
      <td>Enabled by default</td>
      <td>Enabled by default</td>
      <td>Enabled by default</td>
      <td>Enabled by default</td>
    </tr>

    <tr>
      <td>Authentication</td>
      <td>Claude.ai SSO or email</td>
      <td>API key</td>
      <td>API key or AWS credentials</td>
      <td>GCP credentials</td>
      <td>API key or Microsoft Entra ID</td>
    </tr>

    <tr>
      <td>Cost tracking</td>
      <td>Usage dashboard</td>
      <td>Usage dashboard</td>
      <td>AWS Cost Explorer</td>
      <td>GCP Billing</td>
      <td>Azure Cost Management</td>
    </tr>

    <tr>
      <td>Includes Claude on web</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>

    <tr>
      <td>Enterprise features</td>
      <td>Team management, SSO, usage monitoring</td>
      <td>None</td>
      <td>IAM policies, CloudTrail</td>
      <td>IAM roles, Cloud Audit Logs</td>
      <td>RBAC policies, Azure Monitor</td>
    </tr>
  </tbody>
</table>

Select a deployment option to view setup instructions:

* [Claude for Teams or Enterprise](/en/authentication#claude-for-teams-or-enterprise)
* [Anthropic Console](/en/authentication#claude-console-authentication)
* [Amazon Bedrock](/en/amazon-bedrock)
* [Google Vertex AI](/en/google-vertex-ai)
* [Microsoft Foundry](/en/microsoft-foundry)

## Configure proxies and gateways

Most organizations can use a cloud provider directly without additional configuration. However, you may need to configure a corporate proxy or LLM gateway if your organization has specific network or management requirements. These are different configurations that can be used together:

* **Corporate proxy**: Routes traffic through an HTTP/HTTPS proxy. Use this if your organization requires all outbound traffic to pass through a proxy server for security monitoring, compliance, or network policy enforcement. Configure with the `HTTPS_PROXY` or `HTTP_PROXY` environment variables. Learn more in [Enterprise network configuration](/en/network-config).
* **LLM Gateway**: A service that sits between Claude Code and the cloud provider to handle authentication and routing. Use this if you need centralized usage tracking across teams, custom rate limiting or budgets, or centralized authentication management. Configure with the `ANTHROPIC_BASE_URL`, `ANTHROPIC_BEDROCK_BASE_URL`, or `ANTHROPIC_VERTEX_BASE_URL` environment variables. Learn more in [LLM gateway configuration](/en/llm-gateway).

The following examples show the environment variables to set in your shell or shell profile (`.bashrc`, `.zshrc`). See [Settings](/en/settings) for other configuration methods.

### Amazon Bedrock

<Tabs>
  <Tab title="Corporate proxy">
    Route Bedrock traffic through your corporate proxy by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Bedrock
    export CLAUDE_CODE_USE_BEDROCK=1
    export AWS_REGION=us-east-1

    # Configure corporate proxy
    export HTTPS_PROXY='https://proxy.example.com:8080'
    ```
  </Tab>

  <Tab title="LLM Gateway">
    Route Bedrock traffic through your LLM gateway by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Bedrock
    export CLAUDE_CODE_USE_BEDROCK=1

    # Configure LLM gateway
    export ANTHROPIC_BEDROCK_BASE_URL='https://your-llm-gateway.com/bedrock'
    export CLAUDE_CODE_SKIP_BEDROCK_AUTH=1  # If gateway handles AWS auth
    ```
  </Tab>
</Tabs>

### Microsoft Foundry

<Tabs>
  <Tab title="Corporate proxy">
    Route Foundry traffic through your corporate proxy by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Microsoft Foundry
    export CLAUDE_CODE_USE_FOUNDRY=1
    export ANTHROPIC_FOUNDRY_RESOURCE=your-resource
    export ANTHROPIC_FOUNDRY_API_KEY=your-api-key  # Or omit for Entra ID auth

    # Configure corporate proxy
    export HTTPS_PROXY='https://proxy.example.com:8080'
    ```
  </Tab>

  <Tab title="LLM Gateway">
    Route Foundry traffic through your LLM gateway by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Microsoft Foundry
    export CLAUDE_CODE_USE_FOUNDRY=1

    # Configure LLM gateway
    export ANTHROPIC_FOUNDRY_BASE_URL='https://your-llm-gateway.com'
    export CLAUDE_CODE_SKIP_FOUNDRY_AUTH=1  # If gateway handles Azure auth
    ```
  </Tab>
</Tabs>

### Google Vertex AI

<Tabs>
  <Tab title="Corporate proxy">
    Route Vertex AI traffic through your corporate proxy by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Vertex
    export CLAUDE_CODE_USE_VERTEX=1
    export CLOUD_ML_REGION=us-east5
    export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id

    # Configure corporate proxy
    export HTTPS_PROXY='https://proxy.example.com:8080'
    ```
  </Tab>

  <Tab title="LLM Gateway">
    Route Vertex AI traffic through your LLM gateway by setting the following [environment variables](/en/env-vars):

    ```bash  theme={null}
    # Enable Vertex
    export CLAUDE_CODE_USE_VERTEX=1

    # Configure LLM gateway
    export ANTHROPIC_VERTEX_BASE_URL='https://your-llm-gateway.com/vertex'
    export CLAUDE_CODE_SKIP_VERTEX_AUTH=1  # If gateway handles GCP auth
    ```
  </Tab>
</Tabs>

<Tip>
  Use `/status` in Claude Code to verify your proxy and gateway configuration is applied correctly.
</Tip>

## Best practices for organizations

### Invest in documentation and memory

We strongly recommend investing in documentation so that Claude Code understands your codebase. Organizations can deploy CLAUDE.md files at multiple levels:

* **Organization-wide**: Deploy to system directories like `/Library/Application Support/ClaudeCode/CLAUDE.md` (macOS) for company-wide standards
* **Repository-level**: Create `CLAUDE.md` files in repository roots containing project architecture, build commands, and contribution guidelines. Check these into source control so all users benefit

Learn more in [Memory and CLAUDE.md files](/en/memory).

### Simplify deployment

If you have a custom development environment, we find that creating a "one click" way to install Claude Code is key to growing adoption across an organization.

### Start with guided usage

Encourage new users to try Claude Code for codebase Q\&A, or on smaller bug fixes or feature requests. Ask Claude Code to make a plan. Check Claude's suggestions and give feedback if it's off-track. Over time, as users understand this new paradigm better, then they'll be more effective at letting Claude Code run more agentically.

### Pin model versions for cloud providers

If you deploy through [Bedrock](/en/amazon-bedrock), [Vertex AI](/en/google-vertex-ai), or [Foundry](/en/microsoft-foundry), pin specific model versions using `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, and `ANTHROPIC_DEFAULT_HAIKU_MODEL`. Without pinning, Claude Code aliases resolve to the latest version, which can break users when Anthropic releases a new model that isn't yet enabled in your account. See [Model configuration](/en/model-config#pin-models-for-third-party-deployments) for details.

### Configure security policies

Security teams can configure managed permissions for what Claude Code is and is not allowed to do, which cannot be overwritten by local configuration. [Learn more](/en/security).

### Leverage MCP for integrations

MCP is a great way to give Claude Code more information, such as connecting to ticket management systems or error logs. We recommend that one central team configures MCP servers and checks a `.mcp.json` configuration into the codebase so that all users benefit. [Learn more](/en/mcp).

At Anthropic, we trust Claude Code to power development across every Anthropic codebase. We hope you enjoy using Claude Code as much as we do.

## Next steps

Once you've chosen a deployment option and configured access for your team:

1. **Roll out to your team**: Share installation instructions and have team members [install Claude Code](/en/setup) and authenticate with their credentials.
2. **Set up shared configuration**: Create a [CLAUDE.md file](/en/memory) in your repositories to help Claude Code understand your codebase and coding standards.
3. **Configure permissions**: Review [security settings](/en/security) to define what Claude Code can and cannot do in your environment.
