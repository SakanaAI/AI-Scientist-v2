# Connect Claude Code to tools via MCP

> Learn how to connect Claude Code to your tools with the Model Context Protocol.

Claude Code connects to external services, databases, and custom tools through the Model Context Protocol (MCP). MCP is an open standard for AI models to interact with external systems in a structured way.

With MCP, Claude Code can:

* Query databases and data warehouses
* Integrate with project management tools (Jira, Linear, Notion)
* Control source code tools (GitHub, GitLab)
* Interact with communication platforms (Slack)
* Access cloud infrastructure (AWS, Google Cloud, Azure)
* Use custom internal tools specific to your organization

This guide covers how to set up and use MCP servers in Claude Code. For creating your own MCP servers, see the [MCP server documentation](https://modelcontextprotocol.io/).

## How MCP works

MCP uses a client-server model:

* **MCP Client**: Claude Code
* **MCP Server**: external service (database, API, tool)

When you configure an MCP server in Claude Code, it:

1. Starts the server process (or connects to an existing one)
2. Discovers what tools and resources the server provides
3. Makes those tools available to Claude in the current session

Claude can then use these tools the same way it uses built-in tools like Read, Edit, and Bash.

## Install and enable MCP servers

### Built-in MCP servers

Claude Code comes with pre-configured MCP servers for common services. To enable one, add it to your settings file.

For example, to enable the GitHub MCP server, add this to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"]
    }
  }
}
```

### Configure an MCP server

MCP servers are configured in the `mcpServers` object in your settings files. Each server has:

| Field   | Description                                                   | Example                          |
| :------ | :------------------------------------------------------------ | :------------------------------- |
| `name`  | Identifier for the server                                     | `"github"`                       |
| `command` | The executable to run                                        | `"npx"`, `"python"`, `"/path/to/server"` |
| `args`  | Command-line arguments                                        | `["-y", "@anthropic-ai/mcp-server-github"]` |
| `env`   | Environment variables to set                                  | `{"GITHUB_TOKEN": "..."}` |

## Scope hierarchy and precedence

MCP servers are loaded from your settings files at different scopes. When the same server is defined at multiple levels, the highest-priority definition wins:

1. **Project settings** (`.claude/settings.json`)
2. **User settings** (`~/.claude/settings.json`)
3. **Managed settings** (organization-wide)

This means a project can override a user-level server configuration, and user configurations override managed defaults.

## Use MCP tools in Claude Code

Once an MCP server is connected, its tools appear in Claude's available tools. You can:

* Let Claude use them automatically when relevant to your task
* Explicitly ask Claude to use a specific tool
* Include them in [skills](/en/skills) to teach Claude how to use them effectively

For example, with the GitHub MCP server connected, you could ask Claude:

```
Create a GitHub issue for the bug we just found
```

Claude would use the `github_create_issue` tool (or equivalent) to complete the task.

## Common MCP servers

Popular MCP servers you can use with Claude Code include:

* **GitHub**: manage repositories, issues, pull requests
* **GitLab**: GitLab integration
* **Linear**: project management
* **Jira**: issue tracking
* **Slack**: send messages, read channels
* **Google Drive**: access files and documents
* **PostgreSQL**: query databases
* **AWS**: interact with AWS services

Many more are available. Check the [official MCP server registry](https://modelcontextprotocol.io/servers) for a complete list.

## Troubleshooting

### MCP server not connecting

If a server isn't appearing in your available tools:

1. Check that it's configured in the correct settings file
2. Verify the command path is correct and the executable exists
3. Run `claude --debug` to see startup errors
4. Check that any required environment variables (like API tokens) are set

### Tools not appearing

After configuring an MCP server, run `/mcp` to see the status of all connected servers and the tools they provide.

## Advanced configuration

### Environment variables

Pass secrets and configuration to MCP servers via environment variables:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

Do not commit tokens and secrets to version control. Store them in environment files or use your system's credential manager.

## Learn more

* [MCP Official Documentation](https://modelcontextprotocol.io/)
* [Creating MCP servers](https://modelcontextprotocol.io/docs/concepts/architecture)
* [MCP servers reference](https://modelcontextprotocol.io/servers)
