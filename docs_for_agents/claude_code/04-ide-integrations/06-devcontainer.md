# Development containers

Learn about the Claude Code development container for teams that need consistent, secure environments.

The reference [devcontainer setup](https://github.com/anthropics/claude-code/tree/main/.devcontainer) and associated [Dockerfile](https://github.com/anthropics/claude-code/blob/main/.devcontainer/Dockerfile) offer a preconfigured development container that you can use as is, or customize for your needs. This devcontainer works with the Visual Studio Code [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers) and similar tools.

The container's enhanced security measures (isolation and firewall rules) allow you to run `claude --dangerously-skip-permissions` to bypass permission prompts for unattended operation.

> While the devcontainer provides substantial protections, no system is completely immune to all attacks.
> When executed with `--dangerously-skip-permissions`, devcontainers don't prevent a malicious project from exfiltrating anything accessible in the devcontainer including Claude Code credentials.
> We recommend only using devcontainers when developing with trusted repositories.
> Always maintain good security practices and monitor Claude's activities.

## Key features

* **Production-ready Node.js**: Built on Node.js 20 with essential development dependencies
* **Security by design**: Custom firewall restricting network access to only necessary services
* **Developer-friendly tools**: Includes git, ZSH with productivity enhancements, fzf, and more
* **Seamless VS Code integration**: Pre-configured extensions and optimized settings
* **Session persistence**: Preserves command history and configurations between container restarts
* **Works everywhere**: Compatible with macOS, Windows, and Linux development environments

## Getting started in 4 steps

1. Install VS Code and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Clone the [Claude Code reference implementation](https://github.com/anthropics/claude-code/tree/main/.devcontainer) repository
3. Open the repository in VS Code
4. When prompted, click "Reopen in Container" (or use Command Palette: Cmd+Shift+P → "Dev Containers: Reopen in Container")

## Configuration breakdown

The devcontainer setup consists of three primary components:

* [**devcontainer.json**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/devcontainer.json): Controls container settings, extensions, and volume mounts
* [**Dockerfile**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/Dockerfile): Defines the container image and installed tools
* [**init-firewall.sh**](https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh): Establishes network security rules

## Security features

The container implements a multi-layered security approach with its firewall configuration:

* **Precise access control**: Restricts outbound connections to whitelisted domains only (npm registry, GitHub, Claude API, etc.)
* **Allowed outbound connections**: The firewall permits outbound DNS and SSH connections
* **Default-deny policy**: Blocks all other external network access
* **Startup verification**: Validates firewall rules when the container initializes
* **Isolation**: Creates a secure development environment separated from your main system

## Customization options

The devcontainer configuration is designed to be adaptable to your needs:

* Add or remove VS Code extensions based on your workflow
* Modify resource allocations for different hardware environments
* Adjust network access permissions
* Customize shell configurations and developer tooling

## Example use cases

### Secure client work

Use devcontainers to isolate different client projects, ensuring code and credentials never mix between environments.

### Team onboarding

New team members can get a fully configured development environment in minutes, with all necessary tools and settings pre-installed.

### Consistent CI/CD environments

Mirror your devcontainer configuration in CI/CD pipelines to ensure development and production environments match.

## Related resources

* [VS Code devcontainers documentation](https://code.visualstudio.com/docs/devcontainers/containers)
* [Claude Code security best practices](/en/security)
* [Enterprise network configuration](/en/network-config)
