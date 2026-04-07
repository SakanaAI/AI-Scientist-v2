# Advanced setup

System requirements, platform-specific installation, version management, and uninstallation for Claude Code.

This page covers system requirements, platform-specific installation details, updates, and uninstallation. For a guided walkthrough of your first session, see the [quickstart](/en/quickstart). If you've never used a terminal before, see the [terminal guide](/en/terminal-guide).

## System requirements

Claude Code runs on the following platforms and configurations:

* **Operating system**:
  * macOS 13.0+
  * Windows 10 1809+ or Windows Server 2019+
  * Ubuntu 20.04+
  * Debian 10+
  * Alpine Linux 3.19+
* **Hardware**: 4 GB+ RAM
* **Network**: internet connection required. See [network configuration](/en/network-config#network-access-requirements).
* **Shell**: Bash, Zsh, PowerShell, or CMD. On Windows, [Git for Windows](https://git-scm.com/downloads/win) is required.
* **Location**: [Anthropic supported countries](https://www.anthropic.com/supported-countries)

### Additional dependencies

* **ripgrep**: usually included with Claude Code. If search fails, see [search troubleshooting](/en/troubleshooting#search-and-discovery-issues).

## Install Claude Code

> Prefer a graphical interface? The [Desktop app](/en/desktop-quickstart) lets you use Claude Code without the terminal. Download it for [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) or [Windows](https://claude.com/download?utm_source=claude_code&utm_medium=docs).
>
> New to the terminal? See the [terminal guide](/en/terminal-guide) for step-by-step instructions.

To install Claude Code, use one of the following methods:

### Native Install (Recommended)

**macOS, Linux, WSL:**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```powershell
irm https://claude.ai/install.ps1 | iex
```

**Windows CMD:**

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

If you see `The token '&&' is not a valid statement separator`, you're in PowerShell, not CMD. Use the PowerShell command above instead. Your prompt shows `PS C:\` when you're in PowerShell.

**Windows requires [Git for Windows](https://git-scm.com/downloads/win).** Install it first if you don't have it.

> Native installations automatically update in the background to keep you on the latest version.

### Homebrew

```bash
brew install --cask claude-code
```

> Homebrew installations do not auto-update. Run `brew upgrade claude-code` periodically to get the latest features and security fixes.

### WinGet

```powershell
winget install Anthropic.ClaudeCode
```

> WinGet installations do not auto-update. Run `winget upgrade Anthropic.ClaudeCode` periodically to get the latest features and security fixes.

After installation completes, open a terminal in the project you want to work in and start Claude Code:

```bash
claude
```

If you encounter any issues during installation, see the [troubleshooting guide](/en/troubleshooting).

### Set up on Windows

Claude Code on Windows requires [Git for Windows](https://git-scm.com/downloads/win) or WSL. You can launch `claude` from PowerShell, CMD, or Git Bash. Claude Code uses Git Bash internally to run commands. You do not need to run PowerShell as Administrator.

**Option 1: Native Windows with Git Bash**

Install [Git for Windows](https://git-scm.com/downloads/win), then run the install command from PowerShell or CMD.

If Claude Code can't find your Git Bash installation, set the path in your [settings.json file](/en/settings):

```json
{
  "env": {
    "CLAUDE_CODE_GIT_BASH_PATH": "C:\\Program Files\\Git\\bin\\bash.exe"
  }
}
```

Claude Code can also run PowerShell natively on Windows as an opt-in preview. See [PowerShell tool](/en/tools-reference#powershell-tool) for setup and limitations.

**Option 2: WSL**

Both WSL 1 and WSL 2 are supported. WSL 2 supports [sandboxing](/en/sandboxing) for enhanced security. WSL 1 does not support sandboxing.

### Alpine Linux and musl-based distributions

The native installer on Alpine and other musl/uClibc-based distributions requires `libgcc`, `libstdc++`, and `ripgrep`. Install these using your distribution's package manager, then set `USE_BUILTIN_RIPGREP=0`.

This example installs the required packages on Alpine:

```bash
apk add libgcc libstdc++ ripgrep
```

Then set `USE_BUILTIN_RIPGREP` to `0` in your [`settings.json`](/en/settings#available-settings) file:

```json
{
  "env": {
    "USE_BUILTIN_RIPGREP": "0"
  }
}
```

## Verify your installation

After installing, confirm Claude Code is working:

```bash
claude --version
```

For a more detailed check of your installation and configuration, run [`claude doctor`](/en/troubleshooting#get-more-help):

```bash
claude doctor
```

## Authenticate

Claude Code requires a Pro, Max, Team, Enterprise, or Console account. The free Claude.ai plan does not include Claude Code access. You can also use Claude Code with a third-party API provider like [Amazon Bedrock](/en/amazon-bedrock), [Google Vertex AI](/en/google-vertex-ai), or [Microsoft Foundry](/en/microsoft-foundry).

After installing, log in by running `claude` and following the browser prompts. See [Authentication](/en/authentication) for all account types and team setup options.

## Update Claude Code

Native installations automatically update in the background. You can [configure the release channel](#configure-release-channel) to control whether you receive updates immediately or on a delayed stable schedule, or [disable auto-updates](#disable-auto-updates) entirely. Homebrew and WinGet installations require manual updates.

### Auto-updates

Claude Code checks for updates on startup and periodically while running. Updates download and install in the background, then take effect the next time you start Claude Code.

> Homebrew and WinGet installations do not auto-update. Use `brew upgrade claude-code` or `winget upgrade Anthropic.ClaudeCode` to update manually.
>
> **Known issue:** Claude Code may notify you of updates before the new version is available in these package managers. If an upgrade fails, wait and try again later.
>
> Homebrew keeps old versions on disk after upgrades. Run `brew cleanup claude-code` periodically to reclaim disk space.

### Configure release channel

Control which release channel Claude Code follows for auto-updates and `claude update` with the `autoUpdatesChannel` setting:

* `"latest"`, the default: receive new features as soon as they're released
* `"stable"`: use a version that is typically about one week old, skipping releases with major regressions

Configure this via `/config` → **Auto-update channel**, or add it to your [settings.json file](/en/settings):

```json
{
  "autoUpdatesChannel": "stable"
}
```

For enterprise deployments, you can enforce a consistent release channel across your organization using [managed settings](/en/permissions#managed-settings).

### Disable auto-updates

Set `DISABLE_AUTOUPDATER` to `"1"` in the `env` key of your [`settings.json`](/en/settings#available-settings) file:

```json
{
  "env": {
    "DISABLE_AUTOUPDATER": "1"
  }
}
```

### Update manually

To apply an update immediately without waiting for the next background check, run:

```bash
claude update
```

## Advanced installation options

These options are for version pinning, migrating from npm, and verifying binary integrity.

### Install a specific version

The native installer accepts either a specific version number or a release channel (`latest` or `stable`). The channel you choose at install time becomes your default for auto-updates. See [configure release channel](#configure-release-channel) for more information.

To install the latest version (default):

**macOS, Linux, WSL:**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

```powershell
irm https://claude.ai/install.ps1 | iex
```

**Windows CMD:**

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

To install the stable version:

**macOS, Linux, WSL:**

```bash
curl -fsSL https://claude.ai/install.sh | bash -s stable
```

**Windows PowerShell:**

```powershell
& ([scriptblock]::Create((irm https://claude.ai/install.ps1))) stable
```

**Windows CMD:**

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd stable && del install.cmd
```

To install a specific version number:

**macOS, Linux, WSL:**

```bash
curl -fsSL https://claude.ai/install.sh | bash -s 2.1.89
```

**Windows PowerShell:**

```powershell
& ([scriptblock]::Create((irm https://claude.ai/install.ps1))) 2.1.89
```

**Windows CMD:**

```batch
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd 2.1.89 && del install.cmd
```

### Deprecated npm installation

npm installation is deprecated. The native installer is faster, requires no dependencies, and auto-updates in the background. Use the [native installation](#install-claude-code) method when possible.

#### Migrate from npm to native

If you previously installed Claude Code with npm, switch to the native installer:

```bash
# Install the native binary
curl -fsSL https://claude.ai/install.sh | bash

# Remove the old npm installation
npm uninstall -g @anthropic-ai/claude-code
```

You can also run `claude install` from an existing npm installation to install the native binary alongside it, then remove the npm version.

#### Install with npm

If you need npm installation for compatibility reasons, you must have [Node.js 18+](https://nodejs.org/en/download) installed. Install the package globally:

```bash
npm install -g @anthropic-ai/claude-code
```

> Do NOT use `sudo npm install -g` as this can lead to permission issues and security risks. If you encounter permission errors, see [troubleshooting permission errors](/en/troubleshooting#permission-errors-during-installation).

### Binary integrity and code signing

Each release publishes a `manifest.json` containing SHA256 checksums for every platform binary. The manifest is signed with an Anthropic GPG key, so verifying the signature on the manifest transitively verifies every binary it lists.

#### Verify the manifest signature

Steps 1-3 require a POSIX shell with `gpg` and `curl`. On Windows, run them in Git Bash or WSL. Step 4 includes a PowerShell option.

**Step 1: Download and import the public key**

The release signing key is published at a fixed URL.

```bash
curl -fsSL https://downloads.claude.ai/keys/claude-code.asc | gpg --import
```

Display the fingerprint of the imported key.

```bash
gpg --fingerprint security@anthropic.com
```

Confirm the output includes this fingerprint:

```text
31DD DE24 DDFA B679 F42D  7BD2 BAA9 29FF 1A7E CACE
```

**Step 2: Download the manifest and signature**

Set `VERSION` to the release you want to verify.

```bash
REPO=https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases
VERSION=2.1.89
curl -fsSLO "$REPO/$VERSION/manifest.json"
curl -fsSLO "$REPO/$VERSION/manifest.json.sig"
```

**Step 3: Verify the signature**

Verify the detached signature against the manifest.

```bash
gpg --verify manifest.json.sig manifest.json
```

A valid result reports `Good signature from "Anthropic Claude Code Release Signing <security@anthropic.com>"`.

`gpg` also prints `WARNING: This key is not certified with a trusted signature!` for any freshly imported key. This is expected. The `Good signature` line confirms the cryptographic check passed. The fingerprint comparison in Step 1 confirms the key itself is authentic.

**Step 4: Check the binary against the manifest**

Compare the SHA256 checksum of your downloaded binary with the value listed under `platforms.<platform>.checksum` in `manifest.json`.

**Linux:**

```bash
sha256sum claude
```

**macOS:**

```bash
shasum -a 256 claude
```

**Windows PowerShell:**

```powershell
(Get-FileHash claude.exe -Algorithm SHA256).Hash.ToLower()
```

> Manifest signatures are available for releases from `2.1.89` onward. Earlier releases publish checksums in `manifest.json` without a detached signature.

#### Platform code signatures

In addition to the signed manifest, individual binaries carry platform-native code signatures where supported.

* **macOS**: signed by "Anthropic PBC" and notarized by Apple. Verify with `codesign --verify --verbose ./claude`.
* **Windows**: signed by "Anthropic, PBC". Verify with `Get-AuthenticodeSignature .\claude.exe`.
* **Linux**: use the manifest signature above to verify integrity. Linux binaries are not individually code-signed.

## Uninstall Claude Code

To remove Claude Code, follow the instructions for your installation method.

### Native installation

Remove the Claude Code binary and version files:

**macOS, Linux, WSL:**

```bash
rm -f ~/.local/bin/claude
rm -rf ~/.local/share/claude
```

**Windows PowerShell:**

```powershell
Remove-Item -Path "$env:USERPROFILE\.local\bin\claude.exe" -Force
Remove-Item -Path "$env:USERPROFILE\.local\share\claude" -Recurse -Force
```

### Homebrew installation

Remove the Homebrew cask:

```bash
brew uninstall --cask claude-code
```

### WinGet installation

Remove the WinGet package:

```powershell
winget uninstall Anthropic.ClaudeCode
```

### npm

Remove the global npm package:

```bash
npm uninstall -g @anthropic-ai/claude-code
```

### Remove configuration files

> Removing configuration files will delete all your settings, allowed tools, MCP server configurations, and session history.

To remove Claude Code settings and cached data:

**macOS, Linux, WSL:**

```bash
# Remove user settings and state
rm -rf ~/.claude
rm ~/.claude.json

# Remove project-specific settings (run from your project directory)
rm -rf .claude
rm -f .mcp.json
```

**Windows PowerShell:**

```powershell
# Remove user settings and state
Remove-Item -Path "$env:USERPROFILE\.claude" -Recurse -Force
Remove-Item -Path "$env:USERPROFILE\.claude.json" -Force

# Remove project-specific settings (run from your project directory)
Remove-Item -Path ".claude" -Recurse -Force
Remove-Item -Path ".mcp.json" -Force
```
