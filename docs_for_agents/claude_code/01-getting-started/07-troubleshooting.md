# Troubleshooting

Discover solutions to common issues with Claude Code installation and usage.

## Troubleshoot installation issues

If you'd rather skip the terminal entirely, the [Claude Code Desktop app](/en/desktop-quickstart) lets you install and use Claude Code through a graphical interface. Download it for [macOS](https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect?utm_source=claude_code&utm_medium=docs) or [Windows](https://claude.com/download?utm_source=claude_code&utm_medium=docs) and start coding without any command-line setup.

Find the error message or symptom you're seeing:

| What you see | Solution |
| :-- | :-- |
| `command not found: claude` or `'claude' is not recognized` | [Fix your PATH](#command-not-found-claude-after-installation) |
| `syntax error near unexpected token '<'` | [Install script returns HTML](#install-script-returns-html-instead-of-a-shell-script) |
| `curl: (56) Failure writing output to destination` | [Download script first, then run it](#curl-56-failure-writing-output-to-destination) |
| `Killed` during install on Linux | [Add swap space for low-memory servers](#install-killed-on-low-memory-linux-servers) |
| `TLS connect error` or `SSL/TLS secure channel` | [Update CA certificates](#tls-or-ssl-connection-errors) |
| `Failed to fetch version` or can't reach download server | [Check network and proxy settings](#check-network-connectivity) |
| `irm is not recognized` or `&& is not valid` | [Use the right command for your shell](#windows-irm-or--not-recognized) |
| `Claude Code on Windows requires git-bash` | [Install or configure Git Bash](#windows-claude-code-on-windows-requires-git-bash) |
| `Error loading shared library` | [Wrong binary variant for your system](#linux-wrong-binary-variant-installed-muslglibc-mismatch) |
| `Illegal instruction` on Linux | [Architecture mismatch](#illegal-instruction-on-linux) |
| `dyld: cannot load` or `Abort trap` on macOS | [Binary incompatibility](#dyld-cannot-load-on-macos) |
| `Invoke-Expression: Missing argument in parameter list` | [Install script returns HTML](#install-script-returns-html-instead-of-a-shell-script) |
| `App unavailable in region` | Claude Code is not available in your country. See [supported countries](https://www.anthropic.com/supported-countries). |
| `unable to get local issuer certificate` | [Configure corporate CA certificates](#tls-or-ssl-connection-errors) |
| `OAuth error` or `403 Forbidden` | [Fix authentication](#authentication-issues) |

If your issue isn't listed, work through these diagnostic steps.

## Debug installation problems

### Check network connectivity

The installer downloads from `storage.googleapis.com`. Verify you can reach it:

```bash
curl -sI https://storage.googleapis.com
```

If this fails, your network may be blocking the connection. Common causes:

* Corporate firewalls or proxies blocking Google Cloud Storage
* Regional network restrictions: try a VPN or alternative network
* TLS/SSL issues: update your system's CA certificates, or check if `HTTPS_PROXY` is configured

If you're behind a corporate proxy, set `HTTPS_PROXY` and `HTTP_PROXY` to your proxy's address before installing. Ask your IT team for the proxy URL if you don't know it, or check your browser's proxy settings.

This example sets both proxy variables, then runs the installer through your proxy:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
curl -fsSL https://claude.ai/install.sh | bash
```

### Verify your PATH

If installation succeeded but you get a `command not found` or `not recognized` error when running `claude`, the install directory isn't in your PATH. Your shell searches for programs in directories listed in PATH, and the installer places `claude` at `~/.local/bin/claude` on macOS/Linux or `%USERPROFILE%\.local\bin\claude.exe` on Windows.

Check if the install directory is in your PATH by listing your PATH entries and filtering for `local/bin`:

**macOS/Linux:**

```bash
echo $PATH | tr ':' '\n' | grep local/bin
```

If there's no output, the directory is missing. Add it to your shell configuration:

```bash
# Zsh (macOS default)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Bash (Linux default)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Alternatively, close and reopen your terminal.

Verify the fix worked:

```bash
claude --version
```

**Windows PowerShell:**

```powershell
$env:PATH -split ';' | Select-String 'local\\bin'
```

If there's no output, add the install directory to your User PATH:

```powershell
$currentPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
[Environment]::SetEnvironmentVariable('PATH', "$currentPath;$env:USERPROFILE\.local\bin", 'User')
```

Restart your terminal for the change to take effect.

Verify the fix worked:

```powershell
claude --version
```

**Windows CMD:**

```batch
echo %PATH% | findstr /i "local\bin"
```

If there's no output, open System Settings, go to Environment Variables, and add `%USERPROFILE%\.local\bin` to your User PATH variable. Restart your terminal.

Verify the fix worked:

```batch
claude --version
```

### Check for conflicting installations

Multiple Claude Code installations can cause version mismatches or unexpected behavior. Check what's installed:

**macOS/Linux:**

List all `claude` binaries found in your PATH:

```bash
which -a claude
```

Check whether the native installer and npm versions are present:

```bash
ls -la ~/.local/bin/claude
```

```bash
ls -la ~/.claude/local/
```

```bash
npm -g ls @anthropic-ai/claude-code 2>/dev/null
```

**Windows PowerShell:**

```powershell
where.exe claude
Test-Path "$env:LOCALAPPDATA\Claude Code\claude.exe"
```

If you find multiple installations, keep only one. The native install at `~/.local/bin/claude` is recommended. Remove any extra installations:

Uninstall an npm global install:

```bash
npm uninstall -g @anthropic-ai/claude-code
```

Remove a Homebrew install on macOS:

```bash
brew uninstall --cask claude-code
```

### Check directory permissions

The installer needs write access to `~/.local/bin/` and `~/.claude/`. If installation fails with permission errors, check whether these directories are writable:

```bash
test -w ~/.local/bin && echo "writable" || echo "not writable"
test -w ~/.claude && echo "writable" || echo "not writable"
```

If either directory isn't writable, create the install directory and set your user as the owner:

```bash
sudo mkdir -p ~/.local/bin
sudo chown -R $(whoami) ~/.local
```

### Verify the binary works

If `claude` is installed but crashes or hangs on startup, run these checks to narrow down the cause.

Confirm the binary exists and is executable:

```bash
ls -la $(which claude)
```

On Linux, check for missing shared libraries. If `ldd` shows missing libraries, you may need to install system packages. On Alpine Linux and other musl-based distributions, see [Alpine Linux setup](/en/setup#alpine-linux-and-musl-based-distributions).

```bash
ldd $(which claude) | grep "not found"
```

Run a quick sanity check that the binary can execute:

```bash
claude --version
```

## Common installation issues

These are the most frequently encountered installation problems and their solutions.

### Install script returns HTML instead of a shell script

When running the install command, you may see one of these errors:

```text
bash: line 1: syntax error near unexpected token `<'
bash: line 1: `<!DOCTYPE html>'
```

On PowerShell, the same problem appears as:

```text
Invoke-Expression: Missing argument in parameter list.
```

This means the install URL returned an HTML page instead of the install script. If the HTML page says "App unavailable in region," Claude Code is not available in your country. See [supported countries](https://www.anthropic.com/supported-countries).

Otherwise, this can happen due to network issues, regional routing, or a temporary service disruption.

**Solutions:**

1. **Use an alternative install method**:

   On macOS or Linux, install via Homebrew:

   ```bash
   brew install --cask claude-code
   ```

   On Windows, install via WinGet:

   ```powershell
   winget install Anthropic.ClaudeCode
   ```

2. **Retry after a few minutes**: the issue is often temporary. Wait and try the original command again.

### `command not found: claude` after installation

The install finished but `claude` doesn't work. The exact error varies by platform:

| Platform | Error message |
| :-- | :-- |
| macOS | `zsh: command not found: claude` |
| Linux | `bash: claude: command not found` |
| Windows CMD | `'claude' is not recognized as an internal or external command` |
| PowerShell | `claude : The term 'claude' is not recognized as the name of a cmdlet` |

This means the install directory isn't in your shell's search path. See [Verify your PATH](#verify-your-path) for the fix on each platform.

### `curl: (56) Failure writing output to destination`

The `curl ... | bash` command downloads the script and passes it directly to Bash for execution using a pipe (`|`). This error means the connection broke before the script finished downloading. Common causes include network interruptions, the download being blocked mid-stream, or system resource limits.

**Solutions:**

1. **Check network stability**: Claude Code binaries are hosted on Google Cloud Storage. Test that you can reach it:
   ```bash
   curl -fsSL https://storage.googleapis.com -o /dev/null
   ```
   If the command completes silently, your connection is fine and the issue is likely intermittent. Retry the install command. If you see an error, your network may be blocking the download.

2. **Try an alternative install method**:

   On macOS or Linux:

   ```bash
   brew install --cask claude-code
   ```

   On Windows:

   ```powershell
   winget install Anthropic.ClaudeCode
   ```

### TLS or SSL connection errors

Errors like `curl: (35) TLS connect error`, `schannel: next InitializeSecurityContext failed`, or PowerShell's `Could not establish trust relationship for the SSL/TLS secure channel` indicate TLS handshake failures.

**Solutions:**

1. **Update your system CA certificates**:

   On Ubuntu/Debian:

   ```bash
   sudo apt-get update && sudo apt-get install ca-certificates
   ```

   On macOS via Homebrew:

   ```bash
   brew install ca-certificates
   ```

2. **On Windows, enable TLS 1.2** in PowerShell before running the installer:
   ```powershell
   [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
   irm https://claude.ai/install.ps1 | iex
   ```

3. **Check for proxy or firewall interference**: corporate proxies that perform TLS inspection can cause these errors, including `unable to get local issuer certificate`. Set `NODE_EXTRA_CA_CERTS` to your corporate CA certificate bundle:
   ```bash
   export NODE_EXTRA_CA_CERTS=/path/to/corporate-ca.pem
   ```
   Ask your IT team for the certificate file if you don't have it. You can also try on a direct connection to confirm the proxy is the cause.

### `Failed to fetch version from storage.googleapis.com`

The installer couldn't reach the download server. This typically means `storage.googleapis.com` is blocked on your network.

**Solutions:**

1. **Test connectivity directly**:
   ```bash
   curl -sI https://storage.googleapis.com
   ```

2. **If behind a proxy**, set `HTTPS_PROXY` so the installer can route through it. See [proxy configuration](/en/network-config#proxy-configuration) for details.
   ```bash
   export HTTPS_PROXY=http://proxy.example.com:8080
   curl -fsSL https://claude.ai/install.sh | bash
   ```

3. **If on a restricted network**, try a different network or VPN, or use an alternative install method:

   On macOS or Linux:

   ```bash
   brew install --cask claude-code
   ```

   On Windows:

   ```powershell
   winget install Anthropic.ClaudeCode
   ```

### Windows: `irm` or `&&` not recognized

If you see `'irm' is not recognized` or `The token '&&' is not valid`, you're running the wrong command for your shell.

* **`irm` not recognized**: you're in CMD, not PowerShell. You have two options:

  Open PowerShell by searching for "PowerShell" in the Start menu, then run the original install command:

  ```powershell
  irm https://claude.ai/install.ps1 | iex
  ```

  Or stay in CMD and use the CMD installer instead:

  ```batch
  curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
  ```

* **`&&` not valid**: you're in PowerShell but ran the CMD installer command. Use the PowerShell installer:
  ```powershell
  irm https://claude.ai/install.ps1 | iex
  ```

### Install killed on low-memory Linux servers

If you see `Killed` during installation on a VPS or cloud instance:

```text
Setting up Claude Code...
Installing Claude Code native build latest...
bash: line 142: 34803 Killed    "$binary_path" install ${TARGET:+"$TARGET"}
```

The Linux OOM killer terminated the process because the system ran out of memory. Claude Code requires at least 4 GB of available RAM.

**Solutions:**

1. **Add swap space** if your server has limited RAM. Swap uses disk space as overflow memory, letting the install complete even with low physical RAM.

   Create a 2 GB swap file and enable it:

   ```bash
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

   Then retry the installation:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

2. **Close other processes** to free memory before installing.

3. **Use a larger instance** if possible. Claude Code requires at least 4 GB of RAM.

### Install hangs in Docker

When installing Claude Code in a Docker container, installing as root into `/` can cause hangs.

**Solutions:**

1. **Set a working directory** before running the installer. When run from `/`, the installer scans the entire filesystem, which causes excessive memory usage. Setting `WORKDIR` limits the scan to a small directory:
   ```dockerfile
   WORKDIR /tmp
   RUN curl -fsSL https://claude.ai/install.sh | bash
   ```

2. **Increase Docker memory limits** if using Docker Desktop:
   ```bash
   docker build --memory=4g .
   ```

### Windows: Claude Desktop overrides `claude` CLI command

If you installed an older version of Claude Desktop, it may register a `Claude.exe` in the `WindowsApps` directory that takes PATH priority over Claude Code CLI. Running `claude` opens the Desktop app instead of the CLI.

Update Claude Desktop to the latest version to fix this issue.

### Windows: "Claude Code on Windows requires git-bash"

Claude Code on native Windows needs [Git for Windows](https://git-scm.com/downloads/win), which includes Git Bash.

**If Git is not installed**, download and install it from [git-scm.com/downloads/win](https://git-scm.com/downloads/win). During setup, select "Add to PATH." Restart your terminal after installing.

**If Git is already installed** but Claude Code still can't find it, set the path in your [settings.json file](/en/settings):

```json
{
  "env": {
    "CLAUDE_CODE_GIT_BASH_PATH": "C:\\Program Files\\Git\\bin\\bash.exe"
  }
}
```

If your Git is installed somewhere else, find the path by running `where.exe git` in PowerShell and use the `bin\bash.exe` path from that directory.

### Linux: wrong binary variant installed (musl/glibc mismatch)

If you see errors about missing shared libraries like `libstdc++.so.6` or `libgcc_s.so.1` after installation, the installer may have downloaded the wrong binary variant for your system.

```text
Error loading shared library libstdc++.so.6: No such file or directory
```

This can happen on glibc-based systems that have musl cross-compilation packages installed, causing the installer to misdetect the system as musl.

**Solutions:**

1. **Check which libc your system uses**:
   ```bash
   ldd /bin/ls | head -1
   ```
   If it shows `linux-vdso.so` or references to `/lib/x86_64-linux-gnu/`, you're on glibc. If it shows `musl`, you're on musl.

2. **If you're on glibc but got the musl binary**, remove the installation and reinstall. You can also manually download the correct binary from the GCS bucket at `https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases/{VERSION}/manifest.json`. File a [GitHub issue](https://github.com/anthropics/claude-code/issues) with the output of `ldd /bin/ls` and `ls /lib/libc.musl*`.

3. **If you're actually on musl** (Alpine Linux), install the required packages:
   ```bash
   apk add libgcc libstdc++ ripgrep
   ```

### `Illegal instruction` on Linux

If the installer prints `Illegal instruction` instead of the OOM `Killed` message, the downloaded binary doesn't match your CPU architecture. This commonly happens on ARM servers that receive an x86 binary, or on older CPUs that lack required instruction sets.

```text
bash: line 142: 2238232 Illegal instruction    "$binary_path" install ${TARGET:+"$TARGET"}
```

**Solutions:**

1. **Verify your architecture**:
   ```bash
   uname -m
   ```
   `x86_64` means 64-bit Intel/AMD, `aarch64` means ARM64. If the binary doesn't match, [file a GitHub issue](https://github.com/anthropics/claude-code/issues) with the output.

2. **Try an alternative install method** while the architecture issue is resolved:
   ```bash
   brew install --cask claude-code
   ```

### `dyld: cannot load` on macOS

If you see `dyld: cannot load` or `Abort trap: 6` during installation, the binary is incompatible with your macOS version or hardware.

```text
dyld: cannot load 'claude-2.1.42-darwin-x64' (load command 0x80000034 is unknown)
Abort trap: 6
```

**Solutions:**

1. **Check your macOS version**: Claude Code requires macOS 13.0 or later. Open the Apple menu and select About This Mac to check your version.

2. **Update macOS** if you're on an older version. The binary uses load commands that older macOS versions don't support.

3. **Try Homebrew** as an alternative install method:
   ```bash
   brew install --cask claude-code
   ```

### Windows installation issues: errors in WSL

You might encounter the following issues in WSL:

**OS/platform detection issues**: if you receive an error during installation, WSL may be using Windows `npm`. Try:

* Run `npm config set os linux` before installation
* Install with `npm install -g @anthropic-ai/claude-code --force --no-os-check`. Do not use `sudo`.

**Node not found errors**: if you see `exec: node: not found` when running `claude`, your WSL environment may be using a Windows installation of Node.js. You can confirm this with `which npm` and `which node`, which should point to Linux paths starting with `/usr/` rather than `/mnt/c/`. To fix this, try installing Node via your Linux distribution's package manager or via [`nvm`](https://github.com/nvm-sh/nvm).

**nvm version conflicts**: if you have nvm installed in both WSL and Windows, you may experience version conflicts when switching Node versions in WSL. This happens because WSL imports the Windows PATH by default, causing Windows nvm/npm to take priority over the WSL installation.

You can identify this issue by:

* Running `which npm` and `which node` - if they point to Windows paths (starting with `/mnt/c/`), Windows versions are being used
* Experiencing broken functionality after switching Node versions with nvm in WSL

To resolve this issue, fix your Linux PATH to ensure the Linux node/npm versions take priority:

**Primary solution: Ensure nvm is properly loaded in your shell**

The most common cause is that nvm isn't loaded in non-interactive shells. Add the following to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Load nvm if it exists
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
```

Or run directly in your current session:

```bash
source ~/.nvm/nvm.sh
```

**Alternative: Adjust PATH order**

If nvm is properly loaded but Windows paths still take priority, you can explicitly prepend your Linux paths to PATH in your shell configuration:

```bash
export PATH="$HOME/.nvm/versions/node/$(node -v)/bin:$PATH"
```

Warning: Avoid disabling Windows PATH importing via `appendWindowsPath = false` as this breaks the ability to call Windows executables from WSL. Similarly, avoid uninstalling Node.js from Windows if you use it for Windows development.

### WSL2 sandbox setup

[Sandboxing](/en/sandboxing) is supported on WSL2 but requires installing additional packages. If you see an error about missing `bubblewrap` or `socat` when running `/sandbox`, install the dependencies:

**Ubuntu/Debian:**

```bash
sudo apt-get install bubblewrap socat
```

**Fedora:**

```bash
sudo dnf install bubblewrap socat
```

WSL1 does not support sandboxing. If you see "Sandboxing requires WSL2", you need to upgrade to WSL2 or run Claude Code without sandboxing.

Sandboxed commands cannot launch Windows binaries such as `cmd.exe`, `powershell.exe`, or executables under `/mnt/c/`. WSL hands these off to the Windows host over a Unix socket, which the sandbox blocks. If a command needs to invoke a Windows binary, add it to [`excludedCommands`](/en/settings#sandbox-settings) so it runs outside the sandbox.

### Permission errors during installation

If the native installer fails with permission errors, the target directory may not be writable. See [Check directory permissions](#check-directory-permissions).

If you previously installed with npm and are hitting npm-specific permission errors, switch to the native installer:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

## Permissions and authentication

These sections address login failures, token issues, and permission prompt behavior.

### Repeated permission prompts

If you find yourself repeatedly approving the same commands, you can allow specific tools to run without approval using the `/permissions` command. See [Permissions docs](/en/permissions#manage-permissions).

### Authentication issues

If you're experiencing authentication problems:

1. Run `/logout` to sign out completely
2. Close Claude Code
3. Restart with `claude` and complete the authentication process again

If the browser doesn't open automatically during login, press `c` to copy the OAuth URL to your clipboard, then paste it into your browser manually.

### OAuth error: Invalid code

If you see `OAuth error: Invalid code. Please make sure the full code was copied`, the login code expired or was truncated during copy-paste.

**Solutions:**

* Press Enter to retry and complete the login quickly after the browser opens
* Type `c` to copy the full URL if the browser doesn't open automatically
* If using a remote/SSH session, the browser may open on the wrong machine. Copy the URL displayed in the terminal and open it in your local browser instead.

### 403 Forbidden after login

If you see `API Error: 403 {"error":{"type":"forbidden","message":"Request not allowed"}}` after logging in:

* **Claude Pro/Max users**: verify your subscription is active at [claude.ai/settings](https://claude.ai/settings)
* **Console users**: confirm your account has the "Claude Code" or "Developer" role assigned by your admin
* **Behind a proxy**: corporate proxies can interfere with API requests. See [network configuration](/en/network-config) for proxy setup.

### "This organization has been disabled" with an active subscription

If you see `API Error: 400 ... "This organization has been disabled"` despite having an active Claude subscription, an `ANTHROPIC_API_KEY` environment variable is overriding your subscription. This commonly happens when an old API key from a previous employer or project is still set in your shell profile.

When `ANTHROPIC_API_KEY` is present and you have approved it, Claude Code uses that key instead of your subscription's OAuth credentials. In non-interactive mode (`-p`), the key is always used when present. See [authentication precedence](/en/authentication#authentication-precedence) for the full resolution order.

To use your subscription instead, unset the environment variable and remove it from your shell profile:

```bash
unset ANTHROPIC_API_KEY
claude
```

Check `~/.zshrc`, `~/.bashrc`, or `~/.profile` for `export ANTHROPIC_API_KEY=...` lines and remove them to make the change permanent. Run `/status` inside Claude Code to confirm which authentication method is active.

### OAuth login fails in WSL2

Browser-based login in WSL2 may fail if WSL can't open your Windows browser. Set the `BROWSER` environment variable:

```bash
export BROWSER="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
claude
```

Or copy the URL manually: when the login prompt appears, press `c` to copy the OAuth URL, then paste it into your Windows browser.

### "Not logged in" or token expired

If Claude Code prompts you to log in again after a session, your OAuth token may have expired.

Run `/login` to re-authenticate. If this happens frequently, check that your system clock is accurate, as token validation depends on correct timestamps.

## Configuration file locations

Claude Code stores configuration in several locations:

| File | Purpose |
| :-- | :-- |
| `~/.claude/settings.json` | User settings (permissions, hooks, model overrides) |
| `.claude/settings.json` | Project settings (checked into source control) |
| `.claude/settings.local.json` | Local project settings (not committed) |
| `~/.claude.json` | Global state (theme, OAuth, MCP servers) |
| `.mcp.json` | Project MCP servers (checked into source control) |
| `managed-mcp.json` | [Managed MCP servers](/en/mcp#managed-mcp-configuration) |
| Managed settings | [Managed settings](/en/settings#settings-files) (server-managed, MDM/OS-level policies, or file-based) |

On Windows, `~` refers to your user home directory, such as `C:\Users\YourName`.

For details on configuring these files, see [Settings](/en/settings) and [MCP](/en/mcp).

### Resetting configuration

To reset Claude Code to default settings, you can remove the configuration files:

```bash
# Reset all user settings and state
rm ~/.claude.json
rm -rf ~/.claude/

# Reset project-specific settings
rm -rf .claude/
rm .mcp.json
```

Warning: This will remove all your settings, MCP server configurations, and session history.

## Performance and stability

These sections cover issues related to resource usage, responsiveness, and search behavior.

### High CPU or memory usage

Claude Code is designed to work with most development environments, but may consume significant resources when processing large codebases. If you're experiencing performance issues:

1. Use `/compact` regularly to reduce context size
2. Close and restart Claude Code between major tasks
3. Consider adding large build directories to your `.gitignore` file

### Command hangs or freezes

If Claude Code seems unresponsive:

1. Press Ctrl+C to attempt to cancel the current operation
2. If unresponsive, you may need to close the terminal and restart

### Search and discovery issues

If Search tool, `@file` mentions, custom agents, and custom skills aren't working, install system `ripgrep`:

```bash
# macOS (Homebrew)
brew install ripgrep

# Windows (winget)
winget install BurntSushi.ripgrep.MSVC

# Ubuntu/Debian
sudo apt install ripgrep

# Alpine Linux
apk add ripgrep

# Arch Linux
pacman -S ripgrep
```

Then set `USE_BUILTIN_RIPGREP=0` in your [environment](/en/env-vars).

### Slow or incomplete search results on WSL

Disk read performance penalties when [working across file systems on WSL](https://learn.microsoft.com/en-us/windows/wsl/filesystems) may result in fewer-than-expected matches when using Claude Code on WSL. Search still functions, but returns fewer results than on a native filesystem.

Note: `/doctor` will show Search as OK in this case.

**Solutions:**

1. **Submit more specific searches**: reduce the number of files searched by specifying directories or file types: "Search for JWT validation logic in the auth-service package" or "Find use of md5 hash in JS files".

2. **Move project to Linux filesystem**: if possible, ensure your project is located on the Linux filesystem (`/home/`) rather than the Windows filesystem (`/mnt/c/`).

3. **Use native Windows instead**: consider running Claude Code natively on Windows instead of through WSL, for better file system performance.

## IDE integration issues

If Claude Code does not connect to your IDE or behaves unexpectedly within an IDE terminal, try the solutions below.

### JetBrains IDE not detected on WSL2

If you're using Claude Code on WSL2 with JetBrains IDEs and getting "No available IDEs detected" errors, this is likely due to WSL2's networking configuration or Windows Firewall blocking the connection.

#### WSL2 networking modes

WSL2 uses NAT networking by default, which can prevent IDE detection. You have two options:

**Option 1: Configure Windows Firewall** (recommended)

1. Find your WSL2 IP address:
   ```bash
   wsl hostname -I
   # Example output: 172.21.123.45
   ```

2. Open PowerShell as Administrator and create a firewall rule:
   ```powershell
   New-NetFirewallRule -DisplayName "Allow WSL2 Internal Traffic" -Direction Inbound -Protocol TCP -Action Allow -RemoteAddress 172.21.0.0/16 -LocalAddress 172.21.0.0/16
   ```
   Adjust the IP range based on your WSL2 subnet from step 1.

3. Restart both your IDE and Claude Code

**Option 2: Switch to mirrored networking**

Add to `.wslconfig` in your Windows user directory:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL with `wsl --shutdown` from PowerShell.

Note: These networking issues only affect WSL2. WSL1 uses the host's network directly and doesn't require these configurations.

For additional JetBrains configuration tips, see the [JetBrains IDE guide](/en/jetbrains#plugin-settings).

### Report Windows IDE integration issues

If you're experiencing IDE integration problems on Windows, [create an issue](https://github.com/anthropics/claude-code/issues) with the following information:

* Environment type: native Windows (Git Bash) or WSL1/WSL2
* WSL networking mode, if applicable: NAT or mirrored
* IDE name and version
* Claude Code extension/plugin version
* Shell type: Bash, Zsh, PowerShell, etc.

### Escape key not working in JetBrains IDE terminals

If you're using Claude Code in JetBrains terminals and the `Esc` key doesn't interrupt the agent as expected, this is likely due to a keybinding clash with JetBrains' default shortcuts.

To fix this issue:

1. Go to Settings → Tools → Terminal
2. Either:
   * Uncheck "Move focus to the editor with Escape", or
   * Click "Configure terminal keybindings" and delete the "Switch focus to Editor" shortcut
3. Apply the changes

This allows the `Esc` key to properly interrupt Claude Code operations.

## Markdown formatting issues

Claude Code sometimes generates markdown files with missing language tags on code fences, which can affect syntax highlighting and readability in GitHub, editors, and documentation tools.

### Missing language tags in code blocks

If you notice code blocks like this in generated markdown:

```
function example() {
  return "hello";
}
```

Instead of properly tagged blocks like:

```javascript
function example() {
  return "hello";
}
```

**Solutions:**

1. **Ask Claude to add language tags**: request "Add appropriate language tags to all code blocks in this markdown file."

2. **Use post-processing hooks**: set up automatic formatting hooks to detect and add missing language tags. See [Auto-format code after edits](/en/hooks-guide#auto-format-code-after-edits) for an example of a PostToolUse formatting hook.

3. **Manual verification**: after generating markdown files, review them for proper code block formatting and request corrections if needed.

### Inconsistent spacing and formatting

If generated markdown has excessive blank lines or inconsistent spacing:

**Solutions:**

1. **Request formatting corrections**: ask Claude to "Fix spacing and formatting issues in this markdown file."

2. **Use formatting tools**: set up hooks to run markdown formatters like `prettier` or custom formatting scripts on generated markdown files.

3. **Specify formatting preferences**: include formatting requirements in your prompts or project [memory](/en/memory) files.

### Reduce markdown formatting issues

To minimize formatting issues:

* **Be explicit in requests**: ask for "properly formatted markdown with language-tagged code blocks"
* **Use project conventions**: document your preferred markdown style in [`CLAUDE.md`](/en/memory)
* **Set up validation hooks**: use post-processing hooks to automatically verify and fix common formatting issues

## Get more help

If you're experiencing issues not covered here:

1. Use the `/feedback` command within Claude Code to report problems directly to Anthropic
2. Check the [GitHub repository](https://github.com/anthropics/claude-code) for known issues
3. Run `/doctor` to diagnose issues. It checks:
   * Installation type, version, and search functionality
   * Auto-update status and available versions
   * Invalid settings files (malformed JSON, incorrect types)
   * MCP server configuration errors
   * Keybinding configuration problems
   * Context usage warnings (large CLAUDE.md files, high MCP token usage, unreachable permission rules)
   * Plugin and agent loading errors
4. Ask Claude directly about its capabilities and features - Claude has built-in access to its documentation
