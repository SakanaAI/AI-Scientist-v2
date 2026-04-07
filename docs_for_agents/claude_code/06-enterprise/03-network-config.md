# Enterprise network configuration

> Configure Claude Code for enterprise environments with proxy servers, custom Certificate Authorities (CA), and mutual Transport Layer Security (mTLS) authentication.

Claude Code supports various enterprise network and security configurations through environment variables. This includes routing traffic through corporate proxy servers, trusting custom Certificate Authorities (CA), and authenticating with mutual Transport Layer Security (mTLS) certificates for enhanced security.

<Note>
  All environment variables shown on this page can also be configured in [`settings.json`](/en/settings).
</Note>

## Proxy configuration

### Environment variables

Claude Code respects standard proxy environment variables:

```bash  theme={null}
# HTTPS proxy (recommended)
export HTTPS_PROXY=https://proxy.example.com:8080

# HTTP proxy (if HTTPS not available)
export HTTP_PROXY=http://proxy.example.com:8080

# Bypass proxy for specific requests - space-separated format
export NO_PROXY="localhost 192.168.1.1 example.com .example.com"
# Bypass proxy for specific requests - comma-separated format
export NO_PROXY="localhost,192.168.1.1,example.com,.example.com"
# Bypass proxy for all requests
export NO_PROXY="*"
```

<Note>
  Claude Code does not support SOCKS proxies.
</Note>

### Basic authentication

If your proxy requires basic authentication, include credentials in the proxy URL:

```bash  theme={null}
export HTTPS_PROXY=http://username:password@proxy.example.com:8080
```

<Warning>
  Avoid hardcoding passwords in scripts. Use environment variables or secure credential storage instead.
</Warning>

<Tip>
  For proxies requiring advanced authentication (NTLM, Kerberos, etc.), consider using an LLM Gateway service that supports your authentication method.
</Tip>

## Custom CA certificates

If your enterprise environment uses custom CAs for HTTPS connections (whether through a proxy or direct API access), configure Claude Code to trust them:

```bash  theme={null}
export NODE_EXTRA_CA_CERTS=/path/to/ca-cert.pem
```

## mTLS authentication

For enterprise environments requiring client certificate authentication:

```bash  theme={null}
# Client certificate for authentication
export CLAUDE_CODE_CLIENT_CERT=/path/to/client-cert.pem

# Client private key
export CLAUDE_CODE_CLIENT_KEY=/path/to/client-key.pem

# Optional: Passphrase for encrypted private key
export CLAUDE_CODE_CLIENT_KEY_PASSPHRASE="your-passphrase"
```

## Network access requirements

Claude Code requires access to the following URLs:

* `api.anthropic.com`: Claude API endpoints
* `claude.ai`: authentication for claude.ai accounts
* `platform.claude.com`: authentication for Anthropic Console accounts

Ensure these URLs are allowlisted in your proxy configuration and firewall rules. This is especially important when using Claude Code in containerized or restricted network environments.

The native installer and update checks also require the following URLs. Allowlist both, since the installer and auto-updater fetch from `storage.googleapis.com` while plugin downloads use `downloads.claude.ai`. If you install Claude Code through npm or manage your own binary distribution, end users may not need access:

* `storage.googleapis.com`: download bucket for the Claude Code binary and auto-updater
* `downloads.claude.ai`: CDN hosting the install script, version pointers, manifests, signing keys, and plugin executables

[Claude Code on the web](/en/claude-code-on-the-web) and [Code Review](/en/code-review) connect to your repositories from Anthropic-managed infrastructure. If your GitHub Enterprise Cloud organization restricts access by IP address, enable [IP allow list inheritance for installed GitHub Apps](https://docs.github.com/en/enterprise-cloud@latest/organizations/keeping-your-organization-secure/managing-security-settings-for-your-organization/managing-allowed-ip-addresses-for-your-organization#allowing-access-by-github-apps). The Claude GitHub App registers its IP ranges, so enabling this setting allows access without manual configuration. To [add the ranges to your allow list manually](https://docs.github.com/en/enterprise-cloud@latest/organizations/keeping-your-organization-secure/managing-security-settings-for-your-organization/managing-allowed-ip-addresses-for-your-organization#adding-an-allowed-ip-address) instead, or to configure other firewalls, see the [Anthropic API IP addresses](https://platform.claude.com/docs/en/api/ip-addresses).

For self-hosted [GitHub Enterprise Server](/en/github-enterprise-server) instances behind a firewall, allowlist the same [Anthropic API IP addresses](https://platform.claude.com/docs/en/api/ip-addresses) so Anthropic infrastructure can reach your GHES host to clone repositories and post review comments.

## Additional resources

* [Claude Code settings](/en/settings)
* [Environment variables reference](/en/env-vars)
* [Troubleshooting guide](/en/troubleshooting)
