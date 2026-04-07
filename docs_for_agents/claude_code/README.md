# Claude Code — Documentação Completa

> Espelho offline da documentação oficial do Claude Code (https://code.claude.com/docs).
> Gerado em: 2026-04-06 | Fonte: https://code.claude.com/docs/llms.txt

---

## Estrutura

```
claude-code-docs/
├── 01-getting-started/     # Instalação, primeiros passos, boas práticas
├── 02-core-concepts/       # Como funciona, memória, permissões, segurança
├── 03-configuration/       # Settings, variáveis, atalhos, terminal
├── 04-ide-integrations/    # VS Code, JetBrains, Desktop, Chrome, Web
├── 05-platforms-and-ci/    # GitHub Actions, GitLab, Slack, Bedrock, Vertex
├── 06-enterprise/          # SSO, gateway, rede, compliance, ZDR
├── 07-extensions/          # MCP, hooks, skills, plugins, agentes, scheduling
├── 08-reference/           # CLI reference, comandos, tools, changelog
└── 09-whats-new/           # Notas de versão semanais
```

---

## 01 — Getting Started

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-overview.md](01-getting-started/01-overview.md) | Visão geral do Claude Code e o que ele pode fazer |
| 2 | [02-quickstart.md](01-getting-started/02-quickstart.md) | Sua primeira tarefa: explorar codebase até commitar fix |
| 3 | [03-setup.md](01-getting-started/03-setup.md) | Instalação avançada, requisitos, atualizações |
| 4 | [04-desktop-quickstart.md](01-getting-started/04-desktop-quickstart.md) | Primeiros passos com o app Desktop |
| 5 | [05-common-workflows.md](01-getting-started/05-common-workflows.md) | Workflows de debug, teste, refactor, commit |
| 6 | [06-best-practices.md](01-getting-started/06-best-practices.md) | Dicas para uso eficiente |
| 7 | [07-troubleshooting.md](01-getting-started/07-troubleshooting.md) | Soluções para problemas comuns |
| 8 | [08-authentication.md](01-getting-started/08-authentication.md) | Login individual, teams e orgs |

## 02 — Core Concepts

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-how-claude-code-works.md](02-core-concepts/01-how-claude-code-works.md) | Agentic loop, tools embutidas, arquitetura |
| 2 | [02-memory.md](02-core-concepts/02-memory.md) | CLAUDE.md, auto memory, instruções persistentes |
| 3 | [03-context-window.md](02-core-concepts/03-context-window.md) | Gerenciamento e otimização da janela de contexto |
| 4 | [04-permissions.md](02-core-concepts/04-permissions.md) | Sistema de regras de permissão |
| 5 | [05-permission-modes.md](02-core-concepts/05-permission-modes.md) | Modos: default, acceptEdits, plan, auto, bypassPermissions |
| 6 | [06-interactive-mode.md](02-core-concepts/06-interactive-mode.md) | Atalhos, comandos, vim mode |
| 7 | [07-headless-mode.md](02-core-concepts/07-headless-mode.md) | Uso programático via CLI e Agent SDK |
| 8 | [08-costs.md](02-core-concepts/08-costs.md) | Rastreamento de tokens e gestão de custos |
| 9 | [09-checkpointing.md](02-core-concepts/09-checkpointing.md) | Estado de sessão e reversão de mudanças |
| 10 | [10-sandboxing.md](02-core-concepts/10-sandboxing.md) | Isolamento de filesystem e rede |
| 11 | [11-data-usage.md](02-core-concepts/11-data-usage.md) | Políticas de dados, retenção, telemetria |
| 12 | [12-security.md](02-core-concepts/12-security.md) | Salvaguardas e práticas de segurança |

## 03 — Configuration

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-settings.md](03-configuration/01-settings.md) | Todas as configurações do Claude Code |
| 2 | [02-model-config.md](03-configuration/02-model-config.md) | Configuração de modelos e aliases |
| 3 | [03-env-vars.md](03-configuration/03-env-vars.md) | Variáveis de ambiente (referência completa) |
| 4 | [04-keybindings.md](03-configuration/04-keybindings.md) | Personalização de atalhos de teclado |
| 5 | [05-terminal-config.md](03-configuration/05-terminal-config.md) | Otimização do terminal |
| 6 | [06-statusline.md](03-configuration/06-statusline.md) | Personalização da barra de status |
| 7 | [07-output-styles.md](03-configuration/07-output-styles.md) | Estilos de output |
| 8 | [08-fast-mode.md](03-configuration/08-fast-mode.md) | Modo rápido (Sonnet em vez de Opus) |
| 9 | [09-fullscreen.md](03-configuration/09-fullscreen.md) | Renderização fullscreen |
| 10 | [10-voice-dictation.md](03-configuration/10-voice-dictation.md) | Ditado por voz |
| 11 | [11-claude-directory.md](03-configuration/11-claude-directory.md) | Explorando o diretório .claude |
| 12 | [12-computer-use.md](03-configuration/12-computer-use.md) | Computer use via CLI |

## 04 — IDE Integrations & Surfaces

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-vs-code.md](04-ide-integrations/01-vs-code.md) | Extensão VS Code / Cursor |
| 2 | [02-jetbrains.md](04-ide-integrations/02-jetbrains.md) | Plugin JetBrains (IntelliJ, PyCharm, etc.) |
| 3 | [03-desktop.md](04-ide-integrations/03-desktop.md) | App Desktop completo |
| 4 | [04-chrome.md](04-ide-integrations/04-chrome.md) | Integração com Chrome |
| 5 | [05-claude-code-on-the-web.md](04-ide-integrations/05-claude-code-on-the-web.md) | Claude Code no navegador |
| 6 | [06-devcontainer.md](04-ide-integrations/06-devcontainer.md) | Dev containers |
| 7 | [07-remote-control.md](04-ide-integrations/07-remote-control.md) | Controle remoto de sessões |
| 8 | [08-channels.md](04-ide-integrations/08-channels.md) | Channels (Telegram, Discord, iMessage) |
| 9 | [09-channels-reference.md](04-ide-integrations/09-channels-reference.md) | Referência da API de Channels |

## 05 — Platforms & CI/CD

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-platforms.md](05-platforms-and-ci/01-platforms.md) | Visão geral de plataformas |
| 2 | [02-github-actions.md](05-platforms-and-ci/02-github-actions.md) | GitHub Actions |
| 3 | [03-gitlab-ci-cd.md](05-platforms-and-ci/03-gitlab-ci-cd.md) | GitLab CI/CD |
| 4 | [04-code-review.md](05-platforms-and-ci/04-code-review.md) | Code review automatizado |
| 5 | [05-slack.md](05-platforms-and-ci/05-slack.md) | Integração com Slack |
| 6 | [06-amazon-bedrock.md](05-platforms-and-ci/06-amazon-bedrock.md) | Amazon Bedrock |
| 7 | [07-google-vertex-ai.md](05-platforms-and-ci/07-google-vertex-ai.md) | Google Vertex AI |
| 8 | [08-microsoft-foundry.md](05-platforms-and-ci/08-microsoft-foundry.md) | Microsoft Foundry |
| 9 | [09-third-party-integrations.md](05-platforms-and-ci/09-third-party-integrations.md) | Enterprise deployment overview |
| 10 | [10-github-enterprise-server.md](05-platforms-and-ci/10-github-enterprise-server.md) | GitHub Enterprise Server |

## 06 — Enterprise

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-server-managed-settings.md](06-enterprise/01-server-managed-settings.md) | Settings gerenciados pelo servidor |
| 2 | [02-llm-gateway.md](06-enterprise/02-llm-gateway.md) | Configuração de LLM gateway |
| 3 | [03-network-config.md](06-enterprise/03-network-config.md) | Rede, proxy, mTLS |
| 4 | [04-analytics.md](06-enterprise/04-analytics.md) | Analytics de uso da equipe |
| 5 | [05-monitoring-usage.md](06-enterprise/05-monitoring-usage.md) | Monitoramento via OpenTelemetry |
| 6 | [06-legal-and-compliance.md](06-enterprise/06-legal-and-compliance.md) | Legal, compliance, BAA |
| 7 | [07-zero-data-retention.md](06-enterprise/07-zero-data-retention.md) | Zero Data Retention |

## 07 — Extensions & Customization

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-features-overview.md](07-extensions/01-features-overview.md) | Visão geral de extensibilidade |
| 2 | [02-mcp.md](07-extensions/02-mcp.md) | Model Context Protocol |
| 3 | [03-hooks-guide.md](07-extensions/03-hooks-guide.md) | Guia de hooks com exemplos |
| 4 | [04-hooks-reference.md](07-extensions/04-hooks-reference.md) | Referência técnica de hooks |
| 5 | [05-skills.md](07-extensions/05-skills.md) | Custom skills / comandos |
| 6 | [06-sub-agents.md](07-extensions/06-sub-agents.md) | Subagentes |
| 7 | [07-agent-teams.md](07-extensions/07-agent-teams.md) | Orquestração de times de agentes |
| 8 | [08-plugins.md](07-extensions/08-plugins.md) | Criação de plugins |
| 9 | [09-plugins-reference.md](07-extensions/09-plugins-reference.md) | Referência técnica de plugins |
| 10 | [10-discover-plugins.md](07-extensions/10-discover-plugins.md) | Descoberta e instalação de plugins |
| 11 | [11-plugin-marketplaces.md](07-extensions/11-plugin-marketplaces.md) | Criação de marketplaces |
| 12 | [12-scheduled-tasks.md](07-extensions/12-scheduled-tasks.md) | Tarefas agendadas (CLI /loop) |
| 13 | [13-desktop-scheduled-tasks.md](07-extensions/13-desktop-scheduled-tasks.md) | Agendamento no Desktop |
| 14 | [14-web-scheduled-tasks.md](07-extensions/14-web-scheduled-tasks.md) | Agendamento na nuvem |
| 15 | [15-ultraplan.md](07-extensions/15-ultraplan.md) | Planejamento na nuvem (ultraplan) |

## 08 — Reference

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [01-cli-reference.md](08-reference/01-cli-reference.md) | Referência completa da CLI |
| 2 | [02-commands.md](08-reference/02-commands.md) | Comandos built-in |
| 3 | [03-tools-reference.md](08-reference/03-tools-reference.md) | Referência de tools (Bash, LSP, etc.) |
| 4 | [04-changelog.md](08-reference/04-changelog.md) | Changelog de versões |

## 09 — What's New

| # | Arquivo | Descrição |
|---|---------|-----------|
| 1 | [00-index.md](09-whats-new/00-index.md) | Índice das notas de versão |
| 2 | [01-2026-w13.md](09-whats-new/01-2026-w13.md) | Semana 13 (23–27 março 2026) |
| 3 | [02-2026-w14.md](09-whats-new/02-2026-w14.md) | Semana 14 (30 março – 3 abril 2026) |

---

## Uso com LLMs

Esta documentação está em formato markdown puro, ideal para ingestão por LLMs. Para carregar tudo de uma vez em um contexto de LLM, você pode concatenar os arquivos:

```bash
# Concatenar toda a documentação em um único arquivo
find claude-code-docs -name "*.md" -not -name "README.md" | sort | xargs cat > claude-code-docs-full.md

# Ou por seção específica
cat claude-code-docs/02-core-concepts/*.md > core-concepts.md
```

## Fonte Original

Documentação oficial: https://code.claude.com/docs/en/overview
Índice completo (llms.txt): https://code.claude.com/docs/llms.txt
