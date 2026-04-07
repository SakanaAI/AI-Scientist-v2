# Claude Code on Microsoft Foundry

> Learn about configuring Claude Code through Microsoft Foundry, including setup, configuration, and troubleshooting.

## Prerequisites

Before configuring Claude Code with Microsoft Foundry, ensure you have:

* An Azure subscription with access to Microsoft Foundry
* RBAC permissions to create Microsoft Foundry resources and deployments
* Azure CLI installed and configured (optional - only needed if you don't have another mechanism for getting credentials)

<Note>
  If you are deploying Claude Code to multiple users, [pin your model versions](#4-pin-model-versions) to prevent breakage when Anthropic releases new models.
</Note>

## Setup

### 1. Provision Microsoft Foundry resource

First, create a Claude resource in Azure:

1. Navigate to the [Microsoft Foundry portal](https://ai.azure.com/)
2. Create a new resource, noting your resource name
3. Create deployments for the Claude models:
   * Claude Opus
   * Claude Sonnet
   * Claude Haiku

### 2. Configure Azure credentials

Claude Code supports two authentication methods for Microsoft Foundry. Choose the method that best fits your security requirements.

**Option A: API key authentication**

1. Navigate to your resource in the Microsoft Foundry portal
2. Go to the **Endpoints and keys** section
3. Copy **API Key**
4. Set the environment variable:

```bash  theme={null}
export ANTHROPIC_FOUNDRY_API_KEY=your-azure-api-key
```

**Option B: Microsoft Entra ID authentication**

When `ANTHROPIC_FOUNDRY_API_KEY` is not set, Claude Code automatically uses the Azure SDK [default credential chain](https://learn.microsoft.com/en-us/azure/developer/javascript/sdk/authentication/credential-chains#defaultazurecredential-overview).
This supports a variety of methods for authenticating local and remote workloads.

On local environments, you commonly may use the Azure CLI:

```bash  theme={null}
az login
```

<Note>
  When using Microsoft Foundry, the `/login` and `/logout` commands are disabled since authentication is handled through Azure credentials.
</Note>

### 3. Configure Claude Code

Set the following environment variables to enable Microsoft Foundry:

```bash  theme={null}
# Enable Microsoft Foundry integration
export CLAUDE_CODE_USE_FOUNDRY=1

# Azure resource name (replace {resource} with your resource name)
export ANTHROPIC_FOUNDRY_RESOURCE={resource}
# Or provide the full base URL:
# export ANTHROPIC_FOUNDRY_BASE_URL=https://{resource}.services.ai.azure.com/anthropic
```

### 4. Pin model versions

<Warning>
  Pin specific model versions for every deployment. If you use model aliases (`sonnet`, `opus`, `haiku`) without pinning, Claude Code may attempt to use a newer model version that isn't available in your Foundry account, breaking existing users when Anthropic releases updates. When you create Azure deployments, select a specific model version rather than "auto-update to latest."
</Warning>

Set the model variables to match the deployment names you created in step 1:

```bash  theme={null}
export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='claude-haiku-4-5'
```

For current and legacy model IDs, see [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview). See [Model configuration](/en/model-config#pin-models-for-third-party-deployments) for the full list of environment variables.

## Azure RBAC configuration

The `Azure AI User` and `Cognitive Services User` default roles include all required permissions for invoking Claude models.

For more restrictive permissions, create a custom role with the following:

```json  theme={null}
{
  "permissions": [
    {
      "dataActions": [
        "Microsoft.CognitiveServices/accounts/providers/*"
      ]
    }
  ]
}
```

For details, see [Microsoft Foundry RBAC documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/rbac-azure-ai-foundry).

## Troubleshooting

If you receive an error "Failed to get token from azureADTokenProvider: ChainedTokenCredential authentication failed":

* Configure Entra ID on the environment, or set `ANTHROPIC_FOUNDRY_API_KEY`.

## Additional resources

* [Microsoft Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/what-is-azure-ai-foundry)
* [Microsoft Foundry models](https://ai.azure.com/explore/models)
* [Microsoft Foundry pricing](https://azure.microsoft.com/en-us/pricing/details/ai-foundry/)
