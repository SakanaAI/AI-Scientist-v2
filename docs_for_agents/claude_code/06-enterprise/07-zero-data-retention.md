# Zero data retention

> Learn about Zero Data Retention (ZDR) for Claude Code on Claude for Enterprise, including scope, disabled features, and how to request enablement.

Zero Data Retention (ZDR) is available for Claude Code when used through Claude for Enterprise. When ZDR is enabled, prompts and model responses generated during Claude Code sessions are processed in real time and not stored by Anthropic after the response is returned, except where needed to comply with law or combat misuse.

ZDR on Claude for Enterprise gives enterprise customers the ability to use Claude Code with zero data retention and access administrative capabilities:

* Cost controls per user
* [Analytics](/en/analytics) dashboard
* [Server-managed settings](/en/server-managed-settings)
* Audit logs

ZDR for Claude Code on Claude for Enterprise applies only to Anthropic's direct platform. For Claude deployments on AWS Bedrock, Google Vertex AI, or Microsoft Foundry, refer to those platforms' data retention policies.

## ZDR scope

ZDR covers Claude Code inference on Claude for Enterprise.

<Warning>
  ZDR is enabled on a per-organization basis. Each new organization requires ZDR to be enabled separately by your Anthropic account team. ZDR does not automatically apply to new organizations created under the same account. Contact your account team to enable ZDR for any new organizations.
</Warning>

### What ZDR covers

ZDR covers model inference calls made through Claude Code on Claude for Enterprise. When you use Claude Code in your terminal, the prompts you send and the responses Claude generates are not retained by Anthropic. This applies regardless of which Claude model is used.

### What ZDR does not cover

ZDR does not extend to the following, even for organizations with ZDR enabled. These features follow [standard data retention policies](/en/data-usage#data-retention):

| Feature                  | Details                                                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat on claude.ai        | Chat conversations through the Claude for Enterprise web interface are not covered by ZDR.                                                                                                                                                                  |
| Cowork                   | Cowork sessions are not covered by ZDR.                                                                                                                                                                                                                     |
| Claude Code Analytics    | Does not store prompts or model responses, but collects productivity metadata such as account emails and usage statistics. Contribution metrics are not available for ZDR organizations; the [analytics dashboard](/en/analytics) shows usage metrics only. |
| User and seat management | Administrative data such as account emails and seat assignments is retained under standard policies.                                                                                                                                                        |
| Third-party integrations | Data processed by third-party tools, MCP servers, or other external integrations is not covered by ZDR. Review those services' data handling practices independently.                                                                                       |

## Features disabled under ZDR

When ZDR is enabled for a Claude Code organization on Claude for Enterprise, certain features that require storing prompts or completions are automatically disabled at the backend level:

| Feature                                                             | Reason                                                                  |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Claude Code on the Web](/en/claude-code-on-the-web)                | Requires server-side storage of conversation history.                   |
| [Remote sessions](/en/desktop#remote-sessions) from the Desktop app | Requires persistent session data that includes prompts and completions. |
| Feedback submission (`/feedback`)                                   | Submitting feedback sends conversation data to Anthropic.               |

These features are blocked in the backend regardless of client-side display. If you see a disabled feature in the Claude Code terminal during startup, attempting to use it returns an error indicating the organization's policies do not allow that action.

Future features may also be disabled if they require storing prompts or completions.

## Data retention for policy violations

Even with ZDR enabled, Anthropic may retain data where required by law or to address Usage Policy violations. If a session is flagged for a policy violation, Anthropic may retain the associated inputs and outputs for up to 2 years, consistent with Anthropic's standard ZDR policy.

## Request ZDR

To request ZDR for Claude Code on Claude for Enterprise, contact your Anthropic account team. Your account team will submit the request internally, and Anthropic will review and enable ZDR on your organization after confirming eligibility. All enablement actions are audit-logged.

If you are currently using ZDR for Claude Code via pay-as-you-go API keys, you can transition to Claude for Enterprise to gain access to administrative features while maintaining ZDR for Claude Code. Contact your account team to coordinate the migration.
