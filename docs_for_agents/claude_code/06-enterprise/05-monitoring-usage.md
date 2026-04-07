# Monitoring

> Learn how to enable and configure OpenTelemetry for Claude Code.

Track Claude Code usage, costs, and tool activity across your organization by exporting telemetry data through OpenTelemetry (OTel). Claude Code exports metrics as time series data via the standard metrics protocol, events via the logs/events protocol, and optionally distributed traces via the [traces protocol](#traces-beta). Configure your metrics, logs, and traces backends to match your monitoring requirements.

## Quick start

Configure OpenTelemetry using environment variables:

```bash  theme={null}
# 1. Enable telemetry
export CLAUDE_CODE_ENABLE_TELEMETRY=1

# 2. Choose exporters (both are optional - configure only what you need)
export OTEL_METRICS_EXPORTER=otlp       # Options: otlp, prometheus, console, none
export OTEL_LOGS_EXPORTER=otlp          # Options: otlp, console, none

# 3. Configure OTLP endpoint (for OTLP exporter)
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# 4. Set authentication (if required)
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"

# 5. For debugging: reduce export intervals
export OTEL_METRIC_EXPORT_INTERVAL=10000  # 10 seconds (default: 60000ms)
export OTEL_LOGS_EXPORT_INTERVAL=5000     # 5 seconds (default: 5000ms)

# 6. Run Claude Code
claude
```

<Note>
  The default export intervals are 60 seconds for metrics and 5 seconds for logs. During setup, you may want to use shorter intervals for debugging purposes. Remember to reset these for production use.
</Note>

For full configuration options, see the [OpenTelemetry specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#configuration-options).

## Administrator configuration

Administrators can configure OpenTelemetry settings for all users through the [managed settings file](/en/settings#settings-files). This allows for centralized control of telemetry settings across an organization. See the [settings precedence](/en/settings#settings-precedence) for more information about how settings are applied.

Example managed settings configuration:

```json  theme={null}
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector.example.com:4317",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer example-token"
  }
}
```

<Note>
  Managed settings can be distributed via MDM (Mobile Device Management) or other device management solutions. Environment variables defined in the managed settings file have high precedence and cannot be overridden by users.
</Note>

## Configuration details

### Common configuration variables

| Environment Variable                                | Description                                                                                                                                                     | Example Values                          |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `CLAUDE_CODE_ENABLE_TELEMETRY`                      | Enables telemetry collection (required)                                                                                                                         | `1`                                     |
| `OTEL_METRICS_EXPORTER`                             | Metrics exporter types, comma-separated. Use `none` to disable                                                                                                  | `console`, `otlp`, `prometheus`, `none` |
| `OTEL_LOGS_EXPORTER`                                | Logs/events exporter types, comma-separated. Use `none` to disable                                                                                              | `console`, `otlp`, `none`               |
| `OTEL_EXPORTER_OTLP_PROTOCOL`                       | Protocol for OTLP exporter, applies to all signals                                                                                                              | `grpc`, `http/json`, `http/protobuf`    |
| `OTEL_EXPORTER_OTLP_ENDPOINT`                       | OTLP collector endpoint for all signals                                                                                                                         | `http://localhost:4317`                 |
| `OTEL_EXPORTER_OTLP_METRICS_PROTOCOL`               | Protocol for metrics, overrides general setting                                                                                                                 | `grpc`, `http/json`, `http/protobuf`    |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`               | OTLP metrics endpoint, overrides general setting                                                                                                                | `http://localhost:4318/v1/metrics`      |
| `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL`                  | Protocol for logs, overrides general setting                                                                                                                    | `grpc`, `http/json`, `http/protobuf`    |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`                  | OTLP logs endpoint, overrides general setting                                                                                                                   | `http://localhost:4318/v1/logs`         |
| `OTEL_EXPORTER_OTLP_HEADERS`                        | Authentication headers for OTLP                                                                                                                                 | `Authorization=Bearer token`            |
| `OTEL_EXPORTER_OTLP_METRICS_CLIENT_KEY`             | Client key for mTLS authentication                                                                                                                              | Path to client key file                 |
| `OTEL_EXPORTER_OTLP_METRICS_CLIENT_CERTIFICATE`     | Client certificate for mTLS authentication                                                                                                                      | Path to client cert file                |
| `OTEL_METRIC_EXPORT_INTERVAL`                       | Export interval in milliseconds (default: 60000)                                                                                                                | `5000`, `60000`                         |
| `OTEL_LOGS_EXPORT_INTERVAL`                         | Logs export interval in milliseconds (default: 5000)                                                                                                            | `1000`, `10000`                         |
| `OTEL_LOG_USER_PROMPTS`                             | Enable logging of user prompt content (default: disabled)                                                                                                       | `1` to enable                           |
| `OTEL_LOG_TOOL_DETAILS`                             | Enable logging of tool parameters and input arguments in tool events: Bash commands, MCP server and tool names, skill names, and tool input (default: disabled) | `1` to enable                           |
| `OTEL_LOG_TOOL_CONTENT`                             | Enable logging of tool input and output content in span events (default: disabled). Requires [tracing](#traces-beta). Content is truncated at 60 KB             | `1` to enable                           |
| `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE` | Metrics temporality preference (default: `delta`). Set to `cumulative` if your backend expects cumulative temporality                                           | `delta`, `cumulative`                   |
| `CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS`       | Interval for refreshing dynamic headers (default: 1740000ms / 29 minutes)                                                                                       | `900000`                                |

### Metrics cardinality control

The following environment variables control which attributes are included in metrics to manage cardinality:

| Environment Variable                | Description                                                           | Default Value | Example to Disable |
| ----------------------------------- | --------------------------------------------------------------------- | ------------- | ------------------ |
| `OTEL_METRICS_INCLUDE_SESSION_ID`   | Include session.id attribute in metrics                               | `true`        | `false`            |
| `OTEL_METRICS_INCLUDE_VERSION`      | Include app.version attribute in metrics                              | `false`       | `true`             |
| `OTEL_METRICS_INCLUDE_ACCOUNT_UUID` | Include user.account\_uuid and user.account\_id attributes in metrics | `true`        | `false`            |

These variables help control the cardinality of metrics, which affects storage requirements and query performance in your metrics backend. Lower cardinality generally means better performance and lower storage costs but less granular data for analysis.

### Traces (beta)

Distributed tracing exports spans that link each user prompt to the API requests and tool executions it triggers, so you can view a full request as a single trace in your tracing backend.

Tracing is off by default. To enable it, set both `CLAUDE_CODE_ENABLE_TELEMETRY=1` and `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, then set `OTEL_TRACES_EXPORTER` to choose where spans are sent. Traces reuse the [common OTLP configuration](#common-configuration-variables) for endpoint, protocol, and headers.

| Environment Variable                  | Description                                                                       | Example Values                       |
| ------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------ |
| `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA` | Enable span tracing (required). `ENABLE_ENHANCED_TELEMETRY_BETA` is also accepted | `1`                                  |
| `OTEL_TRACES_EXPORTER`                | Traces exporter types, comma-separated. Use `none` to disable                     | `console`, `otlp`, `none`            |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`  | Protocol for traces, overrides `OTEL_EXPORTER_OTLP_PROTOCOL`                      | `grpc`, `http/json`, `http/protobuf` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`  | OTLP traces endpoint, overrides `OTEL_EXPORTER_OTLP_ENDPOINT`                     | `http://localhost:4318/v1/traces`    |
| `OTEL_TRACES_EXPORT_INTERVAL`         | Span batch export interval in milliseconds (default: 5000)                        | `1000`, `10000`                      |

Spans redact user prompt text and tool content by default. Set `OTEL_LOG_USER_PROMPTS=1` and `OTEL_LOG_TOOL_CONTENT=1` to include them.

### Dynamic headers

For enterprise environments that require dynamic authentication, you can configure a script to generate headers dynamically:

#### Settings configuration

Add to your `.claude/settings.json`:

```json  theme={null}
{
  "otelHeadersHelper": "/bin/generate_opentelemetry_headers.sh"
}
```

#### Script requirements

The script must output valid JSON with string key-value pairs representing HTTP headers:

```bash  theme={null}
#!/bin/bash
# Example: Multiple headers
echo "{\"Authorization\": \"Bearer $(get-token.sh)\", \"X-API-Key\": \"$(get-api-key.sh)\"}"
```

#### Refresh behavior

The headers helper script runs at startup and periodically thereafter to support token refresh. By default, the script runs every 29 minutes. Customize the interval with the `CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS` environment variable.

### Multi-team organization support

Organizations with multiple teams or departments can add custom attributes to distinguish between different groups using the `OTEL_RESOURCE_ATTRIBUTES` environment variable:

```bash  theme={null}
# Add custom attributes for team identification
export OTEL_RESOURCE_ATTRIBUTES="department=engineering,team.id=platform,cost_center=eng-123"
```

These custom attributes will be included in all metrics and events, allowing you to:

* Filter metrics by team or department
* Track costs per cost center
* Create team-specific dashboards
* Set up alerts for specific teams

<Warning>
  **Important formatting requirements for OTEL\_RESOURCE\_ATTRIBUTES:**

  The `OTEL_RESOURCE_ATTRIBUTES` environment variable uses comma-separated key=value pairs with strict formatting requirements:

  * **No spaces allowed**: Values cannot contain spaces. For example, `user.organizationName=My Company` is invalid
  * **Format**: Must be comma-separated key=value pairs: `key1=value1,key2=value2`
  * **Allowed characters**: Only US-ASCII characters excluding control characters, whitespace, double quotes, commas, semicolons, and backslashes
  * **Special characters**: Characters outside the allowed range must be percent-encoded

  **Examples:**

  ```bash  theme={null}
  # Invalid - contains spaces
  export OTEL_RESOURCE_ATTRIBUTES="org.name=John's Organization"

  # Valid - use underscores or camelCase instead
  export OTEL_RESOURCE_ATTRIBUTES="org.name=Johns_Organization"
  export OTEL_RESOURCE_ATTRIBUTES="org.name=JohnsOrganization"

  # Valid - percent-encode special characters if needed
  export OTEL_RESOURCE_ATTRIBUTES="org.name=John%27s%20Organization"
  ```

  Note: wrapping values in quotes doesn't escape spaces. For example, `org.name="My Company"` results in the literal value `"My Company"` (with quotes included), not `My Company`.
</Warning>

### Example configurations

Set these environment variables before running `claude`. Each block shows a complete configuration for a different exporter or deployment scenario:

```bash  theme={null}
# Console debugging (1-second intervals)
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=console
export OTEL_METRIC_EXPORT_INTERVAL=1000

# OTLP/gRPC
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Prometheus
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=prometheus

# Multiple exporters
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=console,otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json

# Different endpoints/backends for metrics and logs
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_METRICS_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://metrics.example.com:4318
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://logs.example.com:4317

# Metrics only (no events/logs)
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Events/logs only (no metrics)
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Available metrics and events

### Standard attributes

All metrics and events share these standard attributes:

| Attribute           | Description                                                                                                 | Controlled By                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `session.id`        | Unique session identifier                                                                                   | `OTEL_METRICS_INCLUDE_SESSION_ID` (default: true)   |
| `app.version`       | Current Claude Code version                                                                                 | `OTEL_METRICS_INCLUDE_VERSION` (default: false)     |
| `organization.id`   | Organization UUID (when authenticated)                                                                      | Always included when available                      |
| `user.account_uuid` | Account UUID (when authenticated)                                                                           | `OTEL_METRICS_INCLUDE_ACCOUNT_UUID` (default: true) |
| `user.account_id`   | Account ID in tagged format matching Anthropic admin APIs (when authenticated), such as `user_01BWBeN28...` | `OTEL_METRICS_INCLUDE_ACCOUNT_UUID` (default: true) |
| `user.id`           | Anonymous device/installation identifier, generated per Claude Code installation                            | Always included                                     |
| `user.email`        | User email address (when authenticated via OAuth)                                                           | Always included when available                      |
| `terminal.type`     | Terminal type, such as `iTerm.app`, `vscode`, `cursor`, or `tmux`                                           | Always included when detected                       |

Events additionally include the following attributes. These are never attached to metrics because they would cause unbounded cardinality:

* `prompt.id`: UUID correlating a user prompt with all subsequent events until the next prompt. See [Event correlation attributes](#event-correlation-attributes).
* `workspace.host_paths`: host workspace directories selected in the desktop app, as a string array

### Metrics

Claude Code exports the following metrics:

| Metric Name                           | Description                                     | Unit   |
| ------------------------------------- | ----------------------------------------------- | ------ |
| `claude_code.session.count`           | Count of CLI sessions started                   | count  |
| `claude_code.lines_of_code.count`     | Count of lines of code modified                 | count  |
| `claude_code.pull_request.count`      | Number of pull requests created                 | count  |
| `claude_code.commit.count`            | Number of git commits created                   | count  |
| `claude_code.cost.usage`              | Cost of the Claude Code session                 | USD    |
| `claude_code.token.usage`             | Number of tokens used                           | tokens |
| `claude_code.code_edit_tool.decision` | Count of code editing tool permission decisions | count  |
| `claude_code.active_time.total`       | Total active time in seconds                    | s      |

### Metric details

Each metric includes the standard attributes listed above. Metrics with additional context-specific attributes are noted below.

#### Session counter

Incremented at the start of each session.

**Attributes**:

* All [standard attributes](#standard-attributes)

#### Lines of code counter

Incremented when code is added or removed.

**Attributes**:

* All [standard attributes](#standard-attributes)
* `type`: (`"added"`, `"removed"`)

#### Pull request counter

Incremented when creating pull requests via Claude Code.

**Attributes**:

* All [standard attributes](#standard-attributes)

#### Commit counter

Incremented when creating git commits via Claude Code.

**Attributes**:

* All [standard attributes](#standard-attributes)

#### Cost counter

Incremented after each API request.

**Attributes**:

* All [standard attributes](#standard-attributes)
* `model`: Model identifier (for example, "claude-sonnet-4-6")

#### Token counter

Incremented after each API request.

**Attributes**:

* All [standard attributes](#standard-attributes)
* `type`: (`"input"`, `"output"`, `"cacheRead"`, `"cacheCreation"`)
* `model`: Model identifier (for example, "claude-sonnet-4-6")

#### Code edit tool decision counter

Incremented when user accepts or rejects Edit, Write, or NotebookEdit tool usage.

**Attributes**:

* All [standard attributes](#standard-attributes)
* `tool_name`: Tool name (`"Edit"`, `"Write"`, `"NotebookEdit"`)
* `decision`: User decision (`"accept"`, `"reject"`)
* `source`: Decision source - `"config"`, `"hook"`, `"user_permanent"`, `"user_temporary"`, `"user_abort"`, or `"user_reject"`
* `language`: Programming language of the edited file, such as `"TypeScript"`, `"Python"`, `"JavaScript"`, or `"Markdown"`. Returns `"unknown"` for unrecognized file extensions.

#### Active time counter

Tracks actual time spent actively using Claude Code, excluding idle time. This metric is incremented during user interactions (typing, reading responses) and during CLI processing (tool execution, AI response generation).

**Attributes**:

* All [standard attributes](#standard-attributes)
* `type`: `"user"` for keyboard interactions, `"cli"` for tool execution and AI responses

### Events

Claude Code exports the following events via OpenTelemetry logs/events (when `OTEL_LOGS_EXPORTER` is configured):

#### Event correlation attributes

When a user submits a prompt, Claude Code may make multiple API calls and run several tools. The `prompt.id` attribute lets you tie all of those events back to the single prompt that triggered them.

| Attribute   | Description                                                                          |
| ----------- | ------------------------------------------------------------------------------------ |
| `prompt.id` | UUID v4 identifier linking all events produced while processing a single user prompt |

To trace all activity triggered by a single prompt, filter your events by a specific `prompt.id` value. This returns the user\_prompt event, any api\_request events, and any tool\_result events that occurred while processing that prompt.

<Note>
  `prompt.id` is intentionally excluded from metrics because each prompt generates a unique ID, which would create an ever-growing number of time series. Use it for event-level analysis and audit trails only.
</Note>

#### User prompt event

Logged when a user submits a prompt.

**Event Name**: `claude_code.user_prompt`

**Attributes**:

* All [standard attributes](#standard-attributes)
* `event.name`: `"user_prompt"`
* `event.timestamp`: ISO 8601 timestamp
* `event.sequence`: monotonically increasing counter for ordering events within a session
* `prompt_length`: Length of the prompt
* `prompt`: Prompt content (redacted by default, enable with `OTEL_LOG_USER_PROMPTS=1`)

#### Tool result event

Logged when a tool completes execution.

**Event Name**: `claude_code.tool_result`

**Attributes**:

* All [standard attributes](#standard-attributes)
* `event.name`: `"tool_result"`
* `event.timestamp`: ISO 8601 timestamp
* `event.sequence`: monotonically increasing counter for ordering events within a session
* `tool_name`: Name of the tool
* `success`: `"true"` or `"false"`
* `duration_ms`: Execution time in milliseconds
* `error`: Error message (if failed)
* `decision_type`: Either `"accept"` or `"reject"`
* `decision_source`: Decision source - `"config"`, `"hook"`, `"user_permanent"`, `"user_temporary"`, `"user_abort"`, or `"user_reject"`
* `tool_result_size_bytes`: Size of the tool result in bytes
* `mcp_server_scope`: MCP server scope identifier (for MCP tools)
* `tool_parameters` (when `OTEL_LOG_TOOL_DETAILS=1`): JSON string containing tool-specific parameters:
  * For Bash tool: includes `bash_command`, `full_command`, `timeout`, `description`, `dangerouslyDisableSandbox`, and `git_commit_id` (the commit SHA, when a `git commit` command succeeds)
  * For MCP tools: includes `mcp_server_name`, `mcp_tool_name`
  * For Skill tool: includes `skill_name`
* `tool_input` (when `OTEL_LOG_TOOL_DETAILS=1`): JSON-serialized tool arguments. Individual values over 512 characters are truncated, and the full payload is bounded to \~4 K characters. Applies to all tools including MCP tools.

#### API request event

Logged for each API request to Claude.

**Event Name**: `claude_code.api_request`

**Attributes**:

* All [standard attributes](#standard-attributes)
* `event.name`: `"api_request"`
* `event.timestamp`: ISO 8601 timestamp
* `event.sequence`: monotonically increasing counter for ordering events within a session
* `model`: Model used (for example, "claude-sonnet-4-6")
* `cost_usd`: Estimated cost in USD
* `duration_ms`: Request duration in milliseconds
* `input_tokens`: Number of input tokens
* `output_tokens`: Number of output tokens
* `cache_read_tokens`: Number of tokens read from cache
* `cache_creation_tokens`: Number of tokens used for cache creation
* `speed`: `"fast"` or `"normal"`, indicating whether fast mode was active

#### API error event

Logged when an API request to Claude fails.

**Event Name**: `claude_code.api_error`

**Attributes**:

* All [standard attributes](#standard-attributes)
* `event.name`: `"api_error"`
* `event.timestamp`: ISO 8601 timestamp
* `event.sequence`: monotonically increasing counter for ordering events within a session
* `model`: Model used (for example, "claude-sonnet-4-6")
* `error`: Error message
* `status_code`: HTTP status code as a string, or `"undefined"` for non-HTTP errors
* `duration_ms`: Request duration in milliseconds
* `attempt`: Attempt number (for retried requests)
* `speed`: `"fast"` or `"normal"`, indicating whether fast mode was active

#### Tool decision event

Logged when a tool permission decision is made (accept/reject).

**Event Name**: `claude_code.tool_decision`

**Attributes**:

* All [standard attributes](#standard-attributes)
* `event.name`: `"tool_decision"`
* `event.timestamp`: ISO 8601 timestamp
* `event.sequence`: monotonically increasing counter for ordering events within a session
* `tool_name`: Name of the tool (for example, "Read", "Edit", "Write", "NotebookEdit")
* `decision`: Either `"accept"` or `"reject"`
* `source`: Decision source - `"config"`, `"hook"`, `"user_permanent"`, `"user_temporary"`, `"user_abort"`, or `"user_reject"`

## Interpret metrics and events data

The exported metrics and events support a range of analyses:

### Usage monitoring

| Metric                                                        | Analysis Opportunity                                      |
| ------------------------------------------------------------- | --------------------------------------------------------- |
| `claude_code.token.usage`                                     | Break down by `type` (input/output), user, team, or model |
| `claude_code.session.count`                                   | Track adoption and engagement over time                   |
| `claude_code.lines_of_code.count`                             | Measure productivity by tracking code additions/removals  |
| `claude_code.commit.count` & `claude_code.pull_request.count` | Understand impact on development workflows                |

### Cost monitoring

The `claude_code.cost.usage` metric helps with:

* Tracking usage trends across teams or individuals
* Identifying high-usage sessions for optimization

<Note>
  Cost metrics are approximations. For official billing data, refer to your API provider (Claude Console, AWS Bedrock, or Google Cloud Vertex).
</Note>

### Alerting and segmentation

Common alerts to consider:

* Cost spikes
* Unusual token consumption
* High session volume from specific users

All metrics can be segmented by `user.account_uuid`, `user.account_id`, `organization.id`, `session.id`, `model`, and `app.version`.

### Event analysis

The event data provides detailed insights into Claude Code interactions:

**Tool Usage Patterns**: analyze tool result events to identify:

* Most frequently used tools
* Tool success rates
* Average tool execution times
* Error patterns by tool type

**Performance Monitoring**: track API request durations and tool execution times to identify performance bottlenecks.

## Backend considerations

Your choice of metrics, logs, and traces backends determines the types of analyses you can perform:

### For metrics

* **Time series databases (for example, Prometheus)**: Rate calculations, aggregated metrics
* **Columnar stores (for example, ClickHouse)**: Complex queries, unique user analysis
* **Full-featured observability platforms (for example, Honeycomb, Datadog)**: Advanced querying, visualization, alerting

### For events/logs

* **Log aggregation systems (for example, Elasticsearch, Loki)**: Full-text search, log analysis
* **Columnar stores (for example, ClickHouse)**: Structured event analysis
* **Full-featured observability platforms (for example, Honeycomb, Datadog)**: Correlation between metrics and events

### For traces

Choose a backend that supports distributed trace storage and span correlation:

* **Distributed tracing systems (for example, Jaeger, Zipkin, Grafana Tempo)**: Span visualization, request waterfalls, latency analysis
* **Full-featured observability platforms (for example, Honeycomb, Datadog)**: Trace search and correlation with metrics and logs

For organizations requiring Daily/Weekly/Monthly Active User (DAU/WAU/MAU) metrics, consider backends that support efficient unique value queries.

## Service information

All metrics and events are exported with the following resource attributes:

* `service.name`: `claude-code`
* `service.version`: Current Claude Code version
* `os.type`: Operating system type (for example, `linux`, `darwin`, `windows`)
* `os.version`: Operating system version string
* `host.arch`: Host architecture (for example, `amd64`, `arm64`)
* `wsl.version`: WSL version number (only present when running on Windows Subsystem for Linux)
* Meter Name: `com.anthropic.claude_code`

## ROI measurement resources

For a comprehensive guide on measuring return on investment for Claude Code, including telemetry setup, cost analysis, productivity metrics, and automated reporting, see the [Claude Code ROI Measurement Guide](https://github.com/anthropics/claude-code-monitoring-guide). This repository provides ready-to-use Docker Compose configurations, Prometheus and OpenTelemetry setups, and templates for generating productivity reports integrated with tools like Linear.

## Security and privacy

* Telemetry is opt-in and requires explicit configuration
* Raw file contents and code snippets are not included in metrics or events. Trace spans are a separate data path: see the `OTEL_LOG_TOOL_CONTENT` bullet below
* When authenticated via OAuth, `user.email` is included in telemetry attributes. If this is a concern for your organization, work with your telemetry backend to filter or redact this field
* User prompt content is not collected by default. Only prompt length is recorded. To include prompt content, set `OTEL_LOG_USER_PROMPTS=1`
* Tool input arguments and parameters are not logged by default. To include them, set `OTEL_LOG_TOOL_DETAILS=1`. When enabled, `tool_result` events include a `tool_parameters` attribute with Bash commands, MCP server and tool names, and skill names, plus a `tool_input` attribute with file paths, URLs, search patterns, and other arguments. Individual values over 512 characters are truncated and the total is bounded to \~4 K characters, but the arguments may still contain sensitive values. Configure your telemetry backend to filter or redact these attributes as needed
* Tool input and output content is not logged in trace spans by default. To include it, set `OTEL_LOG_TOOL_CONTENT=1`. When enabled, span events include full tool input and output content truncated at 60 KB per span. This can include raw file contents from Read tool results and Bash command output. Configure your telemetry backend to filter or redact these attributes as needed

## Monitor Claude Code on Amazon Bedrock

For detailed Claude Code usage monitoring guidance for Amazon Bedrock, see [Claude Code Monitoring Implementation (Bedrock)](https://github.com/aws-solutions-library-samples/guidance-for-claude-code-with-amazon-bedrock/blob/main/assets/docs/MONITORING.md).
