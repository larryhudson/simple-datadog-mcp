# simple-datadog-mcp

A minimal MCP server that queries Datadog logs and LLM traces from URLs.

## What it does

Paste a Datadog Log Explorer or LLM Observability URL and get the data back. Useful for investigating alerts, debugging LLM agent traces, or sharing queries with AI assistants.

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) or [Poetry](https://python-poetry.org/)
- Datadog API credentials (see below)

### Getting Datadog credentials

Get your API keys from the [Datadog API Keys page](https://docs.datadoghq.com/account_management/api-app-keys/):

- **DD_API_KEY**: Organization Settings → API Keys → "New Key"
- **DD_APP_KEY**: Organization Settings → Application Keys → "New Application Key"

When creating the Application Key, you need to add the following scopes:
- `logs_read_data` - for querying logs
- `llm_observability_read` - for querying LLM traces

### Add to Claude Code

**With Poetry:**

```bash
cd /path/to/simple-datadog-mcp
poetry install

claude mcp add --transport stdio \
  -e DD_API_KEY=your_api_key \
  -e DD_APP_KEY=your_app_key \
  datadog-logs \
  -- poetry run -C /path/to/simple-datadog-mcp simple-datadog-mcp
```

**With uv:**

```bash
claude mcp add --transport stdio \
  -e DD_API_KEY=your_api_key \
  -e DD_APP_KEY=your_app_key \
  datadog-logs \
  -- uv run --directory /path/to/simple-datadog-mcp simple-datadog-mcp
```


Replace `/path/to/simple-datadog-mcp` with the actual path to this repository.

## Usage

Once configured, you can ask Claude to query data by pasting a Datadog URL:

**Logs:**
```
Can you check the logs from this URL?
https://app.datadoghq.eu/logs?query=service:my-service%20status:error&from_ts=1234567890000&to_ts=1234567899000
```

**LLM Traces:**
```
Can you show me the details of this LLM trace?
https://app.datadoghq.eu/llm/traces?query=@ml_app:my-agent&spanId=12345&start=1234567890000&end=1234567899000
```

The server extracts the query, time range, and site from the URL and returns the matching data.

## Tools

### `query_logs_from_url`

Query Datadog logs using a Log Explorer URL.

- **url**: A Datadog Log Explorer URL
- **limit**: Maximum logs to return (default: 100, max: 1000)

Returns JSON with the query details and log entries including timestamps, messages, services, and attributes.

### `query_llm_traces_from_url`

Query LLM Observability traces using a URL from the LLM Traces Explorer.

- **url**: A Datadog LLM traces URL
- **limit**: Maximum spans to return (default: 100, max: 1000)

Returns JSON with LLM spans including ml_app, span_kind, model info, inputs/outputs, and metrics.

### `get_llm_trace_details`

Get detailed conversation flow for an LLM trace, including tool calls.

- **url**: A Datadog LLM traces URL (with a specific spanId)

Returns a concise conversation flow showing:
- User request
- LLM calls (with model, tokens, response)
- Tool calls (with inputs and outputs)

This is useful for debugging LLM agent behavior and understanding the full request/response flow.
