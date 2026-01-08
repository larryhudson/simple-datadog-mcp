# simple-datadog-mcp

A minimal MCP server that queries Datadog logs from Log Explorer URLs.

## What it does

Paste a Datadog Log Explorer URL and get the logs back. Useful for investigating alerts or sharing log queries with AI assistants.

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) or [Poetry](https://python-poetry.org/)
- Datadog API credentials (see below)

### Getting Datadog credentials

Get your API keys from the [Datadog API Keys page](https://docs.datadoghq.com/account_management/api-app-keys/):

- **DD_API_KEY**: Organization Settings → API Keys → "New Key"
- **DD_APP_KEY**: Organization Settings → Application Keys → "New Application Key"

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

Once configured, you can ask Claude to query logs by pasting a Datadog URL:

```
Can you check the logs from this URL?
https://app.datadoghq.eu/logs?query=service:my-service%20status:error&from_ts=1234567890000&to_ts=1234567899000
```

The server extracts the query, time range, and site from the URL and returns the matching logs.

## Tool

### `query_logs_from_url`

- **url**: A Datadog Log Explorer URL
- **limit**: Maximum logs to return (default: 100, max: 1000)

Returns JSON with the query details and log entries including timestamps, messages, services, and attributes.
