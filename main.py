from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime, timezone
import json
import os

import requests
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi

load_dotenv()


mcp = FastMCP("datadog-logs")


def parse_datadog_url(url: str) -> dict:
    """Parse a Datadog logs URL and extract query parameters."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    # Determine the site from the hostname (e.g., app.datadoghq.eu -> datadoghq.eu)
    host = parsed.netloc
    if host.startswith("app."):
        site = host[4:]  # Remove "app." prefix
    else:
        site = host

    result = {
        "site": site,
        "query": unquote(params.get("query", ["*"])[0]),
        "from_ts": params.get("from_ts", [None])[0],
        "to_ts": params.get("to_ts", [None])[0],
        "index": unquote(params.get("index", ["*"])[0]),
    }

    return result


def ts_to_datetime(ts_ms: str) -> datetime:
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)


@mcp.tool()
def query_logs_from_url(url: str, limit: int = 100) -> str:
    """
    Query Datadog logs using a URL from the Datadog Log Explorer.

    Takes a Datadog logs URL (e.g., from "View in Log Explorer" links in alerts)
    and returns the matching log events with full details including stack traces.

    Args:
        url: A Datadog logs URL from the Log Explorer
        limit: Maximum number of logs to return (default 100, max 1000)

    Returns:
        JSON string containing the log events
    """
    # Parse the URL
    params = parse_datadog_url(url)

    # Configure the Datadog client
    configuration = Configuration()
    configuration.server_variables["site"] = params["site"]

    with ApiClient(configuration) as api_client:
        api = LogsApi(api_client)

        # Build the query parameters
        kwargs = {
            "filter_query": params["query"],
            "page_limit": min(limit, 1000),
        }

        # Add time range if specified
        if params["from_ts"]:
            kwargs["filter_from"] = ts_to_datetime(params["from_ts"])
        if params["to_ts"]:
            kwargs["filter_to"] = ts_to_datetime(params["to_ts"])

        # Add index filter if not wildcard
        if params["index"] and params["index"] != "*":
            kwargs["filter_indexes"] = [params["index"]]

        # Execute the query
        response = api.list_logs_get(**kwargs)

        # Convert response to serializable format
        logs = []
        if response.data:
            for log in response.data:
                attrs = log.attributes.to_dict() if log.attributes else {}
                log_entry = {
                    "id": log.id,
                    "timestamp": str(attrs.get("timestamp", "")),
                    "message": attrs.get("message"),
                    "service": attrs.get("service"),
                    "status": attrs.get("status"),
                    "host": attrs.get("host"),
                    "attributes": attrs.get("attributes", {}),
                    "tags": attrs.get("tags", []),
                }
                logs.append(log_entry)

        return json.dumps(
            {
                "query": params["query"],
                "from": params["from_ts"],
                "to": params["to_ts"],
                "count": len(logs),
                "logs": logs,
            },
            indent=2,
            default=str,
        )


def parse_llm_traces_url(url: str) -> dict:
    """Parse a Datadog LLM traces URL and extract query parameters."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    # Determine the site from the hostname (e.g., app.datadoghq.eu -> datadoghq.eu)
    host = parsed.netloc
    if host.startswith("app."):
        site = host[4:]  # Remove "app." prefix
    else:
        site = host

    # Determine the API host from the site
    api_host = f"api.{site}"

    result = {
        "site": site,
        "api_host": api_host,
        "query": unquote(params.get("query", ["*"])[0]),
        "start": params.get("start", [None])[0],
        "end": params.get("end", [None])[0],
        "span_id": params.get("spanId", [None])[0],
    }

    return result


@mcp.tool()
def query_llm_traces_from_url(url: str, limit: int = 100) -> str:
    """
    Query Datadog LLM Observability traces using a URL from the LLM Traces Explorer.

    Takes a Datadog LLM traces URL (e.g., from the LLM Observability section)
    and returns the matching LLM spans with full details.

    Args:
        url: A Datadog LLM traces URL from the LLM Observability Explorer
        limit: Maximum number of spans to return (default 100)

    Returns:
        JSON string containing the LLM spans
    """
    # Parse the URL
    params = parse_llm_traces_url(url)

    # Get API credentials from environment
    api_key = os.environ.get("DD_API_KEY")
    app_key = os.environ.get("DD_APP_KEY")

    if not api_key or not app_key:
        return json.dumps(
            {"error": "Missing DD_API_KEY or DD_APP_KEY environment variables"},
            indent=2,
        )

    # Build the API URL
    api_url = f"https://{params['api_host']}/api/v2/llm-obs/v1/spans/events/search"

    # Build the request payload
    filter_params = {}

    # Add time range if specified (convert from ms to ISO8601)
    if params["start"]:
        from_dt = datetime.fromtimestamp(int(params["start"]) / 1000, tz=timezone.utc)
        filter_params["from"] = from_dt.isoformat()
    if params["end"]:
        to_dt = datetime.fromtimestamp(int(params["end"]) / 1000, tz=timezone.utc)
        filter_params["to"] = to_dt.isoformat()

    # Add query filter
    if params["query"] and params["query"] != "*":
        filter_params["query"] = params["query"]

    # Add span_id if specified
    if params["span_id"]:
        filter_params["span_id"] = params["span_id"]

    payload = {
        "data": {
            "type": "spans",
            "attributes": {
                "filter": filter_params,
                "page": {"limit": min(limit, 1000)},
            },
        }
    }

    # Make the API request
    headers = {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
        "Content-Type": "application/vnd.api+json",
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        return json.dumps(
            {
                "error": f"API request failed with status {response.status_code}",
                "detail": response.text,
                "request_url": api_url,
                "request_payload": payload,
            },
            indent=2,
        )

    # Parse the response
    data = response.json()

    # Extract spans from response
    spans = []
    if "data" in data:
        for span in data["data"]:
            attrs = span.get("attributes", {})
            spans.append(
                {
                    "id": span.get("id"),
                    "type": span.get("type"),
                    "ml_app": attrs.get("ml_app"),
                    "span_kind": attrs.get("span_kind"),
                    "name": attrs.get("name"),
                    "status": attrs.get("status"),
                    "start_ns": attrs.get("start_ns"),
                    "duration": attrs.get("duration"),
                    "model_name": attrs.get("model_name"),
                    "model_provider": attrs.get("model_provider"),
                    "input": attrs.get("input"),
                    "output": attrs.get("output"),
                    "meta": attrs.get("meta", {}),
                    "metrics": attrs.get("metrics", {}),
                    "tags": attrs.get("tags", []),
                    "trace_id": attrs.get("trace_id"),
                    "span_id": attrs.get("span_id"),
                    "parent_id": attrs.get("parent_id"),
                }
            )

    return json.dumps(
        {
            "query": params["query"],
            "from": params["start"],
            "to": params["end"],
            "count": len(spans),
            "spans": spans,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def get_llm_trace_details(url: str) -> str:
    """
    Get detailed LLM trace with full conversation flow including tool calls.

    Takes a Datadog LLM traces URL and returns a formatted view showing:
    - The original user request
    - LLM calls with their inputs/outputs
    - Tool calls and their results
    - The full conversation flow

    Args:
        url: A Datadog LLM traces URL from the LLM Observability Explorer

    Returns:
        JSON string containing the formatted trace details
    """
    # Parse the URL
    params = parse_llm_traces_url(url)

    # Get API credentials from environment
    api_key = os.environ.get("DD_API_KEY")
    app_key = os.environ.get("DD_APP_KEY")

    if not api_key or not app_key:
        return json.dumps(
            {"error": "Missing DD_API_KEY or DD_APP_KEY environment variables"},
            indent=2,
        )

    # Build the API URL
    api_url = f"https://{params['api_host']}/api/v2/llm-obs/v1/spans/events/search"

    headers = {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
        "Content-Type": "application/vnd.api+json",
    }

    # First, get the root span if span_id is specified
    root_span_id = params.get("span_id")

    # Build filter to get all spans in the time range for this ml_app
    # Remove @parent_id:undefined from query to get nested spans
    query = params["query"]
    # Remove parent_id filter if present
    query_parts = query.split()
    query_parts = [p for p in query_parts if not p.startswith("@parent_id:")]
    filtered_query = " ".join(query_parts) if query_parts else "*"

    filter_params = {"query": filtered_query}

    if params["start"]:
        from_dt = datetime.fromtimestamp(int(params["start"]) / 1000, tz=timezone.utc)
        filter_params["from"] = from_dt.isoformat()
    if params["end"]:
        to_dt = datetime.fromtimestamp(int(params["end"]) / 1000, tz=timezone.utc)
        filter_params["to"] = to_dt.isoformat()

    payload = {
        "data": {
            "type": "spans",
            "attributes": {"filter": filter_params, "page": {"limit": 1000}},
        }
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        return json.dumps(
            {
                "error": f"API request failed with status {response.status_code}",
                "detail": response.text,
            },
            indent=2,
        )

    data = response.json()

    # Build span map
    spans = []
    for span in data.get("data", []):
        attrs = span.get("attributes", {})
        spans.append(
            {
                "id": span.get("id"),
                "span_id": attrs.get("span_id"),
                "parent_id": attrs.get("parent_id"),
                "trace_id": attrs.get("trace_id"),
                "name": attrs.get("name"),
                "span_kind": attrs.get("span_kind"),
                "status": attrs.get("status"),
                "start_ns": attrs.get("start_ns"),
                "duration": attrs.get("duration"),
                "model_name": attrs.get("model_name"),
                "model_provider": attrs.get("model_provider"),
                "input": attrs.get("input"),
                "output": attrs.get("output"),
                "meta": attrs.get("meta", {}),
                "metrics": attrs.get("metrics", {}),
                "tags": attrs.get("tags", []),
            }
        )

    span_map = {s["span_id"]: s for s in spans}

    # Find the root span
    root_span = None
    if root_span_id and root_span_id in span_map:
        root_span = span_map[root_span_id]
    else:
        # Find spans with parent_id = undefined
        root_spans = [s for s in spans if s.get("parent_id") == "undefined"]
        if root_spans:
            # Sort by start time and pick the one matching our criteria
            root_spans.sort(key=lambda x: x.get("start_ns", 0))
            root_span = root_spans[0]

    if not root_span:
        return json.dumps(
            {
                "error": "Could not find root span",
                "available_spans": len(spans),
            },
            indent=2,
        )

    def get_children(parent_id):
        children = [s for s in spans if s.get("parent_id") == parent_id]
        children.sort(key=lambda x: x.get("start_ns", 0))
        return children

    # Create a concise conversation flow (no duplication)
    def build_conversation_flow(span):
        """Extract just the key events: user request, LLM calls, tool calls, final response."""
        events = []

        # Add the initial user request from the root workflow
        if span["span_kind"] == "workflow":
            if span.get("input"):
                events.append(
                    {
                        "event": "user_request",
                        "content": span["input"].get("value")
                        if isinstance(span["input"], dict)
                        else span["input"],
                    }
                )

        # Recursively collect LLM and tool events
        def collect_events(s):
            collected = []

            if s["span_kind"] == "llm":
                event = {
                    "event": "llm_call",
                    "name": s["name"],
                    "model": s.get("model_name"),
                    "duration_ms": s["duration"] / 1_000_000
                    if s.get("duration")
                    else None,
                }

                # Get the input (last user message or tool result that triggered this call)
                if s.get("input"):
                    inp = s["input"]
                    if isinstance(inp, dict) and "messages" in inp:
                        messages = inp["messages"]
                        if messages:
                            # Get last user message
                            last_user = [m for m in messages if m.get("role") == "user"]
                            if last_user:
                                event["prompt"] = last_user[-1].get("content")

                # Get the response
                if s.get("output"):
                    out = s["output"]
                    if isinstance(out, dict):
                        if "messages" in out and out["messages"]:
                            msg = out["messages"][0] if out["messages"] else {}
                            content = msg.get("content")
                            if content:
                                event["response"] = content
                            # Check for tool use in response
                            tool_use = msg.get("tool_calls") or msg.get("tool_use")
                            if tool_use:
                                event["tool_calls"] = tool_use
                        elif "value" in out:
                            event["response"] = out["value"]

                # Add metrics
                if s.get("metrics"):
                    event["tokens"] = {
                        "input": s["metrics"].get("input_tokens"),
                        "output": s["metrics"].get("output_tokens"),
                    }

                collected.append(event)

            elif s["span_kind"] == "tool":
                event = {
                    "event": "tool_call",
                    "name": s["name"],
                    "duration_ms": s["duration"] / 1_000_000
                    if s.get("duration")
                    else None,
                }
                if s.get("input"):
                    inp = s["input"]
                    event["input"] = inp.get("value") if isinstance(inp, dict) else inp
                if s.get("output"):
                    out = s["output"]
                    event["output"] = out.get("value") if isinstance(out, dict) else out
                collected.append(event)

            # Process children
            children = get_children(s["span_id"])
            for child in children:
                collected.extend(collect_events(child))

            return collected

        events.extend(collect_events(span))

        return events

    conversation_flow = build_conversation_flow(root_span)

    return json.dumps(
        {
            "conversation_flow": conversation_flow,
            "total_spans_fetched": len(spans),
        },
        indent=2,
        default=str,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
