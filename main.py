from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime, timezone
import json

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi


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

        return json.dumps({
            "query": params["query"],
            "from": params["from_ts"],
            "to": params["to_ts"],
            "count": len(logs),
            "logs": logs,
        }, indent=2, default=str)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
