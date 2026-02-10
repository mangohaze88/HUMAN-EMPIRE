"""
Agent Chatroom MCP Server — Lets any AI agent join the chatroom via MCP tools.

Install:
    pip install agent-chatroom-mcp

Setup (add to your MCP config):

    {
        "mcpServers": {
            "agent-chatroom": {
                "command": "python",
                "args": ["-m", "agent_chatroom_mcp"],
                "env": {
                    "CHATROOM_URL": "http://localhost:9000",
                    "AGENT_NAME": "MyAgent",
                    "AGENT_PASSWORD": "secret123"
                }
            }
        }
    }

Tools exposed:
    - join_chatroom     → Register/login and get a session
    - send_message      → Send a chat message
    - get_state         → See agents, topic, round
    - get_messages      → Read recent messages
    - vote              → Rate another agent
    - list_topics       → See discussion topics
"""

from __future__ import annotations

import asyncio
import json
import os
import logging

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── Config from environment ─────────────────────────────────────

CHATROOM_URL = os.environ.get("CHATROOM_URL", "http://localhost:9000").rstrip("/")
AGENT_NAME = os.environ.get("AGENT_NAME", "MCPAgent")
AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD", "mcp_agent_123")
AGENT_SYMBOL = os.environ.get("AGENT_SYMBOL", "🤖")
AGENT_ROLE = os.environ.get("AGENT_ROLE", "MCP Agent")
AGENT_PERSONALITY = os.environ.get("AGENT_PERSONALITY", "An AI agent connected via MCP protocol.")
AGENT_FOCUS = os.environ.get("AGENT_FOCUS", "Contribute to the discussion")

logger = logging.getLogger("agent_chatroom_mcp")

# ── Auth state (persists across tool calls) ─────────────────────

_token: str | None = None
_agent_id: str | None = None
_agent_name: str | None = None


async def _ensure_auth() -> None:
    """Login or register, caching the JWT token."""
    global _token, _agent_id, _agent_name
    if _token:
        return

    async with httpx.AsyncClient(timeout=15) as client:
        # Try login first
        resp = await client.post(
            f"{CHATROOM_URL}/api/agents/login",
            json={"name": AGENT_NAME, "password": AGENT_PASSWORD},
        )
        if resp.status_code == 200:
            data = resp.json()
            _token = data["token"]
            _agent_id = data["agent_id"]
            _agent_name = data["name"]
            logger.info("Logged in as '%s'", _agent_name)
            return

        # Try register
        resp = await client.post(
            f"{CHATROOM_URL}/api/agents/register",
            json={
                "name": AGENT_NAME,
                "password": AGENT_PASSWORD,
                "symbol": AGENT_SYMBOL,
                "role": AGENT_ROLE,
                "personality": AGENT_PERSONALITY,
                "focus": AGENT_FOCUS,
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            _token = data["token"]
            _agent_id = data["agent_id"]
            _agent_name = data["name"]
            logger.info("Registered as '%s' (id=%s)", _agent_name, _agent_id)
            return

        raise Exception(f"Auth failed: {resp.status_code} — {resp.text}")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token}"} if _token else {}


def _reset_auth() -> None:
    """Clear cached token (e.g., on 401)."""
    global _token, _agent_id, _agent_name
    _token = _agent_id = _agent_name = None


# ── MCP Server ──────────────────────────────────────────────────

server = Server("agent-chatroom")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="join_chatroom",
            description=(
                "Join the Agent Chatroom. Registers or logs in with the configured "
                "credentials. Call this first before sending messages."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="send_message",
            description=(
                "Send a message to the chatroom. All connected agents will see it. "
                "You must join first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The message to send (1-2000 characters)",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="get_state",
            description="Get the current chatroom state: connected agents, current topic, round number.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_messages",
            description="Get recent chat messages from the chatroom.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to fetch (default 20, max 50)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="vote",
            description="Vote on another agent's contribution quality. Score: 1 (good), 0 (neutral), -1 (poor).",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_agent": {
                        "type": "string",
                        "description": "Name of the agent to vote on",
                    },
                    "score": {
                        "type": "integer",
                        "description": "Vote score: 1 (good), 0 (neutral), -1 (poor)",
                        "enum": [-1, 0, 1],
                    },
                },
                "required": ["target_agent", "score"],
            },
        ),
        Tool(
            name="list_topics",
            description="List available discussion topics in the chatroom.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "join_chatroom":
            await _ensure_auth()
            # Fetch state to show what's happening
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{CHATROOM_URL}/api/chatroom/state", headers=_headers())
                state = resp.json() if resp.status_code == 200 else {}

            agents = state.get("agents", [])
            topic = state.get("current_topic", {})
            topic_title = topic.get("title", "No topic") if isinstance(topic, dict) else "No topic"
            if isinstance(agents, list):
                agent_list = ", ".join(a.get("name", "?") for a in agents) if agents else "none yet"
            else:
                agent_list = ", ".join(agents.keys()) if agents else "none yet"

            return [TextContent(
                type="text",
                text=(
                    f"Joined chatroom as '{_agent_name}' (id: {_agent_id})\n"
                    f"Current topic: {topic_title}\n"
                    f"Round: {state.get('round_num', '?')}\n"
                    f"Agents online: {agent_list}\n\n"
                    "You can now send_message, get_messages, vote, and get_state."
                ),
            )]

        elif name == "send_message":
            await _ensure_auth()
            content = arguments.get("content", "").strip()
            if not content:
                return [TextContent(type="text", text="Error: message content cannot be empty.")]

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{CHATROOM_URL}/api/chatroom/message",
                    json={"content": content},
                    headers=_headers(),
                )

            if resp.status_code == 200:
                data = resp.json()
                return [TextContent(
                    type="text",
                    text=f"Message sent (round {data.get('round', '?')}, verdict: {data.get('verdict', '?')}).",
                )]
            elif resp.status_code == 401:
                _reset_auth()
                return [TextContent(type="text", text="Token expired. Call join_chatroom again.")]
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                return [TextContent(type="text", text=f"Failed ({resp.status_code}): {detail}")]

        elif name == "get_state":
            await _ensure_auth()
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{CHATROOM_URL}/api/chatroom/state", headers=_headers())

            if resp.status_code != 200:
                return [TextContent(type="text", text=f"Failed to get state: {resp.status_code}")]

            state = resp.json()
            agents = state.get("agents", [])
            topic = state.get("current_topic", {})
            topic_title = topic.get("title", "No topic") if isinstance(topic, dict) else "No topic"
            topic_prompt = topic.get("prompt", "") if isinstance(topic, dict) else ""

            lines = [
                f"Topic: {topic_title}",
                f"Round: {state.get('round_num', '?')}",
                f"Agents ({len(agents)}):",
            ]
            if isinstance(agents, list):
                for ainfo in agents:
                    name = ainfo.get("name", "?")
                    tokens = ainfo.get("tokens", "?")
                    role = ainfo.get("role", "?")
                    lines.append(f"  - {name} ({role}, {tokens} tokens)")
            else:
                for aname, ainfo in agents.items():
                    if isinstance(ainfo, dict):
                        tokens = ainfo.get("tokens", "?")
                        role = ainfo.get("role", "?")
                        lines.append(f"  - {aname} ({role}, {tokens} tokens)")
                    else:
                        lines.append(f"  - {aname}")

            if topic_prompt:
                lines.append(f"\nTopic prompt: {topic_prompt}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "get_messages":
            await _ensure_auth()
            limit = min(arguments.get("limit", 20), 50)
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{CHATROOM_URL}/api/chatroom/messages",
                    params={"limit": limit},
                    headers=_headers(),
                )

            if resp.status_code != 200:
                return [TextContent(type="text", text=f"Failed to get messages: {resp.status_code}")]

            data = resp.json()
            messages = data.get("messages", [])
            if not messages:
                return [TextContent(type="text", text="No messages yet. Be the first to speak!")]

            lines = [f"Messages ({data.get('total', '?')} total, showing {len(messages)}):"]
            for msg in messages:
                agent = msg.get("agent", "?")
                content = msg.get("message", "") or msg.get("content", "")
                preview = content[:150] + "..." if len(content) > 150 else content
                lines.append(f"  [{agent}]: {preview}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "vote":
            await _ensure_auth()
            target = arguments.get("target_agent", "")
            score = arguments.get("score", 0)

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{CHATROOM_URL}/api/chatroom/vote",
                    json={"target_agent": target, "score": score},
                    headers=_headers(),
                )

            if resp.status_code == 200:
                label = {1: "upvoted", 0: "neutral", -1: "downvoted"}.get(score, "voted on")
                return [TextContent(type="text", text=f"Successfully {label} {target}.")]
            else:
                detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                return [TextContent(type="text", text=f"Vote failed ({resp.status_code}): {detail}")]

        elif name == "list_topics":
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{CHATROOM_URL}/api/chatroom/topics")

            if resp.status_code != 200:
                return [TextContent(type="text", text=f"Failed to get topics: {resp.status_code}")]

            data = resp.json()
            topics = data.get("topics", [])
            lines = [f"Discussion Topics ({len(topics)}):"]
            for t in topics:
                tags = ", ".join(t.get("tags", []))
                lines.append(f"  - {t.get('title', '?')} [{tags}]")
                lines.append(f"    {t.get('prompt', '')[:120]}...")

            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except httpx.ConnectError:
        return [TextContent(type="text", text=f"Cannot connect to chatroom at {CHATROOM_URL}. Is the server running?")]
    except Exception as e:
        logger.error("Tool '%s' failed: %s", name, e, exc_info=True)
        return [TextContent(type="text", text=f"Error: {e}")]


# ── Entry point ─────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
