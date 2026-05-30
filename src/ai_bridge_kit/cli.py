from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

from .client import AIClient
from .errors import AIBridgeError


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, default=str, indent=2))


def _build_client() -> AIClient:
    client = AIClient.from_env()

    try:
        from .providers import (
            AnthropicProvider,
            OllamaProvider,
            OpenAICompatibleProvider,
            OpenAIProvider,
        )

        client.register_provider(
            OllamaProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                default_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
                default_embedding_model=os.getenv(
                    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
                ),
            )
        )

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and OpenAIProvider is not None:
            client.register_provider(
                OpenAIProvider(
                    api_key=openai_key,
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    default_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    default_embedding_model=os.getenv(
                        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
                    ),
                )
            )

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            client.register_provider(
                AnthropicProvider(
                    api_key=anthropic_key,
                    default_chat_model=os.getenv(
                        "ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest"
                    ),
                )
            )

        if OpenAICompatibleProvider is not None:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                client.register_provider(
                    OpenAICompatibleProvider.for_openrouter(api_key=openrouter_key)
                )

            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                client.register_provider(OpenAICompatibleProvider.for_groq(api_key=groq_key))

            together_key = os.getenv("TOGETHER_API_KEY")
            if together_key:
                client.register_provider(
                    OpenAICompatibleProvider.for_together(api_key=together_key)
                )
    except Exception:
        pass

    return client


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-bridge", description="Unified CLI for AI provider integrations."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("providers", help="List registered providers.")

    chat = subparsers.add_parser("chat", help="Run a chat prompt.")
    chat.add_argument("--message", required=True, help="Prompt to send.")
    chat.add_argument("--provider", default=None, help="Provider name.")
    chat.add_argument("--model", default=None, help="Model name.")
    chat.add_argument("--temperature", type=float, default=None, help="Temperature value.")
    chat.add_argument(
        "--max-output-tokens", type=int, default=None, help="Max output token limit."
    )

    embed = subparsers.add_parser("embed", help="Create embeddings.")
    embed.add_argument(
        "--text",
        action="append",
        required=True,
        help="Input text. Repeat --text for multiple values.",
    )
    embed.add_argument("--provider", default=None, help="Provider name.")
    embed.add_argument("--model", default=None, help="Embedding model.")

    call = subparsers.add_parser("call", help="Call a provider function.")
    call.add_argument("--function", required=True, help="Function name.")
    call.add_argument(
        "--arguments",
        default="{}",
        help="Function arguments as JSON object. Example: '{\"text\":\"hello\"}'",
    )
    call.add_argument("--provider", default=None, help="Provider name.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    client = _build_client()

    try:
        if args.command == "providers":
            _print_json({"providers": client.available_providers()})
            return 0

        if args.command == "chat":
            response = client.chat(
                args.message,
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            _print_json(asdict(response))
            return 0

        if args.command == "embed":
            response = client.embed(
                args.text,
                provider=args.provider,
                model=args.model,
            )
            _print_json(asdict(response))
            return 0

        if args.command == "call":
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            response = client.call_function(
                args.function, arguments=arguments, provider=args.provider
            )
            _print_json(asdict(response))
            return 0

        parser.print_help()
        return 1

    except json.JSONDecodeError as exc:
        print(f"Invalid JSON for --arguments: {exc}", file=sys.stderr)
        return 2
    except (AIBridgeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
