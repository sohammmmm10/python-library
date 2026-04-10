import os

from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import OpenAIProvider


def main() -> None:
    client = AIClient()

    print("== Local Chat ==")
    local_reply = client.chat("How can I integrate many AI services?").content
    print(local_reply)

    print("\n== Local Embeddings ==")
    emb = client.embed(["hello", "world"])
    print(f"vectors: {len(emb.vectors)}; dimension: {len(emb.vectors[0])}")

    print("\n== Local Function Call ==")
    called = client.call_function("echo", arguments={"use_case": "agent routing"})
    print(called.result)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nOPENAI_API_KEY not found, skipping OpenAI example.")
        return

    try:
        client.register_provider(OpenAIProvider(api_key=api_key), set_default=True)
        print("\n== OpenAI Chat ==")
        openai_reply = client.chat(
            "Give one product name for a universal AI bridge library.",
            model="gpt-4o-mini",
            provider="openai",
        )
        print(openai_reply.content)
    except Exception as exc:
        print(f"\nOpenAI example skipped: {exc}")


if __name__ == "__main__":
    main()
