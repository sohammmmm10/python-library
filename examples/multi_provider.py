import os

from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import AnthropicProvider, OllamaProvider, OpenAICompatibleProvider


def main() -> None:
    client = AIClient()

    client.register_provider(OllamaProvider(), set_default=True)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        client.register_provider(AnthropicProvider(api_key=anthropic_key))

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and OpenAICompatibleProvider is not None:
        client.register_provider(OpenAICompatibleProvider.for_openrouter(api_key=openrouter_key))

    print("Providers:", client.available_providers())
    print(client.chat("How do I design a clean provider abstraction?").content)


if __name__ == "__main__":
    main()
