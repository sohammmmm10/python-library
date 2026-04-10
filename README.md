# ai-bridge-kit

`ai-bridge-kit` is a Python library to integrate AI providers through one clean API.

You can:
- Switch providers without rewriting app logic.
- Register your own local AI functions.
- Use retries and timeouts consistently.
- Keep your invention logic inside a reusable, package-ready SDK.

## Install

```bash
pip install -e .
```

With OpenAI support:

```bash
pip install -e ".[openai]"
```

With Anthropic support:

```bash
pip install -e ".[anthropic]"
```

Install all provider extras:

```bash
pip install -e ".[all]"
```

For development:

```bash
pip install -e ".[dev,all]"
```

## Quick Start

```python
from ai_bridge_kit import AIClient

client = AIClient()

# Uses built-in local provider by default.
chat = client.chat("Explain AI integration in one line.")
print(chat.content)

emb = client.embed(["hello world"])
print(len(emb.vectors[0]))
```

## Register your own AI functions

```python
from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import LocalFunctionProvider

provider = LocalFunctionProvider(name="my-ai")
provider.register("chat", lambda payload: "Custom answer")
provider.set_chat_function("chat")

client = AIClient()
client.register_provider(provider, set_default=True)

print(client.chat("hi").content)
```

## OpenAI Provider (optional)

```python
import os
from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import OpenAIProvider

client = AIClient()
client.register_provider(
    OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
    set_default=True,
)

resp = client.chat("Give 3 startup names for an AI integration SDK.", model="gpt-4o-mini")
print(resp.content)
```

## Additional Provider Adapters

### Anthropic

```python
import os
from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import AnthropicProvider

client = AIClient()
client.register_provider(
    AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
    set_default=True,
)
print(client.chat("Summarize agentic AI in one sentence.").content)
```

### Ollama (local)

```python
from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import OllamaProvider

client = AIClient()
client.register_provider(
    OllamaProvider(base_url="http://localhost:11434", default_chat_model="llama3.2"),
    set_default=True,
)
print(client.chat("Explain RAG briefly.").content)
```

### OpenAI-compatible APIs (OpenRouter/Groq/Together)

```python
import os
from ai_bridge_kit import AIClient
from ai_bridge_kit.providers import OpenAICompatibleProvider

client = AIClient()
client.register_provider(
    OpenAICompatibleProvider.for_openrouter(api_key=os.environ["OPENROUTER_API_KEY"]),
    set_default=True,
)
print(client.chat("Give 3 names for an AI bridge SDK.").content)
```

## CLI

```bash
ai-bridge providers
ai-bridge chat --message "What is retrieval augmented generation?"
ai-bridge embed --text "ai" --text "python"
ai-bridge call --function echo --arguments "{}"
```

## Run tests

```bash
python -m pytest
```

## Publish to PyPI

Use [`RELEASE.md`](RELEASE.md) for the full `build` + `twine` process.

## Patent workflow support

Use [`PATENT_DISCLOSURE_TEMPLATE.md`](PATENT_DISCLOSURE_TEMPLATE.md) to document your technical novelty before filing.
