import pytest

from ai_bridge_kit import AIClient
from ai_bridge_kit.errors import ProviderNotFoundError, ValidationError
from ai_bridge_kit.providers import LocalFunctionProvider


def test_default_local_chat_and_embed() -> None:
    client = AIClient()

    chat = client.chat("hello")
    assert "hello" in chat.content
    assert chat.provider == "local"

    emb = client.embed(["a", "b"])
    assert emb.provider == "local"
    assert len(emb.vectors) == 2
    assert len(emb.vectors[0]) == 16


def test_register_custom_provider_and_call_function() -> None:
    provider = LocalFunctionProvider(name="custom")
    provider.register("chat", lambda payload: "custom-chat")
    provider.register(
        "score", lambda payload: {"label": "positive", "score": float(len(payload["text"]))}
    )
    provider.set_chat_function("chat")

    client = AIClient()
    client.register_provider(provider, set_default=True)

    chat = client.chat("any")
    assert chat.provider == "custom"
    assert chat.content == "custom-chat"

    output = client.call_function("score", provider="custom", arguments={"text": "great"})
    assert output.result["label"] == "positive"


def test_missing_provider_raises() -> None:
    client = AIClient()
    with pytest.raises(ProviderNotFoundError):
        client.chat("x", provider="does-not-exist")


def test_empty_chat_message_raises() -> None:
    client = AIClient()
    with pytest.raises(ValidationError):
        client.chat("   ")
