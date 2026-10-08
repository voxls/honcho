import array
import base64
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from google.genai import types as genai_types

from src.config import (
    EmbeddingEncodingFormat,
    EmbeddingModelConfig,
    resolve_embedding_model_config,
)
from src.embedding_client import (
    BatchItem,
    EmbeddingClient,
    EmbeddingTokenLimitError,
    _EmbeddingClient,  # pyright: ignore[reportPrivateUsage]
)


def gemini_call_texts(contents: Any) -> list[str]:
    """Unwrap a recorded Gemini `contents` argument back to plain texts."""
    return [content.parts[0].text for content in contents]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding: list[float] = embedding
        self.calls: list[dict[str, Any]] = []
        # Simulate a provider answering 200 with missing embeddings.
        self.returns_no_data: bool = False
        self.truncate_data_to: int | None = None

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> SimpleNamespace:
        call: dict[str, Any] = {"model": model, "input": input}
        call.update(kwargs)
        self.calls.append(call)
        # Mirror the SDK: a named encoding_format skips its base64 decode, so the
        # response carries the raw string instead of floats.
        payload: Any = self.embedding
        if kwargs.get("encoding_format") == "base64":
            payload = base64.b64encode(
                array.array("f", self.embedding).tobytes()
            ).decode()
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=payload) for _ in input]
        else:
            data = [SimpleNamespace(embedding=payload)]
        if self.returns_no_data:
            data = []
        elif self.truncate_data_to is not None:
            data = data[: self.truncate_data_to]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_openai_embedding_client_uses_configured_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 8)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.1] * 8
    assert fake_embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello world"],
            "encoding_format": "float",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 7)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await client.embed("hello world")


def test_gemini_embedding_client_gemini_001_caps_at_2048(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemini-embedding-001 should cap max_embedding_tokens at 2048."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            self.api_key: str = api_key

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    # When max_input_tokens is above the model cap, it should be clamped to 2048.
    client_above_cap = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=20_000,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client_above_cap.max_embedding_tokens == 2048

    # When max_input_tokens is below the model cap, it should pass through unchanged.
    client_below_cap = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=1024,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client_below_cap.max_embedding_tokens == 1024


def test_gemini_embedding_client_gemini_2_caps_at_8192(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemini-embedding-2 should cap max_embedding_tokens at 8192."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            self.api_key: str = api_key

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    # When max_input_tokens is above the model cap, it should be clamped to 8192.
    client_above_cap = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-2",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=20_000,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client_above_cap.max_embedding_tokens == 8192

    # When max_input_tokens is below the model cap, it should pass through unchanged.
    client_below_cap = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-2",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client_below_cap.max_embedding_tokens == 4096


@pytest.mark.parametrize(
    "model",
    [
        "models/gemini-embedding-2",
        "gemini-embedding-2-preview",
        "models/gemini-embedding-2-preview",
    ],
)
def test_gemini_embedding_client_models_prefixed_gemini_2_caps_at_8192(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """The canonical 'models/' form (as returned by Gemini's API) and the
    -preview alias must still be recognized and granted the 8192 cap."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            self.api_key: str = api_key

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model=model,
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=20_000,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client.max_embedding_tokens == 8192


def test_gemini_embedding_client_unknown_model_defaults_to_2048(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown Gemini model names should conservatively default to a 2048 cap."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            self.api_key: str = api_key

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-unknown-future-model",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=20_000,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client.max_embedding_tokens == 2048


def test_gemini_embedding_client_near_miss_model_id_defaults_to_2048(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model id that merely contains 'gemini-embedding-2' as a substring
    (e.g. 'gemini-embedding-20') must not be granted the 8192 cap reserved
    for the exact 'gemini-embedding-2' model id."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str, http_options: Any) -> None:
            self.api_key: str = api_key

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-20",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=20_000,
        max_tokens_per_request=300_000,
        send_dimensions=True,
    )
    assert client.max_embedding_tokens == 2048


@pytest.mark.asyncio
async def test_gemini_embedding_client_uses_output_dimensionality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12)],
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            base_url="https://gemini-proxy.example/v1beta",
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.2] * 12
    # 10-minute HTTP timeout, in lockstep with the LLM registry's Gemini client
    # (see #785). Without this, a stalled Gemini embedding socket wedges the
    # deriver worker — the same failure mode the LLM fix addresses.
    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.base_url == "https://gemini-proxy.example/v1beta"
    assert gemini_client.http_options.timeout == 600_000
    assert calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "hello world",
            "config": {"output_dimensionality": 12},
        }
    ]


@pytest.mark.asyncio
async def test_gemini_embedding_client_keeps_timeout_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-base-url Gemini embedding client must still carry an HTTP timeout."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.base_url is None
    assert gemini_client.http_options.timeout == 600_000


@pytest.mark.asyncio
async def test_openai_embedding_client_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured embedding timeout reaches the OpenAI-compatible client."""

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.timeout: float | None = timeout
            self.embeddings: FakeOpenAIEmbeddingsAPI = FakeOpenAIEmbeddingsAPI(
                [0.1] * 8
            )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            timeout=45,
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    openai_client = cast(Any, client.client)
    assert openai_client.timeout == 45.0


@pytest.mark.asyncio
async def test_openai_embedding_client_omits_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset timeout omits the kwarg so the OpenAI SDK keeps its default."""

    missing = object()

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: object = missing,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.timeout: object = timeout
            self.embeddings: FakeOpenAIEmbeddingsAPI = FakeOpenAIEmbeddingsAPI(
                [0.1] * 8
            )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    openai_client = cast(Any, client.client)
    assert openai_client.timeout is missing


@pytest.mark.asyncio
async def test_gemini_embedding_client_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured embedding timeout reaches Gemini as milliseconds."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            timeout=45,
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.timeout == 45_000


def _build_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding: list[float],
    model: str,
    send_dimensions: bool,
    vector_dimensions: int,
    max_batch_size: int | None = None,
    encoding_format: EmbeddingEncodingFormat = "float",
) -> tuple[_EmbeddingClient, FakeOpenAIEmbeddingsAPI]:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model=model,
            api_key="test-key",
            max_batch_size=max_batch_size,
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=send_dimensions,
        encoding_format=encoding_format,
    )
    return client, fake_embeddings


@pytest.mark.asyncio
async def test_openai_embed_forwards_dimensions_when_send_dimensions_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.embed("hello")

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "encoding_format": "float",
            "dimensions": 768,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embed_omits_dimensions_when_send_dimensions_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
    )

    await client.embed("hello")

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "encoding_format": "float",
        }
    ]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.simple_batch_embed(["a", "b"])

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768
    assert fake.calls[0]["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_respects_configured_max_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
        max_batch_size=2,
    )

    await client.simple_batch_embed(["a", "b", "c"])

    assert [call["input"] for call in fake.calls] == [["a", "b"], ["c"]]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_defaults_to_2048_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset max_batch_size must keep the OpenAI default: one request."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
    )
    assert client.max_batch_size == 2048

    await client.simple_batch_embed(["a", "b", "c"])

    assert [call["input"] for call in fake.calls] == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_gemini_simple_batch_embed_respects_configured_max_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini transport must split batches at the configured limit too."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            n = len(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12) for _ in range(n)]
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            max_batch_size=2,
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    await client.simple_batch_embed(["a", "b", "c"])

    assert [gemini_call_texts(call["contents"]) for call in calls] == [
        ["a", "b"],
        ["c"],
    ]


@pytest.mark.asyncio
async def test_gemini_simple_batch_embed_defaults_to_100_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset max_batch_size must keep the Gemini conservative default."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            n = len(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12) for _ in range(n)]
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )
    assert client.max_batch_size == 100

    await client.simple_batch_embed(["a", "b", "c"])

    assert [gemini_call_texts(call["contents"]) for call in calls] == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_openai_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768


@pytest.mark.asyncio
async def test_openai_embed_requests_float_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-query path must request float embeddings explicitly.

    Without an explicit encoding_format, the openai SDK defaults to base64,
    which OpenAI-compatible providers such as OpenRouter answer with empty
    embedding data for models that don't support base64 encoding.
    """
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )

    await client.embed("hello")

    assert fake.calls[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_openai_batch_embed_requests_float_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch path must request float embeddings explicitly, like embed()."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(fake.calls) == 1
    assert fake.calls[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_openai_embed_reports_missing_embedding_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit encoding_format turns off the SDK's own empty-data check, so
    a provider answering 200 with no embeddings must still fail legibly."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )
    fake.returns_no_data = True

    with pytest.raises(ValueError, match="Embedding count mismatch"):
        await client.embed("hello")


@pytest.mark.asyncio
async def test_openai_batch_embed_reports_short_embedding_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch answered with fewer embeddings than inputs must name the counts
    rather than surface a bare zip() error."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )
    fake.truncate_data_to = 1

    with pytest.raises(ValueError, match="Expected 2, got 1"):
        await client.batch_embed({"a": "hello", "b": "world"})


def _build_embedding_settings(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Construct a fresh EmbeddingSettings from the given env, isolated from os.environ."""
    from src.config import EmbeddingSettings

    for key in (
        "EMBEDDING_VECTOR_DIMENSIONS",
        "EMBEDDING_MODEL_CONFIG__MODEL",
        "EMBEDDING_MODEL_CONFIG__TRANSPORT",
        "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE",
        "EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE",
        "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL",
        "EMBEDDING_MODEL_CONFIG__MAX_BATCH_SIZE",
        "EMBEDDING_MODEL_CONFIG__QUERY_PREFIX",
        "EMBEDDING_MODEL_CONFIG__QUERY_SUFFIX",
        "EMBEDDING_MODEL_CONFIG__DOCUMENT_PREFIX",
        "EMBEDDING_MODEL_CONFIG__DOCUMENT_SUFFIX",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return EmbeddingSettings()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # No base_url means real OpenAI, which serves base64 at ~1/3.6 the bytes.
        ({}, "base64"),
        (
            {
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://api.openai.com/v1"
            },
            "base64",
        ),
        (
            {
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://openrouter.ai/api/v1"
            },
            "float",
        ),
        (
            {"EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "http://localhost:8000/v1"},
            "float",
        ),
        ({"EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE": "float"}, "float"),
        (
            {
                "EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE": "base64",
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://openrouter.ai/api/v1",
            },
            "base64",
        ),
    ],
)
def test_resolve_encoding_format(
    env: dict[str, str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _build_embedding_settings(env, monkeypatch)
    assert s.resolve_encoding_format() == expected


@pytest.mark.asyncio
async def test_openai_base64_mode_omits_encoding_format_and_returns_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base64 mode must request by omission on both paths.

    Naming `base64` explicitly makes the SDK skip its own decode and hand back
    the raw string, which then fails the dimension check.
    """
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
        encoding_format="base64",
    )

    embedding = await client.embed("hello")
    batched = await client.batch_embed({"a": "hello", "b": "world"})

    assert all("encoding_format" not in call for call in fake.calls)
    assert len(embedding) == 8
    assert [len(vectors[0]) for vectors in batched.values()] == [8, 8]


def test_resolve_send_dimensions_auto_default_dim_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({}, monkeypatch)
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_auto_explicit_dim_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({"EMBEDDING_VECTOR_DIMENSIONS": "768"}, monkeypatch)
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_auto_ada_002_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_VECTOR_DIMENSIONS": "1536",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_always_returns_true_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always"},
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_always_overrides_ada_rejecting_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_never_returns_false_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "never",
            "EMBEDDING_VECTOR_DIMENSIONS": "768",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


@pytest.mark.asyncio
async def test_simple_batch_embed_respects_token_budget_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """simple_batch_embed must split inputs across requests so per-request token cap holds."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.5] * 4)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    # max_input_tokens=100 per single input; max_tokens_per_request=120 total,
    # so two ~80-token inputs must end up in *separate* requests.
    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=100,
        max_tokens_per_request=120,
        send_dimensions=False,
    )

    # "word " * 80 produces ~80 tokens with cl100k_base/the model encoding.
    long_a = ("alpha " * 80).strip()
    long_b = ("beta " * 80).strip()

    out = await client.simple_batch_embed([long_a, long_b])
    assert len(out) == 2
    # Per-request token cap forces two separate requests.
    assert len(fake_embeddings.calls) == 2


@pytest.mark.asyncio
async def test_simple_batch_embed_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs that exceed max_embedding_tokens must raise ValueError immediately."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    too_long = ("word " * 50).strip()
    with pytest.raises(ValueError, match="maximum token limit"):
        await client.simple_batch_embed([too_long])


@pytest.mark.asyncio
async def test_simple_batch_embed_truncates_oversize_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_oversize='truncate' embeds a prefix instead of failing the batch."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    short = "hello"
    too_long = ("word " * 50).strip()
    assert len(client.encoding.encode(too_long)) > client.max_embedding_tokens

    out = await client.simple_batch_embed([short, too_long], on_oversize="truncate")

    assert len(out) == 2
    assert fake_embeddings.calls, "expected a provider call after truncation"
    received = fake_embeddings.calls[0]["input"]
    assert received[0] == short
    truncated = received[1]
    assert isinstance(truncated, str)
    assert truncated != too_long
    assert len(client.encoding.encode(truncated)) <= client.max_embedding_tokens


@pytest.mark.asyncio
async def test_simple_batch_embed_truncate_reencodes_until_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """decode(ids[:n]) can re-encode past n; truncate must re-verify the count."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    encode_calls = {"n": 0}

    def encode(text: str) -> list[int]:
        encode_calls["n"] += 1
        if text.startswith("LONG"):
            # 1: original oversize; 2: still over after first slice; 3+: fits.
            if encode_calls["n"] == 1:
                return list(range(20))
            if encode_calls["n"] == 2:
                return list(range(12))
            return list(range(8))
        return [1]

    def decode(ids: list[int]) -> str:
        return "LONG" + "x" * len(ids)

    monkeypatch.setattr(client.encoding, "encode", encode)
    monkeypatch.setattr(client.encoding, "decode", decode)

    out = await client.simple_batch_embed(["LONG-input"], on_oversize="truncate")

    assert len(out) == 1
    received = fake_embeddings.calls[0]["input"][0]
    assert isinstance(received, str)
    # The provider must see the post-loop text, which encodes to 8 (<= cap).
    assert encode(received) == list(range(8))
    assert encode_calls["n"] >= 3


@pytest.mark.asyncio
async def test_public_embedding_client_forwards_on_oversize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The singleton wrapper must forward on_oversize to the inner client."""
    captured: dict[str, object] = {}

    class FakeInner:
        async def simple_batch_embed(
            self,
            texts: list[str],
            *,
            on_oversize: str = "raise",
            input_type: str = "document",
        ) -> list[list[float]]:
            captured["texts"] = texts
            captured["on_oversize"] = on_oversize
            captured["input_type"] = input_type
            return [[0.1]]

    wrapper = EmbeddingClient()
    monkeypatch.setattr(wrapper, "_get_client", lambda: FakeInner())

    out = await wrapper.simple_batch_embed(["hi"], on_oversize="truncate")

    assert out == [[0.1]]
    assert captured["texts"] == ["hi"]
    assert captured["on_oversize"] == "truncate"
    assert captured["input_type"] == "document"


def test_prepare_chunks_returns_ordered_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_chunks must split oversized inputs using the same rules as batch_embed."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    short_text = "hello"
    long_text = ("word " * 50).strip()

    out = client.prepare_chunks({"short": short_text, "long": long_text})

    assert out["short"] == [short_text]
    assert len(out["long"]) > 1
    # Order preserved
    assert isinstance(out["long"][0], str)


def test_embedding_model_config_parses_max_batch_size_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__MAX_BATCH_SIZE": "10"},
        monkeypatch,
    )

    assert s.MODEL_CONFIG.max_batch_size == 10

    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)
    assert resolved.max_batch_size == 10


def test_embedding_model_config_parses_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__TIMEOUT": "90.0"},
        monkeypatch,
    )

    assert s.MODEL_CONFIG.timeout == 90.0

    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)
    assert resolved.timeout == 90.0


def test_embedding_model_config_rejects_invalid_timeout() -> None:
    with pytest.raises(
        ValueError, match=r"provider_params\.timeout must be a positive number"
    ):
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            timeout=-1,
        )


@pytest.mark.asyncio
async def test_gemini_process_batch_wraps_contents_as_content_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each batch item must be its own Content so gemini-embedding-2* returns
    one embedding per item instead of merging them into one document."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            embeddings = [SimpleNamespace(values=[0.3] * 8) for _ in contents]
            return SimpleNamespace(embeddings=embeddings)

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-2",
            api_key="gemini-key",
            base_url=None,
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    batch = [
        BatchItem("hello", "id1", 0, 1),
        BatchItem("world", "id2", 0, 1),
    ]
    result = await client._process_batch(batch)  # pyright: ignore[reportPrivateUsage]

    assert result["id1"][0] == [0.3] * 8
    assert result["id2"][0] == [0.3] * 8

    assert len(calls) == 1
    contents = calls[0]["contents"]
    assert len(contents) == 2
    assert all(isinstance(c, genai_types.Content) for c in contents)
    assert contents[0].parts[0].text == "hello"
    assert contents[1].parts[0].text == "world"


# --- Token-limit classification (issue #568) -------------------------------
#
# Only genuine "content too long" conditions may raise
# EmbeddingTokenLimitError. Provider/config failures must stay plain
# ValueError so callers don't rewrite them as token-limit errors.


def test_embedding_token_limit_error_is_value_error() -> None:
    """Subclassing ValueError keeps pre-existing broad handlers working."""
    assert issubclass(EmbeddingTokenLimitError, ValueError)


@pytest.mark.asyncio
async def test_embed_raises_token_limit_error_before_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake_embeddings = _build_openai_client(
        monkeypatch,
        embedding=[0.1, 0.2],
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=2,
    )

    with pytest.raises(EmbeddingTokenLimitError):
        await client.embed("word " * 20_000)

    assert fake_embeddings.calls == [], "provider must not be called on oversize input"


@pytest.mark.asyncio
async def test_simple_batch_embed_raises_token_limit_error_before_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake_embeddings = _build_openai_client(
        monkeypatch,
        embedding=[0.1, 0.2],
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=2,
    )

    with pytest.raises(EmbeddingTokenLimitError):
        await client.simple_batch_embed(["fine", "word " * 20_000])

    assert fake_embeddings.calls == [], "provider must not be called on oversize input"


@pytest.mark.asyncio
async def test_provider_dimension_mismatch_is_not_a_token_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong-width vector is a provider/config fault, not an oversized input."""
    client, _ = _build_openai_client(
        monkeypatch,
        embedding=[0.1, 0.2, 0.3],  # 3 wide, client expects 2
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=2,
    )

    with pytest.raises(ValueError) as excinfo:
        await client.embed("short query")

    assert not isinstance(excinfo.value, EmbeddingTokenLimitError)


# --- Per-side input templates (prefix + suffix) ------------------------------
#
# Chat-template embedding models with last-token pooling (Qwen3-Embedding,
# Qwen3-VL-Embedding) need every input wrapped as prefix + text + suffix, with
# a different template for queries and documents.

QUERY_PREFIX = (
    "<|im_start|>system\nGiven a search query, retrieve relevant memories or "
    + "conversation messages that help answer it.<|im_end|>\n<|im_start|>user\n"
)
QUERY_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
DOCUMENT_PREFIX = (
    "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n"
)
DOCUMENT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _templated_config(transport: Literal["openai", "gemini"] = "openai") -> Any:
    return EmbeddingModelConfig(
        transport=transport,
        model="qwen3-vl-embedding-2b"
        if transport == "openai"
        else "gemini-embedding-001",
        api_key="test-key",
        query_prefix=QUERY_PREFIX,
        query_suffix=QUERY_SUFFIX,
        document_prefix=DOCUMENT_PREFIX,
        document_suffix=DOCUMENT_SUFFIX,
    )


def _build_templated_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_input_tokens: int = 8192,
    max_tokens_per_request: int = 300_000,
    config: EmbeddingModelConfig | None = None,
) -> tuple[_EmbeddingClient, FakeOpenAIEmbeddingsAPI]:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        config or _templated_config(),
        vector_dimensions=4,
        max_input_tokens=max_input_tokens,
        max_tokens_per_request=max_tokens_per_request,
        send_dimensions=False,
    )
    return client, fake_embeddings


def _count(client: _EmbeddingClient, text: str) -> int:
    return len(client.encoding.encode(text, disallowed_special=()))


def _overhead(client: _EmbeddingClient, prefix: str, suffix: str) -> int:
    return _count(client, prefix) + _count(client, suffix)


def _cap_with_document_budget(
    monkeypatch: pytest.MonkeyPatch, document_budget: int
) -> int:
    """`max_input_tokens` leaving `document_budget` raw tokens per document.

    The cap also has to admit the (longer) query template, or construction
    fails before the document side can be exercised.
    """
    probe, _ = _build_templated_client(monkeypatch)
    query_overhead = _overhead(probe, QUERY_PREFIX, QUERY_SUFFIX)
    document_overhead = _overhead(probe, DOCUMENT_PREFIX, DOCUMENT_SUFFIX)
    assert query_overhead - document_overhead < document_budget
    return document_overhead + document_budget


@pytest.mark.asyncio
async def test_embed_wraps_query_by_default_and_document_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_templated_client(monkeypatch)

    await client.embed("who is alice", input_type="query")
    await client.embed("alice lives in berlin", input_type="document")
    await client.embed("default side")

    assert fake.calls[0]["input"] == [QUERY_PREFIX + "who is alice" + QUERY_SUFFIX]
    assert fake.calls[1]["input"] == [
        DOCUMENT_PREFIX + "alice lives in berlin" + DOCUMENT_SUFFIX
    ]
    assert fake.calls[2]["input"] == [QUERY_PREFIX + "default side" + QUERY_SUFFIX]


@pytest.mark.asyncio
async def test_simple_batch_embed_wraps_documents_by_default_and_queries_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_templated_client(monkeypatch)

    await client.simple_batch_embed(["one", "two"])
    await client.simple_batch_embed(["what does she want"], input_type="query")

    assert fake.calls[0]["input"] == [
        DOCUMENT_PREFIX + "one" + DOCUMENT_SUFFIX,
        DOCUMENT_PREFIX + "two" + DOCUMENT_SUFFIX,
    ]
    assert fake.calls[1]["input"] == [
        QUERY_PREFIX + "what does she want" + QUERY_SUFFIX
    ]


@pytest.mark.asyncio
async def test_no_template_configured_sends_raw_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default config must put byte-identical input on the wire to before."""
    client, fake = _build_templated_client(
        monkeypatch,
        config=EmbeddingModelConfig(
            transport="openai", model="text-embedding-3-small", api_key="test-key"
        ),
    )

    await client.embed("plain query", input_type="query")
    await client.embed("plain doc", input_type="document")
    await client.simple_batch_embed(["a", "b"], input_type="document")
    await client.simple_batch_embed(["q"], input_type="query")
    await client.batch_embed({"x": "chunk me"}, input_type="document")

    assert [call["input"] for call in fake.calls] == [
        ["plain query"],
        ["plain doc"],
        ["a", "b"],
        ["q"],
        ["chunk me"],
    ]


@pytest.mark.asyncio
async def test_embed_rejects_text_that_only_overflows_once_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, _ = _build_templated_client(monkeypatch)
    overhead = _overhead(probe, QUERY_PREFIX, QUERY_SUFFIX)
    max_input_tokens = overhead + 10
    client, fake = _build_templated_client(
        monkeypatch, max_input_tokens=max_input_tokens
    )

    text = " ".join(["word"] * 15)
    raw_tokens = _count(client, text)
    assert 10 < raw_tokens <= max_input_tokens, "must fit raw but not wrapped"

    with pytest.raises(
        EmbeddingTokenLimitError, match="maximum token limit of 10 tokens"
    ):
        await client.embed(text, input_type="query")
    assert fake.calls == []


@pytest.mark.asyncio
async def test_truncate_keeps_wrapped_payload_within_cap_and_suffix_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_input_tokens = _cap_with_document_budget(monkeypatch, 20)
    client, fake = _build_templated_client(
        monkeypatch, max_input_tokens=max_input_tokens
    )

    text = " ".join(f"word{i}" for i in range(15))
    assert 20 < _count(client, text) <= max_input_tokens, "fits raw, not wrapped"

    await client.simple_batch_embed(
        [text], on_oversize="truncate", input_type="document"
    )

    sent = fake.calls[0]["input"][0]
    assert sent.startswith(DOCUMENT_PREFIX)
    assert sent.endswith(DOCUMENT_SUFFIX)
    assert _count(client, sent) <= max_input_tokens
    body = sent[len(DOCUMENT_PREFIX) : -len(DOCUMENT_SUFFIX)]
    assert body and text.startswith(body)


def test_public_truncate_uses_template_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routers truncate search queries before embed(); the result must then pass
    embed()'s budget check rather than being rejected after wrapping."""
    probe, _ = _build_templated_client(monkeypatch)
    overhead = _overhead(probe, QUERY_PREFIX, QUERY_SUFFIX)
    client, _ = _build_templated_client(monkeypatch, max_input_tokens=overhead + 10)

    truncated, tokens = client.truncate_to_token_limit(
        " ".join(f"word{i}" for i in range(40)), input_type="query"
    )

    assert tokens <= 10
    assert _count(client, QUERY_PREFIX + truncated + QUERY_SUFFIX) <= overhead + 10


@pytest.mark.asyncio
async def test_prepare_chunks_sizes_to_template_budget_and_batch_embed_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    max_input_tokens = _cap_with_document_budget(monkeypatch, 20)
    client, fake = _build_templated_client(
        monkeypatch, max_input_tokens=max_input_tokens
    )
    long_text = " ".join(f"word{i}" for i in range(25))
    # Fits the raw cap in one piece, so any split is down to the template.
    assert _count(client, long_text) <= max_input_tokens

    chunks = client.prepare_chunks({"m": long_text}, input_type="document")["m"]

    assert len(chunks) > 1
    for chunk in chunks:
        # Persisted chunks never carry template text...
        assert "<|im_start|>" not in chunk
        # ...and are sized so the wrapped form still fits the model cap.
        assert _count(client, chunk) <= 20
        assert (
            _count(client, DOCUMENT_PREFIX + chunk + DOCUMENT_SUFFIX)
            <= max_input_tokens
        )

    await client.batch_embed({"m": long_text}, input_type="document")

    assert fake.calls[0]["input"] == [
        DOCUMENT_PREFIX + chunk + DOCUMENT_SUFFIX for chunk in chunks
    ]


@pytest.mark.asyncio
async def test_request_token_cap_counts_wrapped_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two texts whose raw tokens fit one request but whose wrapped tokens do
    not must be split across two requests."""
    probe, _ = _build_templated_client(monkeypatch)
    overhead = _overhead(probe, DOCUMENT_PREFIX, DOCUMENT_SUFFIX)
    client, fake = _build_templated_client(
        monkeypatch, max_tokens_per_request=overhead + 10
    )

    await client.simple_batch_embed(["alpha", "beta"], input_type="document")

    assert [len(call["input"]) for call in fake.calls] == [1, 1]


@pytest.mark.asyncio
async def test_telemetry_reports_wrapped_token_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[int] = []

    def capture(**kwargs: Any) -> None:
        published.append(kwargs["input_tokens_estimate"])

    monkeypatch.setattr("src.embedding_client._publish_embedding_event", capture)
    client, _ = _build_templated_client(monkeypatch)

    await client.embed("hello world", input_type="query")
    await client.simple_batch_embed(["hello world"], input_type="document")

    assert published == [
        _count(client, QUERY_PREFIX + "hello world" + QUERY_SUFFIX),
        _count(client, DOCUMENT_PREFIX + "hello world" + DOCUMENT_SUFFIX),
    ]


@pytest.mark.asyncio
async def test_gemini_paths_send_wrapped_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    class FakeGeminiModels:
        async def embed_content(
            self, *, contents: Any, **_kwargs: Any
        ) -> SimpleNamespace:
            calls.append(contents)
            count = len(cast(list[Any], contents)) if isinstance(contents, list) else 1
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.3] * 4) for _ in range(count)]
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        _templated_config("gemini"),
        vector_dimensions=4,
        max_input_tokens=2048,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    await client.embed("who is alice", input_type="query")
    await client.simple_batch_embed(["alice lives in berlin"], input_type="document")

    assert calls[0] == QUERY_PREFIX + "who is alice" + QUERY_SUFFIX
    assert gemini_call_texts(calls[1]) == [
        DOCUMENT_PREFIX + "alice lives in berlin" + DOCUMENT_SUFFIX
    ]


def test_template_that_consumes_the_whole_budget_raises_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, _ = _build_templated_client(monkeypatch)
    overhead = _overhead(probe, QUERY_PREFIX, QUERY_SUFFIX)

    with pytest.raises(ValueError, match="query template uses"):
        _build_templated_client(monkeypatch, max_input_tokens=overhead)


@pytest.mark.asyncio
async def test_public_embedding_client_forwards_input_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeInner:
        async def embed(self, _query: str, *, input_type: str) -> list[float]:
            captured["embed"] = input_type
            return [0.1]

        async def simple_batch_embed(
            self, _texts: list[str], *, input_type: str, **_kwargs: object
        ) -> list[list[float]]:
            captured["simple_batch_embed"] = input_type
            return [[0.1]]

        async def batch_embed(
            self, _id_resource_dict: dict[str, str], *, input_type: str
        ) -> dict[str, list[list[float]]]:
            captured["batch_embed"] = input_type
            return {}

        def prepare_chunks(
            self, _id_resource_dict: dict[str, str], *, input_type: str
        ) -> dict[str, list[str]]:
            captured["prepare_chunks"] = input_type
            return {}

        def truncate_to_token_limit(
            self, text: str, *, input_type: str
        ) -> tuple[str, int]:
            captured["truncate_to_token_limit"] = input_type
            return text, 1

    wrapper = EmbeddingClient()
    monkeypatch.setattr(wrapper, "_get_client", lambda: FakeInner())

    await wrapper.embed("q", input_type="document")
    await wrapper.simple_batch_embed(["d"], input_type="query")
    await wrapper.batch_embed({"a": "d"}, input_type="query")
    wrapper.prepare_chunks({"a": "d"}, input_type="query")
    wrapper.truncate_to_token_limit("q", input_type="document")

    assert captured == {
        "embed": "document",
        "simple_batch_embed": "query",
        "batch_embed": "query",
        "prepare_chunks": "query",
        "truncate_to_token_limit": "document",
    }


def test_templates_parse_from_env_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newlines and `<|...|>` markers must survive config loading unchanged."""
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__QUERY_PREFIX": QUERY_PREFIX,
            "EMBEDDING_MODEL_CONFIG__QUERY_SUFFIX": QUERY_SUFFIX,
            "EMBEDDING_MODEL_CONFIG__DOCUMENT_PREFIX": DOCUMENT_PREFIX,
            "EMBEDDING_MODEL_CONFIG__DOCUMENT_SUFFIX": DOCUMENT_SUFFIX,
        },
        monkeypatch,
    )

    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)

    assert resolved.query_prefix == QUERY_PREFIX
    assert resolved.query_suffix == QUERY_SUFFIX
    assert resolved.document_prefix == DOCUMENT_PREFIX
    assert resolved.document_suffix == DOCUMENT_SUFFIX
    # Defaults stay empty, so OpenAI/Gemini deployments are untouched.
    default = resolve_embedding_model_config(
        _build_embedding_settings({}, monkeypatch).MODEL_CONFIG
    )
    assert (default.query_prefix, default.query_suffix) == ("", "")
    assert (default.document_prefix, default.document_suffix) == ("", "")
