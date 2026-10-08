from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, cast

import tiktoken
from nanoid import generate as generate_nanoid

from .config import (
    EmbeddingEncodingFormat,
    EmbeddingModelConfig,
    resolve_embedding_model_config,
    settings,
)

if TYPE_CHECKING:
    from google import genai
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Which side of a retrieval pair a text belongs to. Asymmetric embedding models
# want a different instruction template for each; symmetric ones ignore the
# distinction entirely (all template fields default to empty).
InputType = Literal["query", "document"]


async def _emit_embedding_call(
    *,
    provider: str,
    model: str,
    texts: list[str],
    input_tokens_estimate: int,
    fn: Callable[[], Awaitable[_T]],
    is_final_attempt: bool = True,
    attempt: int = 1,
    retry_attempts: int = 1,
) -> _T:
    """time a single embedding-provider call, emit
    `embedding.call.completed` on both success and exception, and return the
    call's result. Errors propagate unchanged — telemetry never bleeds into
    the caller's control flow.

    Caller-supplied `texts` is used only for `input_count`; we don't keep the
    list around for the event to avoid leaking content into telemetry.

    `is_final_attempt` defaults to True so one-shot callers (`embed`,
    `simple_batch_embed`) get correct semantics without changes. Retry-loop
    callers (`_process_batch`) pass the real attempt index so dashboards
    can distinguish exhausted retries from mid-retry failures.
    """
    start = time.perf_counter()
    error: BaseException | None = None
    try:
        return await fn()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is None:
            outcome: Literal["success", "error", "cancelled"] = "success"
        elif isinstance(error, asyncio.CancelledError):
            outcome = "cancelled"
        else:
            outcome = "error"
        _publish_embedding_event(
            provider=provider,
            model=model,
            input_count=len(texts),
            input_tokens_estimate=input_tokens_estimate,
            duration_ms=(time.perf_counter() - start) * 1000,
            outcome=outcome,
            error=error,
            is_final_attempt=is_final_attempt,
            attempt=attempt,
            retry_attempts=retry_attempts,
        )


def _publish_embedding_event(
    *,
    provider: str,
    model: str,
    input_count: int,
    input_tokens_estimate: int,
    duration_ms: float,
    outcome: Literal["success", "error", "cancelled"],
    error: BaseException | None,
    is_final_attempt: bool,
    attempt: int = 1,
    retry_attempts: int = 1,
) -> None:
    """Build and emit the EmbeddingCallCompletedEvent. Best-effort."""
    try:
        from src.telemetry.events import (
            EmbeddingCallCompletedEvent,
            EmbeddingCallPurpose,
            emit,
        )
        from src.utils.types import (
            get_embedding_call_purpose,
            get_embedding_parent_category,
            get_embedding_run_id,
            get_embedding_session_id,
            get_embedding_workspace_name,
        )

        # call_purpose travels via ContextVar so embedding callers don't have
        # to thread it through every call site. Unknown values drop to None
        # rather than raising — keeps telemetry resilient to drift.
        purpose_slug = get_embedding_call_purpose()
        call_purpose: EmbeddingCallPurpose | None = None
        if purpose_slug:
            try:
                call_purpose = EmbeddingCallPurpose(purpose_slug)
            except ValueError:
                logger.debug(
                    "Unknown embedding_call_purpose=%r; emitting without",
                    purpose_slug,
                )

        emit(
            EmbeddingCallCompletedEvent(
                workspace_name=get_embedding_workspace_name(),
                call_purpose=call_purpose,
                parent_category=get_embedding_parent_category(),
                provider=provider,
                model=model,
                input_count=input_count,
                input_tokens_estimate=input_tokens_estimate,
                duration_ms=duration_ms,
                outcome=outcome,
                is_final_attempt=is_final_attempt,
                error_class=type(error).__name__ if error is not None else None,
                run_id=get_embedding_run_id(),
            )
        )

        # Trace stream (ground-truth) — gated on payload tracing. Each embedding
        # gets its own span nested under the driving agent run (parent_span_id =
        # run_id), so multiple embeddings in one run don't share a span id.
        if settings.TELEMETRY.TRACE_PAYLOADS_ENABLED:
            from src.telemetry.events import EmbeddingCallTracedEvent, emit_trace

            run_id = get_embedding_run_id()
            span_id = generate_nanoid()
            emit_trace(
                EmbeddingCallTracedEvent(
                    trace_id=run_id or span_id,
                    span_id=span_id,
                    parent_span_id=run_id,
                    session_id=get_embedding_session_id(),
                    workspace_name=get_embedding_workspace_name(),
                    run_id=run_id,
                    attempt=attempt,
                    retry_attempts=retry_attempts,
                    is_final_attempt=is_final_attempt,
                    duration_ms=duration_ms,
                    outcome=outcome,
                    error_class=type(error).__name__ if error is not None else None,
                    call_purpose=purpose_slug,
                    parent_category=get_embedding_parent_category(),
                    provider=provider,
                    model=model,
                    provider_input_tokens=input_tokens_estimate,
                    provider_output_tokens=0,
                    input_count=input_count,
                )
            )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit EmbeddingCallCompletedEvent", exc_info=True)


class EmbeddingTokenLimitError(ValueError):
    """Raised when input text genuinely exceeds the model's token limit.

    Subclasses ``ValueError`` so existing broad handlers keep working, while
    letting callers tell a real "content too long" condition apart from a
    transient provider or configuration failure (dimension mismatch, empty
    response, upstream error). Only the pre-flight token checks raise this;
    provider failures keep raising plain ``ValueError``.
    """


class BatchItem(NamedTuple):
    """A single item in a batch with its metadata.

    `text` is the raw text, never the templated form: chunking and
    `prepare_chunks` deal in raw text, and the template is applied once at the
    provider-call boundary, so persisted chunks never contain template text.
    `token_count` is what the provider receives, template included, so the
    per-request token cap and telemetry reflect the real payload.
    """

    text: str
    text_id: str
    chunk_index: int
    token_count: int


class _EmbeddingClient:
    """
    Embedding client supporting OpenAI and Gemini with chunking and batching support.
    """

    def __init__(
        self,
        config: EmbeddingModelConfig,
        *,
        vector_dimensions: int,
        max_input_tokens: int,
        max_tokens_per_request: int,
        send_dimensions: bool,
        encoding_format: EmbeddingEncodingFormat = "float",
    ):
        self.transport: str = config.transport
        self.model: str = config.model
        self.vector_dimensions: int = vector_dimensions
        self.send_dimensions: bool = send_dimensions
        self.encoding_format: EmbeddingEncodingFormat = encoding_format

        if self.transport == "gemini":
            if not config.api_key:
                raise ValueError("Gemini API key is required")
            from google import genai
            from google.genai import types as genai_types

            # Default 10-minute HTTP timeout matches the LLM registry Gemini client.
            timeout_ms = (
                int(config.timeout * 1000) if config.timeout is not None else 600_000
            )
            http_options = genai_types.HttpOptions(
                base_url=config.base_url,
                timeout=timeout_ms,
            )
            self.client: genai.Client | AsyncOpenAI = genai.Client(
                api_key=config.api_key,
                http_options=http_options,
            )
            # Gemini's embedding models have model-specific input token caps
            # (shared across all modalities):
            #   - gemini-embedding-001: 2048 tokens
            #   - gemini-embedding-2 (and its -preview): 8192 tokens
            # Unknown models default conservatively to 2048.
            model_id = self.model.removeprefix("models/")
            gemini_model_token_cap = (
                8192
                if model_id in {"gemini-embedding-2", "gemini-embedding-2-preview"}
                else 2048
            )
            self.max_embedding_tokens: int = min(
                max_input_tokens, gemini_model_token_cap
            )
            # Gemini batch size is not documented, using conservative estimate
            self.max_batch_size: int = config.max_batch_size or 100
        else:  # openai
            if not config.api_key:
                raise ValueError("OpenAI API key is required")
            from openai import AsyncOpenAI

            # Omit timeout when unset so the OpenAI SDK keeps its own default.
            client_kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "base_url": config.base_url,
            }
            if config.timeout is not None:
                client_kwargs["timeout"] = config.timeout
            self.client = AsyncOpenAI(**client_kwargs)
            self.max_embedding_tokens = max_input_tokens
            self.max_batch_size = config.max_batch_size or 2048

        try:
            self.encoding: tiktoken.Encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_embedding_tokens_per_request: int = max_tokens_per_request

        # (prefix, suffix) wrapped around every input at the provider-call
        # boundary. Models that expect a chat template and use last-token
        # pooling (Qwen3-Embedding, Qwen3-VL-Embedding) need both halves; a
        # prefix alone leaves the pooled token in the wrong place.
        self._wrappers: dict[InputType, tuple[str, str]] = {
            "query": (config.query_prefix, config.query_suffix),
            "document": (config.document_prefix, config.document_suffix),
        }
        # The template is sent to the provider but is not part of the text being
        # chunked or truncated, so its tokens are reserved out of the per-input
        # budget; otherwise a text that only just fits would overflow once
        # wrapped and the provider would cut off the suffix.
        self._wrapper_tokens: dict[InputType, int] = {
            input_type: self._count_template_tokens(prefix)
            + self._count_template_tokens(suffix)
            for input_type, (prefix, suffix) in self._wrappers.items()
        }
        for input_type, overhead in self._wrapper_tokens.items():
            if self._budget(input_type) <= 0:
                raise ValueError(
                    f"Embedding {input_type} template uses {overhead} tokens, "
                    + "leaving no room for input within the embedding token "
                    + f"limit of {self.max_embedding_tokens}. Shorten "
                    + f"{input_type}_prefix/{input_type}_suffix or raise "
                    + "EMBEDDING_MAX_INPUT_TOKENS."
                )

    def _count_template_tokens(self, template: str) -> int:
        # Templates may carry special-token markers such as `<|im_start|>`;
        # count them as ordinary text rather than letting tiktoken reject them.
        if not template:
            return 0
        return len(self.encoding.encode(template, disallowed_special=()))

    def _budget(self, input_type: InputType) -> int:
        """Per-input token budget net of the template for `input_type`."""
        return self.max_embedding_tokens - self._wrapper_tokens[input_type]

    def _wrap(self, texts: list[str], input_type: InputType) -> list[str]:
        """Apply the configured template; a no-op when none is configured."""
        prefix, suffix = self._wrappers[input_type]
        if not prefix and not suffix:
            return texts
        return [prefix + text + suffix for text in texts]

    def _limit_message(self, input_type: InputType) -> str:
        budget = self._budget(input_type)
        overhead = self._wrapper_tokens[input_type]
        if not overhead:
            return f"maximum token limit of {budget} tokens"
        return (
            f"maximum token limit of {budget} tokens "
            + f"({self.max_embedding_tokens} minus {overhead} reserved for the "
            + f"{input_type} template)"
        )

    @property
    def provider(self) -> str:
        return self.transport

    def _validate_embedding_dimensions(self, embedding: list[float]) -> list[float]:
        if len(embedding) != self.vector_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for {self.transport}:{self.model}. "
                + f"Expected {self.vector_dimensions}, got {len(embedding)}."
            )
        return embedding

    def _apply_encoding_format(self, openai_kwargs: dict[str, Any]) -> None:
        """Set the embedding wire format on an openai request.

        Base64 is requested by omission, not by name: the SDK injects
        `encoding_format=base64` when the caller passes nothing and decodes the
        response, but skips that decode for any format the caller names, handing
        back the raw base64 string.
        """
        if self.encoding_format != "base64":
            openai_kwargs["encoding_format"] = self.encoding_format

    def _validate_embedding_count(self, expected: int, received: int) -> None:
        """Guard against a 200 response whose embedding count differs from inputs.

        An explicit `encoding_format` disables the openai SDK's own empty-data
        check, so this has to live here.
        """
        if received != expected:
            raise ValueError(
                f"Embedding count mismatch for {self.transport}:{self.model}. "
                + f"Expected {expected}, got {received}."
            )

    async def embed(
        self, query: str, *, input_type: InputType = "query"
    ) -> list[float]:
        """Embed a single text.

        Defaults to `input_type="query"` because that is what nearly every
        caller wants; pass `"document"` when embedding stored content through
        this path (for example a single-item fallback around
        `simple_batch_embed`), so both paths land in the same vector space.
        """
        token_count = len(self.encoding.encode(query))

        if token_count > self._budget(input_type):
            raise EmbeddingTokenLimitError(
                f"Query exceeds {self._limit_message(input_type)} (got {token_count} tokens)"
            )

        wrapped = self._wrap([query], input_type)[0]
        wrapped_token_count = token_count + self._wrapper_tokens[input_type]

        # Dispatch on transport rather than isinstance so this module never
        # needs the SDK types at runtime; the cast gives the closures a typed
        # local to close over.
        if self.transport == "gemini":
            gemini_client = cast("genai.Client", self.client)

            async def _call_gemini() -> list[float]:
                # The SDK's contents union includes optional Pillow types, which
                # are unresolved without Pillow; this call only sends text.
                response = await gemini_client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                    model=self.model,
                    contents=wrapped,
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if not response.embeddings or not response.embeddings[0].values:
                    raise ValueError("No embedding returned from Gemini API")
                return self._validate_embedding_dimensions(
                    response.embeddings[0].values
                )

            return await _emit_embedding_call(
                provider=self.transport,
                model=self.model,
                texts=[query],
                input_tokens_estimate=wrapped_token_count,
                fn=_call_gemini,
            )

        openai_client = cast("AsyncOpenAI", self.client)

        async def _call_openai() -> list[float]:
            openai_kwargs: dict[str, Any] = {"model": self.model, "input": [wrapped]}
            self._apply_encoding_format(openai_kwargs)
            if self.send_dimensions:
                openai_kwargs["dimensions"] = self.vector_dimensions
            response = await openai_client.embeddings.create(**openai_kwargs)
            self._validate_embedding_count(1, len(response.data))
            return self._validate_embedding_dimensions(response.data[0].embedding)

        return await _emit_embedding_call(
            provider=self.transport,
            model=self.model,
            texts=[query],
            input_tokens_estimate=wrapped_token_count,
            fn=_call_openai,
        )

    def truncate_to_token_limit(
        self, text: str, *, input_type: InputType = "query"
    ) -> tuple[str, int]:
        """Return a prefix of `text` whose re-encoded token count fits the cap.

        The cap is the budget net of the `input_type` template, so the template
        (in particular its suffix) still fits once applied. The returned text
        and count are raw, without the template.

        Decode/re-encode after slicing: BPE boundaries can re-expand past the cap.
        """
        budget = self._budget(input_type)
        token_ids = self.encoding.encode(text)
        keep = budget
        while len(token_ids) > budget:
            keep = min(keep, len(token_ids) - 1)
            if keep < 1:
                return "", 0
            text = self.encoding.decode(token_ids[:keep])
            token_ids = self.encoding.encode(text)
            keep -= 1
        return text, len(token_ids)

    async def simple_batch_embed(
        self,
        texts: list[str],
        *,
        on_oversize: Literal["raise", "truncate"] = "raise",
        input_type: InputType = "document",
    ) -> list[list[float]]:
        """
        Batch-embed a list of text strings. Does not sub-chunk oversized inputs.

        Internally goes through the same token-aware batching pipeline as
        `batch_embed()` so the per-request token cap is respected.

        Args:
            texts: List of text strings to embed
            on_oversize: ``"raise"`` (default) errors; ``"truncate"`` embeds a
                token-capped prefix.
            input_type: Which side of a retrieval pair these texts are. Defaults
                to "document"; pass "query" when batching search queries, so
                they get the same treatment as a single `embed()` call.

        Returns:
            List of embedding vectors, one per input text (in order)

        Raises:
            EmbeddingTokenLimitError: If any text exceeds token limits and
                `on_oversize` is ``"raise"``
        """
        if not texts:
            return []

        # Validate / cap per-input token limit and collect counts for batching
        budget = self._budget(input_type)
        prepared_texts: list[str] = []
        token_counts: list[int] = []
        for idx, text in enumerate(texts):
            token_ids = self.encoding.encode(text)
            if len(token_ids) > budget:
                if on_oversize == "truncate":
                    original_count = len(token_ids)
                    text, tokens = self.truncate_to_token_limit(
                        text, input_type=input_type
                    )
                    logger.warning(
                        "truncated oversize embedding input at idx %d: %d->%d tokens",
                        idx,
                        original_count,
                        tokens,
                    )
                else:
                    raise EmbeddingTokenLimitError(
                        f"Text at index {idx} exceeds "
                        + f"{self._limit_message(input_type)} (got {len(token_ids)} tokens)"
                    )
            else:
                tokens = len(token_ids)
            prepared_texts.append(text)
            token_counts.append(tokens)

        # Use positional indices as text_ids so we can reassemble in input order.
        text_chunks: dict[str, list[tuple[str, int]]] = {
            str(i): [(prepared_texts[i], token_counts[i])]
            for i in range(len(prepared_texts))
        }

        batches = self._create_batches(text_chunks, input_type=input_type)
        batch_results = await asyncio.gather(
            *[self._process_batch(batch, input_type=input_type) for batch in batches],
        )

        combined: dict[str, list[list[float]]] = self._accumulate_embeddings(
            batch_results
        )
        return [combined[str(i)][0] for i in range(len(texts))]

    def prepare_chunks(
        self, id_resource_dict: dict[str, str], *, input_type: InputType = "document"
    ) -> dict[str, list[str]]:
        """
        Public helper: tokenize and chunk texts using the same rules as
        `batch_embed()`. Returns ordered chunk texts per input id.

        Intended for callers that want to persist embeddable chunks
        before later embedding them off the request path. Chunks are sized to
        leave room for the `input_type` template but are returned without it;
        the template is applied when they are embedded.
        """
        return {
            text_id: [chunk_text for chunk_text, _ in chunks]
            for text_id, chunks in self._prepare_chunks(
                id_resource_dict, input_type=input_type
            ).items()
        }

    async def batch_embed(
        self, id_resource_dict: dict[str, str], *, input_type: InputType = "document"
    ) -> dict[str, list[list[float]]]:
        """
        Embed multiple texts, chunking long ones and batching API calls.

        Args:
            id_resource_dict: Maps text IDs to text content
            input_type: Which side of a retrieval pair these texts are.
                Defaults to "document", which is what this path is for.

        Returns:
            Maps text IDs to lists of embedding vectors (one per chunk)
        """
        if not id_resource_dict:
            return {}

        # 1. Prepare chunks for all texts if needed
        text_chunks = self._prepare_chunks(id_resource_dict, input_type=input_type)

        # 2. Create batches that fit API limits (max 2048 embeddings per request, max 300,000 tokens per request)
        batches = self._create_batches(text_chunks, input_type=input_type)

        # 3. Process all batches concurrently
        batch_results = await asyncio.gather(
            *[self._process_batch(batch, input_type=input_type) for batch in batches],
        )

        # 4. Accumulate results preserving chunk order
        return self._accumulate_embeddings(batch_results)

    def _prepare_chunks(
        self, id_resource_dict: dict[str, str], *, input_type: InputType = "document"
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Chunk texts that exceed token limits.

        Chunk sizes are computed against the budget net of the template, so
        every chunk still fits once wrapped at call time. The chunk text itself
        stays unwrapped.

        Args:
            id_resource_dict: Maps text IDs to text content. We tokenize with
                the embedding client's own encoding so token IDs match the
                decoder vocabulary used by the target embedding API.
            input_type: Which side of a retrieval pair these texts are; selects
                which template to reserve budget for.

        Returns:
            Maps text IDs to lists of (chunk_text, raw_token_count) tuples
        """
        budget = self._budget(input_type)
        out: dict[str, list[tuple[str, int]]] = {}
        for text_id, text in id_resource_dict.items():
            tokens = self.encoding.encode(text)
            if len(tokens) > budget:
                out[text_id] = _chunk_text_with_tokens(
                    text, tokens, budget, self.encoding
                )
            else:
                out[text_id] = [(text, len(tokens))]
        return out

    def _create_batches(
        self,
        text_chunks: dict[str, list[tuple[str, int]]],
        *,
        input_type: InputType = "document",
    ) -> list[list[BatchItem]]:
        """
        Group chunks into batches that fit API limits.

        Args:
            text_chunks: Maps text IDs to lists of (chunk_text, raw_token_count)
                tuples
            input_type: Selects the template whose tokens are added to each
                chunk, so the per-request cap counts what is actually sent

        Returns:
            List of batches, each containing BatchItem objects
        """
        overhead = self._wrapper_tokens[input_type]
        batches: list[list[BatchItem]] = []
        current_batch: list[BatchItem] = []
        current_tokens = 0

        for text_id, chunks in text_chunks.items():
            for chunk_idx, (chunk_text, raw_tokens) in enumerate(chunks):
                chunk_tokens = raw_tokens + overhead
                # Check if adding this chunk would exceed limits
                would_exceed_tokens = (
                    current_tokens + chunk_tokens
                    > self.max_embedding_tokens_per_request
                )
                would_exceed_count = len(current_batch) >= self.max_batch_size

                if current_batch and (would_exceed_tokens or would_exceed_count):
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                current_batch.append(
                    BatchItem(chunk_text, text_id, chunk_idx, chunk_tokens)
                )
                current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch(
        self,
        batch: list[BatchItem],
        max_retries: int = 3,
        *,
        input_type: InputType = "document",
    ) -> dict[str, dict[int, list[float]]]:
        """
        Process a single batch through the embeddings API with retry logic.

        Args:
            batch: List of BatchItem objects to embed
            max_retries: Maximum number of retry attempts (default: 3)
            input_type: Which template to wrap this batch's texts in

        Returns:
            Maps text IDs to {chunk_index: embedding_vector} dictionaries
        """
        last_exception: Exception | None = None
        wrapped_texts = self._wrap([item.text for item in batch], input_type)

        async def _call_provider() -> dict[str, dict[int, list[float]]]:
            """One provider call. Lifted out of the retry loop so
            _emit_embedding_call emits a separate event per attempt — each
            attempt is a distinct provider hit and shows up as its own line
            item in analytics."""
            result: dict[str, dict[int, list[float]]] = defaultdict(dict)
            if self.transport == "gemini":
                from google.genai import types as genai_types

                gemini_client = cast("genai.Client", self.client)
                # The SDK's contents union includes optional Pillow types, which
                # are unresolved without Pillow; this call only sends text.
                response = await gemini_client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                    model=self.model,
                    # One Content per item: a list of bare strings is folded
                    # into a single document by gemini-embedding-2*, which
                    # returns one embedding for the whole batch (#745).
                    contents=[
                        genai_types.Content(parts=[genai_types.Part(text=text)])
                        for text in wrapped_texts
                    ],
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if response.embeddings:
                    for item, embedding in zip(batch, response.embeddings, strict=True):
                        if embedding.values:
                            result[item.text_id][item.chunk_index] = (
                                self._validate_embedding_dimensions(embedding.values)
                            )
            else:  # openai
                openai_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": wrapped_texts,
                }
                self._apply_encoding_format(openai_kwargs)
                if self.send_dimensions:
                    openai_kwargs["dimensions"] = self.vector_dimensions
                openai_client = cast("AsyncOpenAI", self.client)
                response = await openai_client.embeddings.create(**openai_kwargs)
                self._validate_embedding_count(len(batch), len(response.data))
                for item, embedding_data in zip(batch, response.data, strict=True):
                    result[item.text_id][item.chunk_index] = (
                        self._validate_embedding_dimensions(embedding_data.embedding)
                    )
            return result

        # Token counts were computed during chunk prep; reuse them here so the
        # provider call doesn't re-encode every chunk just for the size proxy.
        batch_tokens_estimate = sum(item.token_count for item in batch)
        batch_texts = [item.text for item in batch]

        for attempt in range(max_retries):
            try:
                result = await _emit_embedding_call(
                    provider=self.transport,
                    model=self.model,
                    texts=batch_texts,
                    input_tokens_estimate=batch_tokens_estimate,
                    fn=_call_provider,
                    is_final_attempt=(attempt >= max_retries - 1),
                    attempt=attempt + 1,
                    retry_attempts=max_retries,
                )
                return dict(result)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    logger.warning(
                        f"Embedding batch failed (attempt {attempt + 1}/{max_retries}), "
                        + f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception("Error processing batch after all retries")

        raise last_exception or RuntimeError("Batch processing failed")

    def _accumulate_embeddings(
        self, batch_results: list[dict[str, dict[int, list[float]]]]
    ) -> dict[str, list[list[float]]]:
        """
        Combine batch results into final output, preserving chunk order.

        Args:
            batch_results: List of batch results from _process_batch

        Returns:
            Maps text IDs to ordered lists of embedding vectors
        """
        all_embeddings: dict[str, dict[int, list[float]]] = defaultdict(dict)

        # Collect all embeddings by text_id and chunk_index
        for batch_result in batch_results:
            for text_id, chunk_dict in batch_result.items():
                all_embeddings[text_id].update(chunk_dict)

        # Convert to ordered lists
        return {
            text_id: [chunk_dict[i] for i in sorted(chunk_dict.keys())]
            for text_id, chunk_dict in all_embeddings.items()
        }


def _chunk_text_with_tokens(
    text: str,
    encoded_tokens: list[int],
    max_tokens: int,
    encoding: tiktoken.Encoding,
) -> list[tuple[str, int]]:
    """
    Split text into chunks that fit within token limits, with 20% overlap.

    Args:
        text: Original text to chunk
        encoded_tokens: Pre-encoded tokens for the text
        max_tokens: Maximum tokens per chunk
        encoding: Tiktoken encoding model

    Returns:
        List of (chunk_text, token_count) tuples
    """
    if len(encoded_tokens) <= max_tokens:
        return [(text, len(encoded_tokens))]

    # Use 20% overlap for better semantic continuity
    overlap_tokens = int(max_tokens * 0.2)
    step_size = max_tokens - overlap_tokens

    return [
        (
            encoding.decode(encoded_tokens[i : i + max_tokens]),
            min(max_tokens, len(encoded_tokens) - i),
        )
        for i in range(0, len(encoded_tokens), step_size)
        if i < len(encoded_tokens)  # Ensure we don't create empty chunks
    ]


class EmbeddingClient:
    """
    Singleton wrapper for the embedding client with deferred loading.

    The actual client is only initialized on first use, improving startup time
    and allowing the application to start even if API keys are not yet configured.
    """

    _instance: _EmbeddingClient | None = None
    _instance_signature: tuple[object, ...] | None = None
    _lock: threading.Lock = threading.Lock()
    _wrapper_instance: EmbeddingClient | None = None

    def __new__(cls):
        """Ensure only one instance of EmbeddingClient exists."""
        # We always return the same wrapper instance
        if cls._wrapper_instance is None:
            cls._wrapper_instance = super().__new__(cls)
        return cls._wrapper_instance

    def _get_client(self) -> _EmbeddingClient:
        """
        Get or create the underlying embedding client instance.

        Uses double-checked locking for thread-safe lazy initialization.
        """
        signature = self._get_settings_signature()
        if self._instance is None or self._instance_signature != signature:
            with self._lock:
                if self._instance is None or self._instance_signature != signature:
                    runtime_config = self._resolve_runtime_config()
                    self._instance = _EmbeddingClient(
                        runtime_config,
                        vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
                        max_input_tokens=settings.EMBEDDING.MAX_INPUT_TOKENS,
                        max_tokens_per_request=settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
                        send_dimensions=settings.EMBEDDING.resolve_send_dimensions(),
                        encoding_format=settings.EMBEDDING.resolve_encoding_format(),
                    )
                    self._instance_signature = signature
                    logger.debug(
                        "Initialized embedding client with transport: %s model: %s",
                        runtime_config.transport,
                        runtime_config.model,
                    )

        return self._instance

    def _resolve_runtime_config(self) -> EmbeddingModelConfig:
        return resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)

    def _get_settings_signature(self) -> tuple[object, ...]:
        runtime_config = self._resolve_runtime_config()
        return (
            runtime_config.transport,
            runtime_config.model,
            runtime_config.api_key,
            runtime_config.base_url,
            runtime_config.max_batch_size,
            runtime_config.query_prefix,
            runtime_config.query_suffix,
            runtime_config.document_prefix,
            runtime_config.document_suffix,
            settings.EMBEDDING.VECTOR_DIMENSIONS,
            settings.EMBEDDING.MAX_INPUT_TOKENS,
            settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
            settings.EMBEDDING.resolve_send_dimensions(),
            settings.EMBEDDING.resolve_encoding_format(),
        )

    async def embed(
        self, query: str, *, input_type: InputType = "query"
    ) -> list[float]:
        """Embed a single string. Defaults to the query side."""
        return await self._get_client().embed(query, input_type=input_type)

    async def simple_batch_embed(
        self,
        texts: list[str],
        *,
        on_oversize: Literal["raise", "truncate"] = "raise",
        input_type: InputType = "document",
    ) -> list[list[float]]:
        """Batch embed a list of text strings (each must fit token limit)."""
        return await self._get_client().simple_batch_embed(
            texts, on_oversize=on_oversize, input_type=input_type
        )

    def prepare_chunks(
        self, id_resource_dict: dict[str, str], *, input_type: InputType = "document"
    ) -> dict[str, list[str]]:
        """Chunk texts using the same rules as `batch_embed` (no network)."""
        return self._get_client().prepare_chunks(
            id_resource_dict, input_type=input_type
        )

    def truncate_to_token_limit(
        self, text: str, *, input_type: InputType = "query"
    ) -> str:
        """Truncate text to the embedding token cap (no network)."""
        return self._get_client().truncate_to_token_limit(text, input_type=input_type)[
            0
        ]

    async def batch_embed(
        self, id_resource_dict: dict[str, str], *, input_type: InputType = "document"
    ) -> dict[str, list[list[float]]]:
        """Embed multiple texts, chunking long ones and batching API calls."""
        return await self._get_client().batch_embed(
            id_resource_dict, input_type=input_type
        )

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._get_client().provider

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._get_client().model

    @property
    def transport(self) -> str:
        """Get the transport name."""
        return self._get_client().transport

    @property
    def max_embedding_tokens(self) -> int:
        """Get the maximum embedding tokens."""
        return self._get_client().max_embedding_tokens

    @property
    def vector_dimensions(self) -> int:
        """Get the configured embedding dimensions."""
        return self._get_client().vector_dimensions

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Get the tiktoken encoding.

        Resolved without constructing the underlying client: tiktoken needs no
        API key, and token-counting callers (e.g. the document dedup tie-break)
        must work in environments with no embedding credentials, such as CI for
        pull requests from forks.
        """
        if self._instance is not None:
            return self._instance.encoding
        try:
            return tiktoken.encoding_for_model(self._resolve_runtime_config().model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


# Shared singleton embedding client instance
embedding_client = EmbeddingClient()
