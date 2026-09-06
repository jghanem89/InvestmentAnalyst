"""
Embedding backend selection.

Ollama keeps the whole stack local next to the chat model, but an Ollama server
without embedding support (or without the model pulled) fails at ingestion time
with a confusing 501. This module probes it once and can fall back to Chroma's
bundled MiniLM model so the pipeline still runs.
"""

from __future__ import annotations

from typing import Any, Tuple

from chromadb.utils import embedding_functions

from .config import RAGConfig

FALLBACK_SIGNATURE = "default:chroma-minilm-l6-v2"


def _probe(function: Any) -> Tuple[bool, str]:
    """Embed one short string to confirm the backend actually works."""
    try:
        vectors = function(["probe"])
    except Exception as exc:  # network, missing model, no embedding support
        return False, str(exc)
    if not vectors or not len(vectors[0]):
        return False, "backend returned an empty embedding"
    return True, ""


def get_embedding_function(config: RAGConfig | None = None) -> Tuple[Any, str]:
    """
    Build the embedding function for `config`.

    Returns the function and the signature naming the vector space it produces.
    The signature is stored on the collection so a later run cannot silently
    query MiniLM vectors with a nomic model, which would return nonsense.
    """
    config = config or RAGConfig()

    if config.embed_backend == "ollama":
        function = embedding_functions.OllamaEmbeddingFunction(
            url=config.ollama_url,
            model_name=config.embed_model,
            timeout=config.embed_timeout,
        )
        ok, error = _probe(function)
        if ok:
            return function, config.embed_signature

        message = (
            f"Ollama embeddings unavailable ({config.embed_model} at "
            f"{config.ollama_url}): {error}"
        )
        if not config.allow_embed_fallback:
            raise RuntimeError(
                f"{message}\nRun `ollama pull {config.embed_model}`, or set "
                "RAG_EMBED_BACKEND=default to use Chroma's bundled model."
            )
        print(f"[rag] {message}\n[rag] Falling back to Chroma's bundled MiniLM model.")

    function = embedding_functions.DefaultEmbeddingFunction()
    ok, error = _probe(function)
    if not ok:
        raise RuntimeError(f"No usable embedding backend. Bundled model failed: {error}")
    return function, FALLBACK_SIGNATURE
