"""Vector store abstraction and Pinecone implementation.

Shared by ingestion (upsert) and retrieval (query). The concrete Pinecone
index is injectable so tests run without the SDK, a key, or the network.

Design notes:
- Vector id == chunk id, so re-ingesting identical chunks overwrites rather
  than duplicates (idempotent upsert).
- Chunk text is stored in vector metadata, so retrieval returns snippets
  without a separate document store.
- Deletion is by id (Pinecone serverless does not support delete-by-filter);
  the document store remembers each document's chunk ids.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from app.core.logging import get_logger
from app.models.document import Chunk

logger = get_logger(__name__)

_UPSERT_BATCH = 100


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    ordinal: int
    text: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def query(
        self, vector: list[float], *, top_k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    def delete(self, ids: list[str]) -> None: ...


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "document_id": chunk.document_id,
        "filename": chunk.metadata.filename,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
    }
    if chunk.metadata.page is not None:  # Pinecone rejects null metadata values
        meta["page"] = chunk.metadata.page
    return meta


class PineconeVectorStore:
    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
        client: Any | None = None,
        index: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._dimension = dimension
        self._cloud = cloud
        self._region = region
        self._client = client
        self._index = index

    def _get_client(self) -> Any:
        if self._client is None:
            from pinecone import Pinecone  # lazy

            self._client = Pinecone(api_key=self._api_key)
        return self._client

    def ensure_index(self) -> None:
        """Create the serverless index if it does not already exist."""
        if self._index is not None:
            return
        from pinecone import ServerlessSpec  # lazy

        client = self._get_client()
        existing = {ix["name"] for ix in client.list_indexes()}
        if self._index_name not in existing:
            logger.info("creating_index", index=self._index_name, dimension=self._dimension)
            client.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        self._index = client.Index(self._index_name)

    def _get_index(self) -> Any:
        if self._index is None:
            self.ensure_index()
        return self._index

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return
        index = self._get_index()
        payload = [
            {"id": c.id, "values": v, "metadata": _chunk_metadata(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(payload), _UPSERT_BATCH):
            index.upsert(vectors=payload[start : start + _UPSERT_BATCH])
        logger.info("upserted_vectors", count=len(payload), index=self._index_name)

    def query(
        self, vector: list[float], *, top_k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        index = self._get_index()
        response = index.query(
            vector=vector, top_k=top_k, include_metadata=True, filter=filter
        )
        matches = response["matches"] if isinstance(response, dict) else response.matches
        results: list[SearchResult] = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            mid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            results.append(
                SearchResult(
                    chunk_id=mid,
                    document_id=md["document_id"],
                    filename=md["filename"],
                    page=md.get("page"),
                    ordinal=int(md["ordinal"]),
                    text=md["text"],
                    score=float(score),
                )
            )
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._get_index().delete(ids=ids)
        logger.info("deleted_vectors", count=len(ids), index=self._index_name)
