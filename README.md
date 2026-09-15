# Prism Memory for Python

Persistent context, semantic recall and the vector-store contract. The Python
port of [`particle-academy/prism-memory`](https://github.com/Particle-Academy/prism-memory).

Zero runtime dependencies. Python 3.10+.

```
pip install prism-ai-memory
```

```python
from prism_memory import (
    InMemoryVectorStore,
    RecallSettings,
    Vector,
    VectorRecord,
    Weighting,
    recall,
)

store = InMemoryVectorStore()

store.upsert(
    [
        VectorRecord(
            id="m-1",
            collection="user:7",
            content="Billing address is 12 Harbour Road.",
            created_at=1_789_000_000,
            vector=Vector.of(embedding),
            embedding_model="text-embedding-3-small",
        ),
    ]
)

found = recall(
    store,
    ["user:7"],
    Vector.of(query_embedding),
    RecallSettings(limit=5, weighting=Weighting(relevance=0.8, recency=0.2)),
    now=1_789_100_000,
)

for memory in found.memories:
    print(memory.score, memory.record.content)
```

The package does not call an embedding model. Pass the vectors your provider
returns.

## Vectors

`Vector.of()` validates every component. An empty vector, a value that is not a
number, a NaN or infinity, a zero vector, or one too large to score is refused
with `MemoryError` code `invalid_vector`, when the vector is built rather than
when something first scores against it.

`to_storage()` and `from_storage()` use base64 of little-endian 64-bit floats,
byte for byte the format the PHP reference writes. A PHP, Python or TypeScript
application can read the others' rows.

## Recall

`recall()` asks the store for `limit × overfetch` candidates by similarity, then
rescores them with the `Weighting` and returns the top `limit`.

- `Weighting(relevance, recency, half_life_seconds)` mixes similarity with an
  exponential recency decay. The default is relevance alone. Weights are
  normalised, so `min_score` means the same thing whatever the mix.
- Every `similarity` a caller sees is an exact cosine.
- `filter` matches metadata keys by equality; a list means any of.

`BinarySignature` turns a vector into a short bit signature (256 bits by
default) for picking candidates without reading the full vectors. The planes are
deterministic, so every process computes the same signature for the same vector.

## Stores

`VectorStore` is a `Protocol`: `upsert`, `search`, `forget`, `purge`,
`purge_observed_before`, `count`, `unembedded` and `durability`.

`InMemoryVectorStore` keeps records in this process and reports itself
`volatile`. It refuses to mix vectors from two embedding models in one
collection, with code `embedding_space_mismatch`. For a real deployment,
implement `VectorStore` over your database and declare its `durability()`.

## Errors

Every failure is a `MemoryError` with a stable `code`: `invalid_vector`,
`dimension_mismatch`, `embedding_space_mismatch`,
`unsafe_memory_configuration`, `unstorable_memory`. Match on the code, not the
message.

## Parity

prism-parity's `memory-vector-storage` corpus pins the storage format and scoring
against the PHP reference and the TypeScript port.

## License

MIT. See [LICENSE](LICENSE).
