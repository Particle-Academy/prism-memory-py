"""Persistent context, semantic recall, and the vector-store contract."""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

__all__ = [
    "BinarySignature",
    "Durability",
    "ErrorCode",
    "InMemoryVectorStore",
    "MemoryError",
    "MemoryKind",
    "Provenance",
    "RecallSettings",
    "Recalled",
    "Recollection",
    "Vector",
    "VectorMatch",
    "VectorQuery",
    "VectorRecord",
    "VectorStore",
    "Weighting",
    "recall",
]


class ErrorCode(str, Enum):
    #: A vector was built from values that cannot be scored.
    INVALID_VECTOR = "invalid_vector"
    #: Two vectors of different lengths were compared.
    DIMENSION_MISMATCH = "dimension_mismatch"
    #: A record was written into a collection embedded by a different model.
    EMBEDDING_SPACE_MISMATCH = "embedding_space_mismatch"
    #: A store that reports itself volatile was configured for durable memory.
    UNSAFE_MEMORY_CONFIGURATION = "unsafe_memory_configuration"
    #: A record cannot be stored as given.
    UNSTORABLE_MEMORY = "unstorable_memory"


class MemoryError(Exception):
    def __init__(self, code: ErrorCode | str, message: str) -> None:
        super().__init__(message)
        self.code: str = code.value if isinstance(code, ErrorCode) else code
        self.message = message


class MemoryKind(str, Enum):
    """What a stored record IS.

    ONE case, and that is a deliberate statement rather than an unfinished enum.
    "What is stored -- messages, summaries, or facts?" is the central open design
    question in the reference's spec. Adding ``SUMMARY`` and ``FACT`` before it
    is answered would settle it QUIETLY: the cases would exist, something would
    populate them, and the decision would have been made by whoever built first
    rather than by anyone who weighed it.

    So this slice stores OBSERVATIONS -- text that was actually said, with its
    provenance -- which is the substrate all three candidate answers share.
    """

    OBSERVATION = "observation"


class Durability(str, Enum):
    """Whether a store's contents survive a deploy. Same distinction as the harness."""

    VOLATILE = "volatile"
    DURABLE = "durable"


# -- vectors -----------------------------------------------------------------


class Vector:
    """An embedding, in a form that survives a database round trip UNCHANGED.

    The storage format is base64 of IEEE 754 doubles, explicitly LITTLE-ENDIAN
    -- byte for byte what the PHP reference's ``pack('e*')`` produces. Not JSON
    and not float32, and both alternatives are rejected for reasons about
    correctness rather than taste:

    - **float32** loses roughly nine significant digits per component, so a
      vector written and read back scores differently against the same query
      than it did at write time. Nothing errors; the ranking just drifts, by an
      amount no test that only checks "did it come back" would notice.
    - **JSON** depends on the host's float formatting. A package whose stored
      numbers change because someone tuned a runtime setting is not storing
      numbers, it is storing opinions about them.

    Little-endian rather than machine order matters for the same reason: a row
    written on one architecture and read on another would come back
    byte-reversed. That is not hypothetical for a database, which is the one
    part of a system that routinely outlives the machine that wrote to it.

    Matching the reference exactly is what lets a PHP app and a Python or
    TypeScript one **read each other's rows**.
    """

    __slots__ = ("_sum_of_squares", "values")

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self._sum_of_squares: float | None = None

    @classmethod
    def of(cls, values: Sequence[Any]) -> Vector:
        """Build from untrusted numbers -- a provider response, a config, a test.

        VALIDATES EVERY COMPONENT. This is the write path and runs once per
        record, so the O(n) pass is affordable and the guarantee is worth
        having: a single NaN inside a stored vector makes every score computed
        against it NaN, and NaN comparisons are false, so the record silently
        stops being retrievable rather than failing.
        """
        if not values:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR, "A vector must have at least one component."
            )

        parsed: list[float] = []

        for index, value in enumerate(values):
            try:
                component = float(value)
            except (TypeError, ValueError) as error:
                raise MemoryError(
                    ErrorCode.INVALID_VECTOR, f"Component {index} is not a number."
                ) from error

            if not math.isfinite(component):
                raise MemoryError(
                    ErrorCode.INVALID_VECTOR,
                    f"Component {index} is not a finite number, so every score computed against "
                    "this vector would be NaN.",
                )

            parsed.append(component)

        vector = cls(parsed)

        # Scorability is asserted HERE, on the write path, not lazily at the
        # first comparison. The reference does the same and the difference is
        # not cosmetic: a degenerate vector that is only rejected when something
        # scores against it has already been WRITTEN to a shared store, and
        # every recall that later touches that row raises instead of returning
        # results. Failing at construction puts the error where the caller can
        # still do something about it -- it has the embedding, and it knows
        # which document produced it.
        vector._squares()

        return vector

    def to_storage(self) -> str:
        """The stored form: base64 of little-endian float64s."""
        packed = struct.pack(f"<{len(self.values)}d", *self.values)
        return base64.b64encode(packed).decode("ascii")

    @classmethod
    def from_storage(cls, encoded: str) -> Vector:
        raw = base64.b64decode(encoded)

        if not raw or len(raw) % 8 != 0:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR,
                f"A stored vector must be a whole number of 8-byte doubles; got {len(raw)} byte(s).",
            )

        return cls.of(list(struct.unpack(f"<{len(raw) // 8}d", raw)))

    def dimensions(self) -> int:
        return len(self.values)

    def magnitude(self) -> float:
        return math.sqrt(self._squares())

    def cosine(self, other: Vector) -> float:
        """Cosine similarity, clamped to [-1, 1].

        One square root rather than two: ``dot / sqrt(a2 * b2)`` instead of
        ``dot / (sqrt(a2) * sqrt(b2))``. In the hot path one recall scores
        hundreds of candidates against one query, and a square root is not free.
        """
        if len(self.values) != len(other.values):
            raise MemoryError(
                ErrorCode.DIMENSION_MISMATCH,
                f"Cannot compare a {self.dimensions()}-dimension vector with a "
                f"{other.dimensions()}-dimension one.",
            )

        dot: float = sum(a * b for a, b in zip(self.values, other.values, strict=True))
        product = self._squares() * other._squares()

        if math.isfinite(product):
            similarity = dot / math.sqrt(product)
        else:
            # Two vectors large enough that the product of their squared lengths
            # overflows. Rarer than it sounds and not impossible, and the
            # fallback is exact enough -- only the last bit is at stake, and
            # infinity would cost every bit.
            similarity = dot / (self.magnitude() * other.magnitude())

        # Clamped so the RANGE is a guarantee even where the last bit is not.
        return max(-1.0, min(1.0, similarity))

    def _squares(self) -> float:
        """Squared, because that is the form ``cosine`` wants. Cached: hot path."""
        if self._sum_of_squares is not None:
            return self._sum_of_squares

        total = 0.0
        for value in self.values:
            total += value * value

        # Two genuinely different failures. A sum that OVERFLOWED cannot be
        # scored at all; a sum of ZERO is a vector with no direction. Collapsing
        # them into one message would send whoever reads it looking in the wrong
        # place.
        if not math.isfinite(total):
            raise MemoryError(
                ErrorCode.INVALID_VECTOR,
                "This vector is too large to score: the sum of its squared components overflows.",
            )

        if total <= 0.0:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR,
                "A zero vector has no direction, so cosine similarity against it is undefined.",
            )

        self._sum_of_squares = total
        return total


class BinarySignature:
    """A compact stand-in for a vector, so recall need not read the vectors.

    The bottleneck in a database-backed search is not the arithmetic, it is the
    BYTES. Cosine over 1536 doubles is microseconds; the 12KB that vector
    occupies takes far longer to get out of the database. Scoring ten thousand
    memories means moving 120MB per recall, on every turn.

    So each vector also gets a signature: one bit per random hyperplane,
    recording which side of it the vector falls on. Two vectors pointing in
    similar directions agree on most bits, and the fraction they disagree on
    estimates the angle between them -- Hamming distance over ``bits`` is
    theta/pi. At 256 bits that is 32 bytes instead of 12KB, and enough to RANK.

    Ranking, not answering. The signature picks the candidates and the real
    vectors decide the order, so every score a caller sees is an exact cosine.
    """

    def __init__(self, dimensions: int, bits: int = 256, seed: int = 0x5EED) -> None:
        if dimensions < 1 or bits < 1 or bits % 8 != 0:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR,
                "A signature needs at least one dimension and a whole number of bytes of bits.",
            )

        # A DETERMINISTIC pseudo-random basis. The planes must be identical
        # everywhere the same collection is read, or two processes would compute
        # different signatures for the same vector and neither would be wrong --
        # the recall would simply stop finding things.
        state = seed & 0xFFFFFFFF

        def nxt() -> float:
            nonlocal state
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            return state / 0x100000000 - 0.5

        self._planes = [[nxt() for _ in range(dimensions)] for _ in range(bits)]

    @property
    def bits(self) -> int:
        return len(self._planes)

    def of(self, vector: Vector) -> str:
        """One bit per hyperplane, packed into bytes and base64'd."""
        packed = bytearray(len(self._planes) // 8)

        for index, plane in enumerate(self._planes):
            dot = sum(
                weight * (vector.values[component] if component < len(vector.values) else 0.0)
                for component, weight in enumerate(plane)
            )

            if dot >= 0:
                packed[index >> 3] |= 1 << (index % 8)

        return base64.b64encode(bytes(packed)).decode("ascii")

    @staticmethod
    def distance(left: str, right: str) -> int:
        """Hamming distance. Over ``bits``, this estimates theta/pi."""
        a = base64.b64decode(left)
        b = base64.b64decode(right)

        if len(a) != len(b):
            raise MemoryError(
                ErrorCode.DIMENSION_MISMATCH,
                "Two signatures of different lengths cannot be compared.",
            )

        return sum((x ^ y).bit_count() for x, y in zip(a, b, strict=True))


# -- ranking -----------------------------------------------------------------


class Weighting:
    """How much relevance and how much recency.

    Pure similarity retrieves the most SIMILAR memory, which is not the same
    thing as the most useful one. "What is my billing address?" is most similar
    to every previous time the address was discussed -- including the one from
    two years ago that has since been superseded. Similarity has no opinion
    about which of two matching memories is still true.

    Recency has the opposite failure: it retrieves the newest thing, related or
    not. Neither axis is right alone, and which mix is right depends on what the
    memory is FOR. So the caller weights them, and the default is relevance
    alone, because that is what someone expects from semantic recall.

    Weights are NORMALISED, so a score stays inside [-1, 1] whatever the mix.
    That is what keeps ``min_score`` meaning the same thing when the mix changes;
    without it, raising the recency weight would raise every score and quietly
    disable the caller's threshold.
    """

    DEFAULT_HALF_LIFE = 7 * 24 * 60 * 60

    def __init__(
        self, relevance: float = 1.0, recency: float = 0.0, half_life_seconds: int | None = None
    ) -> None:
        if relevance < 0 or recency < 0 or relevance + recency <= 0:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR,
                "Weights cannot be negative and at least one must be above zero.",
            )

        half_life = self.DEFAULT_HALF_LIFE if half_life_seconds is None else half_life_seconds

        if half_life <= 0:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR, "A half-life must be a positive number of seconds."
            )

        self.relevance = relevance
        self.recency = recency
        #: EXPONENTIAL rather than a cut-off. A cut-off makes a memory disappear
        #: the moment it crosses a boundary, which shows up as an agent that knew
        #: something yesterday and does not today, with nothing in between.
        self.half_life_seconds = half_life

    def decay(self, age_seconds: float) -> float:
        if age_seconds <= 0:
            return 1.0

        decayed: float = 2.0 ** (-age_seconds / self.half_life_seconds)
        return decayed

    def score(self, similarity: float, age_seconds: float) -> float:
        total = self.relevance + self.recency

        return (similarity * self.relevance + self.decay(age_seconds) * self.recency) / total

    def uses_recency(self) -> bool:
        return self.recency > 0.0


class RecallSettings:
    """The defaults a recall uses when the caller does not say.

    ``overfetch`` is the one worth explaining. Ranking that considers anything
    beyond raw similarity has to RESCORE, and rescoring the top 8 by similarity
    can only ever reorder those 8 -- a memory that is the seventieth most
    similar and was written an hour ago cannot win a recency-weighted ranking it
    was never entered into. So the store is asked for ``limit * overfetch``
    candidates and the weighting picks from those.
    """

    def __init__(
        self,
        limit: int = 8,
        overfetch: int = 8,
        min_score: float | None = None,
        weighting: Weighting | None = None,
    ) -> None:
        if limit < 1 or overfetch < 1:
            raise MemoryError(
                ErrorCode.INVALID_VECTOR, "limit and overfetch must both be at least 1."
            )

        self.limit = limit
        self.overfetch = overfetch
        self.min_score = min_score
        self.weighting = weighting if weighting is not None else Weighting()

    def candidate_budget(self) -> int:
        """How many candidates the store is asked for."""
        return self.limit * self.overfetch


# -- records -----------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    #: Where this came from -- a thread id, a document, a source system.
    source: str | None = None
    #: Who or what said it.
    author: str | None = None
    #: When it was said, as a unix timestamp in seconds.
    observed_at: int | None = None


@dataclass
class VectorRecord:
    id: str
    collection: str
    content: str
    created_at: int
    kind: MemoryKind = MemoryKind.OBSERVATION
    #: None until something embeds it. See ``unembedded()``.
    vector: Vector | None = None
    #: The model that produced the vector. Two spaces must never be mixed.
    embedding_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True)
class VectorMatch:
    record: VectorRecord
    #: The EXACT cosine, never a signature estimate.
    similarity: float


@dataclass(frozen=True)
class Recalled:
    record: VectorRecord
    similarity: float
    #: Similarity and recency, combined by the weighting.
    score: float


@dataclass(frozen=True)
class Recollection:
    memories: list[Recalled]
    #: How many the store returned before reranking. Useful for tuning overfetch.
    candidates: int


@dataclass(frozen=True)
class VectorQuery:
    collections: list[str]
    vector: Vector
    #: What the STORE returns, not what the caller sees. See ``RecallSettings``.
    limit: int = 64
    #: Equality on metadata keys; a list means "any of".
    filter: dict[str, Any] = field(default_factory=dict)
    #: Applied by the store, before any reranking.
    min_similarity: float | None = None


class VectorStore(Protocol):
    def upsert(self, records: Iterable[VectorRecord]) -> None: ...
    def search(self, query: VectorQuery) -> list[VectorMatch]: ...
    def forget(self, collection: str, record_ids: Sequence[str]) -> int: ...
    def purge(self, collection: str) -> int: ...
    def purge_observed_before(self, collection: str, before: int) -> int: ...
    def count(self, collection: str, embedded_only: bool = False) -> int: ...
    def unembedded(self, collection: str, limit: int = 100) -> list[VectorRecord]: ...
    def durability(self) -> Durability: ...


class InMemoryVectorStore:
    """A store in this process's memory. VOLATILE, and it says so.

    Right for a test or a single-process tool. A real deployment points at a
    database or a vector database; this exists so the package WORKS ON INSTALL
    rather than requiring infrastructure before the first memory can be written.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], VectorRecord] = {}

    def durability(self) -> Durability:
        return Durability.VOLATILE

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        for record in records:
            # The space guard. Two vectors from different models are not
            # comparable -- the numbers are in different spaces -- and mixing
            # them produces similarities that look plausible and mean nothing.
            for existing in self._records.values():
                if (
                    existing.collection == record.collection
                    and existing.embedding_model is not None
                    and record.embedding_model is not None
                    and existing.embedding_model != record.embedding_model
                ):
                    raise MemoryError(
                        ErrorCode.EMBEDDING_SPACE_MISMATCH,
                        f"Collection [{record.collection}] is embedded by "
                        f"[{existing.embedding_model}]; refusing to mix in "
                        f"[{record.embedding_model}]. Vectors from two models are not comparable, "
                        "and mixing them produces similarities that look plausible and mean "
                        "nothing.",
                    )

            self._records[(record.collection, record.id)] = record

    def search(self, query: VectorQuery) -> list[VectorMatch]:
        matches: list[VectorMatch] = []

        for record in self._records.values():
            if record.collection not in query.collections or record.vector is None:
                continue
            if not _matches_filter(record, query.filter):
                continue

            similarity = query.vector.cosine(record.vector)

            if query.min_similarity is not None and similarity < query.min_similarity:
                continue

            matches.append(VectorMatch(record=record, similarity=similarity))

        matches.sort(key=lambda match: match.similarity, reverse=True)
        return matches[: query.limit]

    def forget(self, collection: str, record_ids: Sequence[str]) -> int:
        removed = 0
        for record_id in record_ids:
            if self._records.pop((collection, record_id), None) is not None:
                removed += 1
        return removed

    def purge(self, collection: str) -> int:
        return self._remove_where(lambda record: record.collection == collection)

    def purge_observed_before(self, collection: str, before: int) -> int:
        return self._remove_where(
            lambda record: (
                record.collection == collection
                and (record.provenance.observed_at or record.created_at) < before
            )
        )

    def count(self, collection: str, embedded_only: bool = False) -> int:
        return sum(
            1
            for record in self._records.values()
            if record.collection == collection and (not embedded_only or record.vector is not None)
        )

    def unembedded(self, collection: str, limit: int = 100) -> list[VectorRecord]:
        return [
            record
            for record in self._records.values()
            if record.collection == collection and record.vector is None
        ][:limit]

    def _remove_where(self, predicate: Any) -> int:
        doomed = [key for key, record in self._records.items() if predicate(record)]
        for key in doomed:
            del self._records[key]
        return len(doomed)


def _matches_filter(record: VectorRecord, filter: dict[str, Any]) -> bool:
    for key, wanted in filter.items():
        actual = record.metadata.get(key)

        if isinstance(wanted, (list, tuple)):
            if actual not in wanted:
                return False
            continue

        if actual != wanted:
            return False

    return True


def recall(
    store: VectorStore,
    collections: Sequence[str],
    vector: Vector,
    settings: RecallSettings | None = None,
    now: int = 0,
    filter: dict[str, Any] | None = None,
    min_similarity: float | None = None,
) -> Recollection:
    """Over-fetch by similarity, then rescore with the weighting.

    The two-stage shape is the whole point -- see ``RecallSettings.overfetch``.
    """
    settings = settings if settings is not None else RecallSettings()

    candidates = store.search(
        VectorQuery(
            collections=list(collections),
            vector=vector,
            limit=settings.candidate_budget(),
            filter=filter or {},
            min_similarity=min_similarity,
        )
    )

    scored = [
        Recalled(
            record=match.record,
            similarity=match.similarity,
            score=settings.weighting.score(
                match.similarity,
                now - (match.record.provenance.observed_at or match.record.created_at),
            ),
        )
        for match in candidates
    ]

    scored.sort(key=lambda match: match.score, reverse=True)

    if settings.min_score is not None:
        scored = [match for match in scored if match.score >= settings.min_score]

    return Recollection(memories=scored[: settings.limit], candidates=len(candidates))
