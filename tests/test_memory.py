"""Mirrors prism-memory-ts/test/memory.test.ts."""

from __future__ import annotations

import base64
import math
from typing import Any

import pytest

from prism_memory import (
    BinarySignature,
    Durability,
    InMemoryVectorStore,
    MemoryError,
    Provenance,
    RecallSettings,
    Vector,
    VectorQuery,
    VectorRecord,
    Weighting,
    recall,
)


def a_record(**overrides: Any) -> VectorRecord:
    defaults: dict[str, Any] = {
        "id": "m-1",
        "collection": "default",
        "content": "something that was said",
        "created_at": 1000,
        "vector": Vector.of([1, 0, 0]),
        "embedding_model": "text-embedding-3-small",
        "metadata": {},
        "provenance": Provenance(observed_at=1000),
    }
    defaults.update(overrides)
    return VectorRecord(**defaults)


# -- storage -----------------------------------------------------------------


def test_round_trips_through_the_stored_form_exactly() -> None:
    # float32 would lose roughly nine significant digits per component, and the
    # ranking would drift by an amount no "did it come back" test would notice.
    values = [0.1, -0.2, 1e-17, 12345.6789012345, math.pi]

    assert Vector.from_storage(Vector.of(values).to_storage()).values == values


def test_stores_little_endian_doubles_byte_for_byte() -> None:
    # The parity claim: a PHP app and this port read each other's rows.
    # `pack('e*')` is little-endian float64, and that is what this must produce.
    raw = base64.b64decode(Vector.of([1]).to_storage())

    assert len(raw) == 8
    assert list(raw) == [0, 0, 0, 0, 0, 0, 0xF0, 0x3F]


def test_refuses_a_non_finite_component_at_the_write_path() -> None:
    # A single NaN inside a stored vector makes every score against it NaN, and
    # NaN comparisons are false -- so the record silently stops being retrievable
    # rather than failing.
    with pytest.raises(MemoryError, match="NaN"):
        Vector.of([1, float("nan"), 3])

    with pytest.raises(MemoryError):
        Vector.of([1, float("inf")])


def test_refuses_an_empty_vector() -> None:
    with pytest.raises(MemoryError, match="at least one component"):
        Vector.of([])


def test_refuses_a_stored_blob_that_is_not_whole_doubles() -> None:
    with pytest.raises(MemoryError, match="8-byte doubles"):
        Vector.from_storage(base64.b64encode(bytes([1, 2, 3])).decode())


def test_parses_numeric_strings() -> None:
    assert Vector.of(["0.5", "0.25"]).values == [0.5, 0.25]


# -- cosine ------------------------------------------------------------------


def test_cosine_is_one_for_same_direction_and_zero_for_orthogonal() -> None:
    assert Vector.of([1, 0]).cosine(Vector.of([2, 0])) == pytest.approx(1.0)
    assert Vector.of([1, 0]).cosine(Vector.of([0, 1])) == pytest.approx(0.0)
    assert Vector.of([1, 0]).cosine(Vector.of([-1, 0])) == pytest.approx(-1.0)


def test_stays_inside_the_range_even_where_floating_point_would_not() -> None:
    similarity = Vector.of([0.1, 0.2, 0.3]).cosine(Vector.of([0.1, 0.2, 0.3]))

    assert -1.0 <= similarity <= 1.0


def test_names_a_dimension_mismatch() -> None:
    with pytest.raises(MemoryError, match="dimension"):
        Vector.of([1, 2]).cosine(Vector.of([1, 2, 3]))


def test_refuses_a_zero_vector_which_has_no_direction() -> None:
    with pytest.raises(MemoryError, match="no direction"):
        Vector.of([0, 0]).cosine(Vector.of([1, 0]))


def test_survives_a_product_that_overflows_while_each_length_is_finite() -> None:
    # The case the fallback exists for: 1e150 squared is 1e300, so each vector's
    # own sum of squares is finite and their product is not.
    huge = Vector.of([1e150, 1e150])

    assert huge.cosine(huge) == pytest.approx(1.0)


def test_refuses_a_vector_too_large_to_score_at_all() -> None:
    # Two genuinely different failures. Collapsing them into one message would
    # send whoever reads it looking in the wrong place.
    with pytest.raises(MemoryError, match="overflows"):
        Vector.of([1e200, 1e200]).cosine(Vector.of([1, 1]))

    with pytest.raises(MemoryError, match="no direction"):
        Vector.of([0, 0]).cosine(Vector.of([1, 1]))


# -- signatures --------------------------------------------------------------


def test_a_signature_is_deterministic_for_the_same_seed() -> None:
    # The planes must be identical everywhere the same collection is read, or
    # two processes would compute different signatures for the same vector and
    # neither would be wrong -- recall would simply stop finding things.
    vector = Vector.of([0.3, -0.7, 0.1, 0.9])

    assert BinarySignature(4, 64).of(vector) == BinarySignature(4, 64).of(vector)


def test_similar_vectors_have_a_smaller_hamming_distance() -> None:
    # Hamming distance over `bits` estimates theta/pi: that is what makes the
    # signature enough to RANK candidates without reading the vectors.
    signature = BinarySignature(3, 256)
    query = Vector.of([1, 0, 0])

    near = BinarySignature.distance(signature.of(query), signature.of(Vector.of([0.99, 0.1, 0])))
    far = BinarySignature.distance(signature.of(query), signature.of(Vector.of([-1, 0, 0])))

    assert near < far


def test_packs_one_bit_per_plane() -> None:
    assert len(base64.b64decode(BinarySignature(3, 64).of(Vector.of([1, 1, 1])))) == 8


def test_refuses_a_bit_count_that_is_not_whole_bytes() -> None:
    with pytest.raises(MemoryError):
        BinarySignature(3, 100)


def test_refuses_to_compare_signatures_of_different_lengths() -> None:
    a = BinarySignature(3, 64).of(Vector.of([1, 0, 0]))
    b = BinarySignature(3, 128).of(Vector.of([1, 0, 0]))

    with pytest.raises(MemoryError, match="different lengths"):
        BinarySignature.distance(a, b)


# -- weighting ---------------------------------------------------------------


def test_defaults_to_relevance_alone() -> None:
    weighting = Weighting()

    assert weighting.uses_recency() is False
    assert weighting.score(0.8, 999_999) == pytest.approx(0.8)


def test_normalises_so_a_score_stays_in_range() -> None:
    # Without this, raising the recency weight would raise every score and
    # quietly disable the caller's min_score threshold.
    assert Weighting(1, 1).score(1, 0) == pytest.approx(1.0)
    assert Weighting(1, 9).score(1, 0) == pytest.approx(1.0)
    assert Weighting(1, 9).score(-1, 0) >= -1.0


def test_halves_the_recency_contribution_over_the_half_life() -> None:
    weighting = Weighting(0, 1, 100)

    assert weighting.decay(0) == pytest.approx(1.0)
    assert weighting.decay(100) == pytest.approx(0.5)
    assert weighting.decay(200) == pytest.approx(0.25)


def test_decays_rather_than_cutting_off() -> None:
    # A cut-off shows up as an agent that knew something yesterday and does not
    # today, with nothing in between.
    assert Weighting(0, 1, 100).decay(10_000) > 0


def test_refuses_weights_that_cannot_rank_anything() -> None:
    with pytest.raises(MemoryError):
        Weighting(0, 0)

    with pytest.raises(MemoryError):
        Weighting(-1, 2)

    with pytest.raises(MemoryError, match="half-life"):
        Weighting(1, 0, 0)


# -- the store ---------------------------------------------------------------


def test_the_store_reports_itself_volatile() -> None:
    assert InMemoryVectorStore().durability() is Durability.VOLATILE


def test_upserts_and_searches_by_similarity() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            a_record(id="a", vector=Vector.of([1, 0, 0])),
            a_record(id="b", vector=Vector.of([0, 1, 0])),
        ]
    )

    matches = store.search(
        VectorQuery(collections=["default"], vector=Vector.of([1, 0, 0]), limit=10)
    )

    assert matches[0].record.id == "a"
    assert matches[0].similarity == pytest.approx(1.0)


def test_refuses_to_mix_two_embedding_spaces_in_one_collection() -> None:
    # Vectors from two models are not comparable -- the numbers are in different
    # spaces -- and mixing them produces similarities that look plausible and
    # mean nothing.
    store = InMemoryVectorStore()
    store.upsert([a_record(id="a", embedding_model="model-one")])

    with pytest.raises(MemoryError) as caught:
        store.upsert([a_record(id="b", embedding_model="model-two")])

    assert caught.value.code == "embedding_space_mismatch"


def test_skips_unembedded_records_and_lists_them_for_an_embedder() -> None:
    store = InMemoryVectorStore()
    store.upsert([a_record(id="pending", vector=None, embedding_model=None)])

    assert store.search(VectorQuery(collections=["default"], vector=Vector.of([1, 0, 0]))) == []
    assert [record.id for record in store.unembedded("default")] == ["pending"]
    assert store.count("default") == 1
    assert store.count("default", embedded_only=True) == 0


def test_filters_on_metadata_and_a_list_means_any_of() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            a_record(id="a", metadata={"topic": "billing"}),
            a_record(id="b", metadata={"topic": "shipping"}),
            a_record(id="c", metadata={"topic": "returns"}),
        ]
    )

    one = store.search(
        VectorQuery(
            collections=["default"], vector=Vector.of([1, 0, 0]), filter={"topic": "billing"}
        )
    )
    many = store.search(
        VectorQuery(
            collections=["default"],
            vector=Vector.of([1, 0, 0]),
            filter={"topic": ["billing", "returns"]},
        )
    )

    assert [match.record.id for match in one] == ["a"]
    assert sorted(match.record.id for match in many) == ["a", "c"]


def test_forgets_purges_and_purges_by_observation_time() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            a_record(id="old", provenance=Provenance(observed_at=100)),
            a_record(id="new", provenance=Provenance(observed_at=9999)),
        ]
    )

    assert store.purge_observed_before("default", 1000) == 1
    assert store.forget("default", ["new"]) == 1
    assert store.count("default") == 0
    assert store.purge("default") == 0


def test_searches_several_collections_together() -> None:
    store = InMemoryVectorStore()
    store.upsert([a_record(id="a", collection="one"), a_record(id="b", collection="two")])

    matches = store.search(VectorQuery(collections=["one", "two"], vector=Vector.of([1, 0, 0])))

    assert len(matches) == 2


# -- recall ------------------------------------------------------------------


def test_over_fetches_so_recency_can_reorder_past_the_visible_limit() -> None:
    # Rescoring the top 8 by similarity can only ever reorder those 8. A memory
    # that is the seventieth most similar and was written an hour ago cannot win
    # a recency-weighted ranking it was never entered into.
    store = InMemoryVectorStore()
    store.upsert(
        [
            a_record(id="stale", vector=Vector.of([1, 0, 0]), provenance=Provenance(observed_at=0)),
            a_record(
                id="fresh",
                vector=Vector.of([0.9, 0.4, 0]),
                provenance=Provenance(observed_at=10_000),
            ),
        ]
    )

    by_relevance = recall(
        store,
        ["default"],
        Vector.of([1, 0, 0]),
        RecallSettings(limit=1, weighting=Weighting(1, 0)),
        now=10_000,
    )
    by_recency = recall(
        store,
        ["default"],
        Vector.of([1, 0, 0]),
        RecallSettings(limit=1, weighting=Weighting(1, 3, 3600)),
        now=10_000,
    )

    assert by_relevance.memories[0].record.id == "stale"
    assert by_recency.memories[0].record.id == "fresh"
    # Both saw the same candidate pool -- the ranking changed, not the fetch.
    assert by_relevance.candidates == 2


def test_asks_the_store_for_limit_times_overfetch_candidates() -> None:
    asked: list[int] = []

    class Spy(InMemoryVectorStore):
        def search(self, query: VectorQuery) -> list[Any]:
            asked.append(query.limit)
            return []

    recall(Spy(), ["default"], Vector.of([1, 0, 0]), RecallSettings(limit=4, overfetch=8))

    assert asked == [32]


def test_applies_min_score_after_reranking_not_before() -> None:
    # The threshold is about the score the caller sees. Applying it to raw
    # similarity would filter on a number the caller never asked about.
    store = InMemoryVectorStore()
    store.upsert([a_record(id="a", vector=Vector.of([0.5, 0.5, 0]))])

    strict = recall(
        store, ["default"], Vector.of([1, 0, 0]), RecallSettings(min_score=0.99), now=1000
    )

    assert strict.memories == []
    # It was still a candidate; it just did not clear the bar.
    assert strict.candidates == 1


def test_reports_how_many_candidates_it_considered() -> None:
    store = InMemoryVectorStore()
    store.upsert([a_record(id="a"), a_record(id="b")])

    result = recall(store, ["default"], Vector.of([1, 0, 0]))

    assert result.candidates == 2
    assert len(result.memories) == 2
