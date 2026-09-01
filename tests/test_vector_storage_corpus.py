"""The cross-language vector-storage corpus from ``prism-parity``.

A vector is written by whichever service embedded it and read back by whichever
service is recalling, so the bytes have to cross the language boundary intact. A
base64 string that decodes to different doubles elsewhere does not error -- it
silently scores wrong, and a recall that returns the wrong memory looks exactly
like one that returned a mediocre one.

This suite is ``full`` in all three languages, and it did not start that way: it
found that this port asserted scorability LAZILY, at the first comparison, where
the reference asserts it on the write path. Two rows disagreed. Fixed here
rather than recorded as a divergence, because unlike G-20 and G-21 it had a
right answer and the reference had it. See G-22.

Mirrors prism-memory-ts/test/vector-storage-corpus.test.ts case for case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prism_memory import MemoryError, Vector

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "memory-vector-storage.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = CORPUS["cases"]


def _id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _storage_of(case: dict[str, Any]) -> dict[str, Any]:
    try:
        vector = Vector.of(case["values"])
        packed = vector.to_storage()

        return {
            "refused": False,
            "packed": packed,
            "round_trips": list(Vector.from_storage(packed).values) == list(vector.values),
        }
    except MemoryError:
        return {"refused": True, "packed": None, "round_trips": None}


def test_the_corpus_is_whole_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CASES) == 9


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_stores_exactly_what_the_reference_stores(case: dict[str, Any]) -> None:
    assert _storage_of(case) == case["storage"]["php"]


def test_agrees_with_the_reference_on_every_row() -> None:
    # Stated as its own assertion so the suite's `full` status is a thing the
    # test claims rather than a thing the manifest asserts about it.
    assert [case for case in CASES if not case["agrees"]] == []


def test_refuses_a_degenerate_vector_at_the_write_path_not_at_the_first_score() -> None:
    # G-22. The difference is not cosmetic: a vector rejected only when something
    # scores against it has already been written to a shared store, and every
    # recall that later touches that row raises instead of returning results.
    with pytest.raises(MemoryError, match="no direction"):
        Vector.of([0.0, -0.0])

    with pytest.raises(MemoryError, match="too large to score"):
        Vector.of([1e-300, 1e300])


def test_round_trips_every_vector_it_accepts() -> None:
    # Asserted separately from the byte comparison: a pack that no longer
    # unpacks to its own input is invisible to a test that only ever compares
    # packs to packs.
    for case in CASES:
        if case["storage"]["php"]["refused"]:
            continue

        assert _storage_of(case)["round_trips"] is True, case["id"]
