"""CI test shards (tools/ci/test_shard.py): every test in exactly one shard, modules kept whole, import failures kept."""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(_HERE), "tools", "ci")]

import test_shard as S  # noqa: E402


class Fake:
    """Stands in for a discovered test: sharding reads only its id."""

    def __init__(self, test_id: str):
        self._id = test_id

    def id(self):
        return self._id


def fakes(*counts: int) -> list:
    return [Fake(f"test_m{m}.C.test_{i}") for m, n in enumerate(counts) for i in range(n)]


class ShardTests(unittest.TestCase):
    def test_every_test_lands_in_exactly_one_shard(self):
        tests = fakes(9, 1, 4, 4, 7, 2, 3)
        parts = [S.shard(tests, i, 4) for i in range(4)]
        ids = [t.id() for part in parts for t in part]
        self.assertEqual(sorted(ids), sorted(t.id() for t in tests))

    def test_a_module_stays_in_one_shard(self):
        tests = fakes(5, 5, 5)
        for i in range(3):
            self.assertEqual(len({S.module(t) for t in S.shard(tests, i, 3)}), 1)

    def test_shards_get_similar_test_counts(self):
        sizes = [len(S.shard(fakes(8, 6, 5, 4, 3, 2, 2, 1, 1), i, 4)) for i in range(4)]
        self.assertLessEqual(max(sizes) - min(sizes), 2)

    def test_a_module_that_fails_to_import_is_its_own_module(self):
        failed = Fake("unittest.loader._FailedTest.test_broken")
        self.assertEqual(S.module(failed), "test_broken")
        parts = [S.shard(fakes(3, 3) + [failed], i, 2) for i in range(2)]
        self.assertEqual(sum(failed in part for part in parts), 1)

    def test_the_same_discovery_always_gives_the_same_shards(self):
        tests = fakes(4, 4, 4, 4)
        forward = sorted(t.id() for t in S.shard(tests, 1, 4))
        self.assertEqual(forward, sorted(t.id() for t in S.shard(list(reversed(tests)), 1, 4)))


if __name__ == "__main__":
    unittest.main()
